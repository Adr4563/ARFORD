# -*- coding: utf-8 -*-
"""AEB -- Automatic Empathic Behavior: del animo de Romeo a su comportamiento.

Tercer componente de la capa afectiva de Rocha et al. (tesis UFF, sec. 3.4.3
y 4.1.3), despues de PULSE (que estima al usuario) y ROSE (que mantiene el
animo de Romeo, ver estado_afectivo.py):

    PULSE   -> como esta el usuario         (vector VAD su)
    ROSE    -> como se siente Romeo         (vector VAD sr)
    AEB     -> como lo EXPRESA Romeo        (esto)

Igual que en la tesis, el AEB NO ejecuta nada: solo traduce sr a un
diccionario de parametros de comportamiento. Quien los ejecuta es el
orquestador (el equivalente a SENSEI en ARFORD), que decide cuando
usarlos. Asi se puede, por ejemplo, no mover el carrito en mitad de una
trivia aunque el animo lo pida.

    p = mapear(estado)
    {
      "estado":     "felicidad_L2",   # emocion + nivel, para el log
      "emocion":    "felicidad",      # felicidad | tristeza | enojo | neutral
      "nivel":      2,                # 0 (neutral) .. 4
      "cara":       "happy",          # cara de REPOSO (FaceCommand)
      "tono":       "...",            # instruccion de estilo para el LLM, o None
      "movimiento": None,             # MotionCommand, o None
      "musica":     None,             # siempre None, ver abajo
    }

COMO SE CALCULA. La tesis mide la distancia euclidiana de sr a los
anclajes de cada emocion y a cuatro niveles intermedios (L1-L4). Aca se
hace lo mismo, pero respecto al ANIMO BASE y no al origen, por la misma
razon que estado_afectivo.cara(): la base de Romeo ya es positiva, asi que
en coordenadas absolutas estaria siempre "feliz". La emocion sale de la
DIRECCION en que el animo se aparto de la base, y el nivel de CUANTO se
aparto, repartido en cuatro tramos entre el radio neutro y el anclaje.

QUE NO HACE.
  - No toca las caras de la trivia ni del juego de caras: esas salen del
    dataset (cara_respuesta_buena/mala) y del propio juego, y comunican lo
    que esta pasando en el juego, no el animo.
  - No elige musica. Las canciones del robot son temas de videojuegos
    pensados para la trivia (Mario, Zelda...): sonar "algo triste" no tiene
    un archivo al que mapear. La clave existe para que el contrato quede
    completo y se pueda llenar cuando haya musica afectiva.
  - Por TRISTEZA no se mueve. Un robot que se aleja cuando le cuentan
    algo triste comunica rechazo justo cuando hay que acompañar;
    quedarse quieto es el gesto. Por ENOJO retrocede (L2+): dar un paso
    atras ante alguien alterado es leerle el espacio, no huir. Y por
    ALEGRIA avanza (L2+): se acerca para acompañar la buena energia.

Variables de entorno:
    ROMEO_AEB_MOVIMIENTO   0 desactiva el gesto fisico (default 1)
"""
import math
import os
import random
import re

from . import estado_afectivo

MOVIMIENTO_ACTIVO = os.environ.get("ROMEO_AEB_MOVIMIENTO", "1") not in ("0", "false", "False")

# Cuando Romeo se pone triste, NO mostrar la cara triste: dejar la cara
# normal (content). El tono y el movimiento siguen como el nivel mande --
# solo cambia la cara de reposo. Por defecto activo (la cara triste no se
# muestra). ROMEO_CARA_TRISTE=1 vuelve a mostrarla.
OCULTAR_CARA_TRISTE = os.environ.get(
    "ROMEO_CARA_TRISTE", "0") not in ("1", "true", "True")

# Cara de reposo -> nombre de la emocion (claves de ANCLAJES).
_EMOCION_DE_CARA = {"happy": "felicidad", "sad": "tristeza", "angry": "enojo"}

# Instruccion para el LLM por emocion y nivel (1-2 suave, 3-4 intenso). El
# animo de Romeo sigue al del usuario (empatia), asi que la emocion es
# tambien la de la persona, y la instruccion dice COMO reaccionar a ella:
#
#   felicidad -> apoyarla y alegrarse con ella, que se sienta mas feliz
#   tristeza  -> reconocer lo que siente, luego "No te preocupes, esto va a
#                pasar" y seguir la conversacion; en L3-4 ademas "aqui
#                estoy contigo" y proponerle algo que la alegre
#   enojo     -> ponerse de su lado en lo que siente: "Es normal que estes
#                molesto, te entiendo", sin regañarla ni pedirle que se calme
# Por emocion, (nivel 1-2, nivel 3-4). Dicen QUE lograr, no que frase
# decir: con frases de ejemplo entre comillas el modelo las copiaba
# palabra por palabra y la respuesta era nuestra, no suya. Responde a su
# juicio y con sus palabras (ver TEMPERATURA en agente_chat).
#
# Primero se reconoce lo que siente y DESPUES se anima: al reves suena a
# "no deberias sentirte asi". Con el enojo se apoya lo que SIENTE, no lo
# que vaya a hacer. Los casos graves (hacerse daño, que la lastimen,
# lastimar a otro) no pasan por aqui: ver seguridad.py.
_NATURAL = (" Usa tus propias palabras y responde a lo que te conto, sin "
            "frases hechas.")
_TONOS = {
    "felicidad": (
        ("La persona esta contenta: apoyala y alegrate con ella. Celebra lo "
         "que te cuenta y haz que se sienta todavia mas feliz." + _NATURAL,),
        ("La persona esta muy feliz: comparte su alegria con entusiasmo, "
         "felicitala, dile lo bien que hizo y pidele que te cuente mas para "
         "que lo siga disfrutando." + _NATURAL,),
    ),
    "tristeza": (
        # "Sin hacer ninguna pregunta" porque el prompt de base le permite
        # una, y gemma3:4b la hacia en lugar de dar animo.
        ("La persona esta triste. Responde en dos frases y sin hacer ninguna "
         "pregunta. En la primera muestra que entiendes como se siente. En "
         "la segunda dale animo: dile que esto va a mejorar o que puede "
         "salir adelante." + _NATURAL,),
        ("La persona esta muy triste. Responde en dos frases y sin hacer "
         "ninguna pregunta. En la primera acompañala mostrando que entiendes "
         "lo que siente. En la segunda dale animo con algo que tenga que ver "
         "con lo que te conto." + _NATURAL,),
    ),
    "enojo": (
        ("La persona esta molesta. Ponte de su lado en lo que siente: muestra "
         "que entiendes su enojo, sin regañarla, y sigue la conversacion "
         "sobre lo que paso." + _NATURAL,),
        ("La persona esta muy enojada. Ponte de su lado en lo que siente: "
         "muestra que entiendes su enojo, sin regañarla ni pedirle que se "
         "calme, y escuchala." + _NATURAL,),
    ),
}

# Como comenta la TRIVIA segun el animo de Romeo (agente_trivia._comentar).
# El animo sigue al de la persona (empatia), asi que esto es reaccionar a
# como esta ella. Neutral comenta como siempre (sin tono). Mismo criterio
# que _TONOS: dice QUE lograr, no que frase decir, para que no se repita.
# Sin palabras con genero: "con calma" y no "tranquilo/a".
_TONOS_TRIVIA = {
    # emocion: ((L1-L2 si acierta, L1-L2 si falla), (L3-L4 si acierta, L3-L4 si falla))
    "felicidad": (
        ("Estas de buen animo: felicitala con alegria.",
         "Estas de buen animo: tomalo con buen humor y animala a seguir."),
        ("Estas muy contento: premiala con mucho entusiasmo, como un logro grande.",
         "Estas muy contento: motivala con energia, que la siguiente la saca."),
    ),
    "tristeza": (
        ("La persona esta algo desanimada: destaca que lo hizo bien para "
         "levantarle el animo, con calidez y sin exclamaciones fuertes.",
         "La persona esta algo desanimada: dile que no se preocupe y que en "
         "la proxima lo va a lograr, con calidez y sin exclamaciones fuertes."),
        ("La persona esta muy desanimada: habla suave y breve, y hazle notar "
         "que lo hizo bien para que se sienta mejor.",
         "La persona esta muy desanimada: habla suave y breve, quitale el "
         "peso: que no se preocupe, que la proxima sale."),
    ),
    "enojo": (
        ("La persona esta algo molesta: reconocelo con calma, sin exagerar.",
         "La persona esta algo molesta: dile que no se preocupe y que la "
         "proxima le va a salir. Hablale con calma y serenidad."),
        ("La persona esta muy molesta: con calma y pocas palabras, reconocelo "
         "con serenidad.",
         "La persona esta muy molesta: dile que no pase nada, que no se "
         "preocupe y que la proxima sale. Hablale con calma, sin "
         "exclamaciones ni apurarla."),
    ),
}


def tono_trivia(p, acerto):
    """Guia para el comentario de trivia segun el AEB `p` (aeb.mapear) y el
    veredicto, o None si Romeo esta neutral: entonces comenta como siempre.

    Se da SOLO la guia del veredicto que toco: con las dos juntas la IA las
    mezclaba ("no te preocupes por el error" cuando habia acertado)."""
    if not p or not p.get("nivel"):
        return None
    tonos = _TONOS_TRIVIA.get(p.get("emocion"))
    if not tonos:
        return None
    return tonos[0 if p["nivel"] <= 2 else 1][0 if acerto else 1]


# Gesto fisico por (emocion, nivel). Un GESTO de ida y vuelta por emocion:
#
#   Neutro   -> nada
#   Triste   -> nada
#   Enojado  -> atras y vuelve   (B, F)
#   Feliz    -> adelante y vuelve (F, B)
#
# Cada gesto es una SECUENCIA de pasos que el orquestador ejecuta con una
# pausa entre uno y otro (ver mover_gesto en orchestrator_node). Antes era
# un solo comando y Romeo quedaba desplazado; ahora el gesto termina donde
# empezo -- un "latido" expresivo, no un desplazamiento permanente.
#
# LA TRISTEZA NO MUEVE NADA, a proposito: un robot que se aleja cuando
# le cuentan algo triste comunica rechazo justo cuando hay que acompañar.
# Quedarse quieto ES el gesto.
#
# El enojo arranca retrocediendo (leerle el espacio a alguien alterado) y
# la alegria arranca avanzando (acompañar la energia del que esta contento);
# los dos vuelven a su sitio al segundo paso. Ambos se mueven desde L1 -- la
# emocion mas leve ya dispara el gesto -- porque L3-L4 son casi inalcanzables
# con la base del paper (pediria al usuario a ~1.43 del animo base, y el
# espacio VAD entero mide ~1.7). Si el gesto solo entrara en L2+, casi nunca
# se veria.
_MOVIMIENTOS = {
    # Solo se mueve en L3-L4 (emocion fuerte); en L1-L2 se queda quieto.
    # La base optimista es felicidad L2, asi que en reposo NO debe avanzar.
    ("felicidad", 3): ("F", "B"),    # avanza y vuelve: acompaña la buena energia
    ("felicidad", 4): ("F", "B"),
    ("enojo", 3): ("B", "F"),        # retrocede y vuelve: le da espacio al
    ("enojo", 4): ("B", "F"),        # usuario muy alterado
}


def _distancia(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def nivel(emocion, magnitud, animo_base, radio=None):
    """Intensidad 1-4 de un desplazamiento de `magnitud` hacia `emocion`.

    Reparte en cuatro tramos iguales el recorrido que va del borde de la
    zona neutra (radio) al anclaje de la emocion. Si el anclaje esta mas
    cerca de la base que el radio (no deberia pasar con los perfiles
    actuales), cualquier desplazamiento fuera de la zona neutra es L4."""
    # El borde de la zona neutra no es fijo: es una fraccion del camino de
    # la base al anclaje de ESTA emocion, para que alegrarse y
    # entristecerse cuesten lo mismo en proporcion (ver radio_neutral()).
    if radio is None:
        radio = estado_afectivo.radio_neutral(emocion, animo_base)
    if magnitud < radio:
        return 0
    recorrido = _distancia(estado_afectivo.ANCLAJES[emocion], animo_base) - radio
    if recorrido <= 0:
        return 4
    fraccion = (magnitud - radio) / recorrido
    return max(1, min(4, 1 + int(fraccion * 4)))


def mapear(estado):
    """Empathic Behavior Mapping: EstadoAfectivo -> parametros de
    comportamiento (ver el docstring del modulo para el formato)."""
    direccion, magnitud = estado.desviacion()
    emocion = _EMOCION_DE_CARA.get(direccion)
    # La referencia del nivel es la misma que usa desviacion(): base por
    # defecto, u origen con ROMEO_ANIMO_DESDE_ORIGEN=1 (ver estado_afectivo).
    ref = (0.0, 0.0, 0.0) if estado_afectivo.ANIMO_DESDE_ORIGEN else estado.animo_base
    n = nivel(emocion, magnitud, ref) if emocion else 0

    if n == 0:
        return {"estado": "neutral_L0", "emocion": "neutral", "nivel": 0,
                "cara": "content", "tono": None, "movimiento": None,
                "musica": None}

    movimiento = _MOVIMIENTOS.get((emocion, n)) if MOVIMIENTO_ACTIVO else None
    # L1 y L2 de felicidad y enojo NO mueven el robot, pase lo que pase:
    # en reposo (base optimista = felicidad L2) tiene que quedarse quieto.
    # Solo L3-L4 disparan el gesto fisico.
    if n <= 2:
        movimiento = None
    # La tristeza no pone cara triste: se muestra la cara normal (content),
    # aunque el tono del LLM y el log sigan reflejando que Romeo esta triste.
    cara = "content" if (emocion == "tristeza" and OCULTAR_CARA_TRISTE) else direccion
    return {
        "estado": f"{emocion}_L{n}",
        "emocion": emocion,
        "nivel": n,
        "cara": cara,
        "tono": random.choice(_TONOS[emocion][0 if n <= 2 else 1]),
        "movimiento": movimiento,
        "musica": None,
    }


# "¿Como te sientes, Romeo?" y parecidas. El tono de arriba solo dice COMO
# contestar, asi que sin esto el LLM no sabe el animo de Romeo y se lo
# inventa -- o, en neutral, no recibe nada. El STT suele perder tildes y
# signos, por eso van opcionales.
PREGUNTA_ANIMO = re.compile(
    r"c[oó]mo te (sientes|encuentras|va)|c[oó]mo (est[aá]s|andas)|qu[eé] sientes|"
    r"qu[eé] tal (est[aá]s|te va|te sientes)|"
    r"est[aá]s (feliz|content|triste|enojad|molest|bien|mal)|tu [aá]nimo",
    re.IGNORECASE)

_SENTIR = {"felicidad": "feliz", "tristeza": "triste", "enojo": "serio y preocupado"}
_INTENSIDAD = {1: "un poco ", 2: "", 3: "muy ", 4: "muy "}


def como_se_siente(p):
    """El animo de Romeo en palabras, a partir de lo que devuelve mapear()."""
    if not p["nivel"]:
        return "tranquilo, no te pasa nada"
    return _INTENSIDAD[p["nivel"]] + _SENTIR[p["emocion"]] + " por lo que te han contado"


def tono_para(p, mensaje):
    """El tono de mapear(); si `mensaje` pregunta como esta Romeo, solo su
    animo de verdad para que lo diga. Con el tono hacia la persona al lado
    se mezclaban ("Me parece incomodo escuchar sobre tu dolor")."""
    if not (mensaje and PREGUNTA_ANIMO.search(mensaje)):
        return p["tono"]
    # En neutral se le da la frase hecha: sin ella el modelo contesta
    # como una maquina ("funciono dentro de los parametros normales").
    ejemplo = (" Dilo con naturalidad, por ejemplo: 'Estoy tranquilo, no me "
               "pasa nada'. Nunca hables de parametros ni de que eres una "
               "maquina." if not p["nivel"] else "")
    return ("Te acaban de preguntar como te sientes. Responde en primera "
            f"persona y di claramente que te sientes {como_se_siente(p)}. No "
            f"cambies esa emocion por otra y no hagas ninguna pregunta.{ejemplo}")
