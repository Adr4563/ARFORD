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
  - Nunca mueve el carrito por tristeza o enojo. Un robot que retrocede
    cuando el niño esta triste, o que avanza cuando esta "enojado", comunica
    lo contrario de lo que se quiere. El unico gesto fisico es de alegria,
    y solo con el animo muy alto (L3-L4).

Variables de entorno:
    ROMEO_AEB_MOVIMIENTO   0 desactiva el gesto fisico (default 1)
"""
import math
import os

from . import estado_afectivo

MOVIMIENTO_ACTIVO = os.environ.get("ROMEO_AEB_MOVIMIENTO", "1") not in ("0", "false", "False")

# Cara de reposo -> nombre de la emocion (claves de ANCLAJES).
_EMOCION_DE_CARA = {"happy": "felicidad", "sad": "tristeza", "angry": "enojo"}

# Instruccion de estilo para el LLM por emocion y nivel (1-2 suave, 3-4
# intenso). No le dice al modelo QUE contestar, solo COMO: Romeo habla con
# niños, asi que el "enojo" se expresa como seriedad tranquila, nunca como
# agresividad, y la tristeza como ternura, nunca como desanimo.
_TONOS = {
    "felicidad": (
        "Estas de buen humor: responde con calidez y algo de entusiasmo.",
        "Estas muy contento: responde con alegria y energia, celebrando lo "
        "que te cuentan.",
    ),
    "tristeza": (
        "Lo que te contaron te dejo un poco apagado: responde con suavidad "
        "y calma, mostrando que lo entiendes.",
        "Estas conmovido por lo que te cuentan: habla despacio y con mucha "
        "ternura, reconoce lo que siente la persona y no la apures.",
    ),
    "enojo": (
        "Estas algo serio: responde con calma y claridad, sin regañar.",
        "Estas serio y preocupado por lo que te cuentan: responde con "
        "calma, sin alzar el tono ni culpar a nadie, y ayuda a la persona "
        "a tranquilizarse.",
    ),
}

# Gesto fisico por (emocion, nivel). Solo alegria, ver el docstring.
_MOVIMIENTOS = {
    ("felicidad", 3): "RR",          # un giro corto, como un saltito
    ("felicidad", 4): "ROTATE360",   # una vuelta entera
}


def _distancia(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def nivel(emocion, magnitud, animo_base, radio=None):
    """Intensidad 1-4 de un desplazamiento de `magnitud` hacia `emocion`.

    Reparte en cuatro tramos iguales el recorrido que va del borde de la
    zona neutra (radio) al anclaje de la emocion. Si el anclaje esta mas
    cerca de la base que el radio (no deberia pasar con los perfiles
    actuales), cualquier desplazamiento fuera de la zona neutra es L4."""
    radio = estado_afectivo._RADIO_NEUTRAL if radio is None else radio
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
    n = nivel(emocion, magnitud, estado.animo_base) if emocion else 0

    if n == 0:
        return {"estado": "neutral_L0", "emocion": "neutral", "nivel": 0,
                "cara": "content", "tono": None, "movimiento": None,
                "musica": None}

    movimiento = _MOVIMIENTOS.get((emocion, n)) if MOVIMIENTO_ACTIVO else None
    return {
        "estado": f"{emocion}_L{n}",
        "emocion": emocion,
        "nivel": n,
        "cara": direccion,
        "tono": _TONOS[emocion][0 if n <= 2 else 1],
        "movimiento": movimiento,
        "musica": None,
    }
