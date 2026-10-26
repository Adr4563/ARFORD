# -*- coding: utf-8 -*-
"""El estado emocional de Romeo, en coordenadas continuas.

Implementa los tres motores de la capa afectiva de Rocha, Muchaluat-Saade y
Favela (UFF/CICESE, 2026) -- el framework del robot FRED:

    PULSE   convierte lo que ven los clasificadores (FER/TER/SER) a
            coordenadas VAD y los fusiona pesando cada uno por su confianza
    ROSE    mantiene el animo PROPIO de Romeo: se contagia del usuario
            segun un factor de empatia, y con el tiempo vuelve a su base

El tercer motor del framework, AEB (que traduce el animo de Romeo a
comportamiento: cara, tono de voz, movimiento), vive en aeb.py y lee el
estado de aca. No pisa las caras de trivia ni del juego de caras: esas
siguen saliendo del dataset y de la logica del orquestador.

Aca NO hay IA: son formulas de aritmetica (promedios ponderados, un
logaritmo e interpolacion lineal). La IA esta afuera,
en los clasificadores que dan la entrada -- en Romeo, el FER de
_emotion_detector.py. Por eso esto corre en microsegundos y no suma ni un
MB a la Pi: el robot FRED del paper hace lo mismo en un Arduino UNO.

POR QUE HACE FALTA. Hoy cada turno es independiente: se detecta "triste",
se dice una frase fija, y se olvida. Con esto Romeo tiene un estado que
PERSISTE entre turnos y decae solo, asi que puede sonar distinto segun como
venga yendo la conversacion, no solo segun el ultimo frame.

El espacio es VAD (Russell y Mehrabian), tres ejes en [-1, 1]:

    valencia    -1 malestar/tristeza      +1 placer/alegria
    activacion  -1 calma/sopor            +1 euforia/rabia
    dominancia  -1 sumision/miedo         +1 control/liderazgo
"""
import math
import os
import time

# ── Puntos de anclaje: categoria discreta -> coordenada VAD ───────────
#
# Valores empiricos de De Bruyne et al. (2020), los mismos que usa el
# framework. Las claves son las ETIQUETAS que ya devuelve el FER de
# _emotion_detector.py, para poder mapear sin traducir nada.
#
# "desprecio" NO esta en De Bruyne. Se le pone una coordenada propia
# (negativo, poco activado, dominante) por analogia con el asco pero con
# mas dominancia, que es como lo describe la literatura. Queda marcado
# porque es el unico valor de esta tabla que no viene del paper.
ANCLAJES = {
    "neutral":    (0.00, 0.00, 0.00),
    "felicidad":  (0.76, 0.48, 0.35),
    "sorpresa":   (0.40, 0.67, -0.13),
    "tristeza":   (-0.63, -0.27, -0.33),
    "enojo":      (-0.51, 0.59, 0.25),
    "asco":       (-0.60, 0.35, 0.11),
    "miedo":      (-0.64, 0.60, -0.43),
    "desprecio":  (-0.55, 0.20, 0.30),   # no es de De Bruyne, ver arriba
}

# Las CUATRO caras que Romeo sabe poner (las de _CARA_A_DISPLAY en
# agente_trivia), con la coordenada VAD de cada una. 'content' hace de
# neutral. No hay cara para miedo, asco ni sorpresa, asi que esas
# emociones se leen del usuario pero el robot no las expresa.
_CARAS = {
    "happy": ANCLAJES["felicidad"],
    "sad": ANCLAJES["tristeza"],
    "angry": ANCLAJES["enojo"],
    "content": ANCLAJES["neutral"],
}

# Radio dentro del cual el animo se considera neutro. Ver cara().
# Medido con detecciones al 90%: con 0.35 la felicidad tardaba CINCO
# turnos en salir -- su desviacion maxima posible es 0.43, porque la base
# ya es positiva y queda poco margen hacia arriba -- mientras tristeza y
# enojo salian en uno.
#
# 0.15 desde que la empatia se aplica entera (ver empatizar): con el
# perfil OPTIMISTIC un mensaje alegre mueve el animo 0.17 (0.4 x 0.43) y
# uno triste o enojado 0.38-0.52, asi que con 0.15 las tres emociones se
# notan desde el primer mensaje que las expresa. Con 0.25 la alegria no
# salia nunca: entre turno y turno el animo decae y no llegaba a acumularse.
# Los mensajes sin carga emocional no mueven nada (el TER devuelve "sin
# evidencia"), asi que la cara no parpadea por cualquier frase.
_RADIO_NEUTRAL = float(os.environ.get("ROMEO_RADIO_NEUTRAL", "0.15"))



# ── Perfil afectivo de Romeo (los parametros de ROSE) ─────────────────
#
# EMPATIA (e): cuanto se deja influir por el usuario.
#   1.0 = absorbe su estado tal cual; 0.0 = inmune.
#   El paper usa 0.4 para el "perfil optimista/regulador" de FRED: ante un
#   usuario triste el robot se entristece un poco pero se queda en la zona
#   positiva y puede animar. Esa es la idea que se quiere -- un robot que
#   se hunda con la persona no la ayuda.
#
EMPATIA = float(os.environ.get("ROMEO_EMPATIA", "0.4"))

# ANIMO_BASE (b): la personalidad por defecto, a la que siempre vuelve.
# Positivo y despierto, sin llegar a eufórico.
#
# Son los valores del perfil optimista de FRED en el paper.
ANIMO_BASE = (
    float(os.environ.get("ROMEO_ANIMO_V", "0.40")),
    float(os.environ.get("ROMEO_ANIMO_A", "0.30")),
    float(os.environ.get("ROMEO_ANIMO_D", "0.20")),
)

# DECAIMIENTO (d): que tan rapido vuelve al animo base, por paso.
#
# 0.3 para que UN MINUTO de silencio lo devuelva a su base. Con el paso
# de 20 s de abajo, desde una desviacion tipica de 0.40:
#
#     0.5 (antes) -> zona neutra a los 40 s, al minuto queda el 12%
#     0.3 (ahora) -> zona neutra a los 20 s, al minuto queda el  3%
#
# El decaimiento es exponencial, asi que nunca llega a cero exacto: un 3%
# de 0.40 son 0.012, muy por dentro del radio neutro (0.15). Para efectos
# practicos, al minuto Romeo esta en su base.
DECAIMIENTO = float(os.environ.get("ROMEO_DECAIMIENTO", "0.3"))

# LATENCIA (l): segundos que sostiene la emocion antes de empezar a decaer.
# Sin esto el estado se evaporaria antes de que llegue a notarse en la voz.
LATENCIA = float(os.environ.get("ROMEO_LATENCIA", "3.0"))

# PASO (dt): cada cuantos segundos se aplica un paso de decaimiento.
#
# El decaimiento depende del TIEMPO, no de cuantos turnos hubo: antes se
# daba un paso por turno, asi que tras diez minutos de silencio Romeo
# seguia casi igual de triste que en el ultimo mensaje. Ahora decaer()
# aplica tantos pasos como intervalos de PASO hayan pasado desde que
# termino la latencia.
#
# 20 s porque es lo que dura un turno real (medido en el robot: 12-25 s
# entre mensajes, contando lo que habla el usuario). Con 5 s el animo
# volvia entero a la base antes del turno siguiente y nunca se acumulaba.
# Asi, en una conversacion activa decae ~un paso por turno, y una pausa
# larga si lo devuelve a la base.
PASO_DECAIMIENTO = float(os.environ.get("ROMEO_PASO_DECAIMIENTO", "20.0"))


# ── Perfiles predefinidos (Robot Affective Profile de la tesis) ───────
#
# Cada perfil fija los cuatro parametros de ROSE. Se elige con el
# parametro ROS `perfil_afectivo` de orchestrator_node (ver
# romeo_params.yaml). Las variables ROMEO_EMPATIA, ROMEO_ANIMO_*,
# ROMEO_DECAIMIENTO y ROMEO_LATENCIA, si estan definidas, pisan el valor
# del perfil -- sirven para afinar uno sin crear otro.
#
# Son los dos perfiles de los escenarios de la tesis (sec. 4.1.4), con su
# empatia y animo base. Decaimiento y latencia no los publica: se dejan los
# de Romeo. Para añadir otro perfil basta una entrada mas aca.
PERFILES = {
    # El de siempre de Romeo: se entristece un poco con el usuario pero
    # se queda en la zona positiva y puede animarlo. (Tesis, escenario 2.)
    "OPTIMISTIC":  {"empatia": 0.4, "animo_base": (0.40, 0.30, 0.20),
                    "decaimiento": 0.3, "latencia": 3.0},
    # Espejo del usuario: absorbe su estado entero. (Tesis, escenario 1.)
    # Util para comparar en experimentos; poco recomendable fuera de experimentos.
    "EMPATHETIC":  {"empatia": 1.0, "animo_base": (0.20, 0.10, 0.00),
                    "decaimiento": 0.3, "latencia": 3.0},
}
PERFIL_POR_DEFECTO = "OPTIMISTIC"


def parametros_de_perfil(nombre=None):
    """Los parametros de ROSE para `nombre`, con las variables de entorno
    aplicadas encima. Un nombre desconocido cae en PERFIL_POR_DEFECTO.

    Devuelve (parametros, nombre_efectivo)."""
    clave = (nombre or PERFIL_POR_DEFECTO).strip().upper()
    if clave not in PERFILES:
        clave = PERFIL_POR_DEFECTO
    p = dict(PERFILES[clave])

    def _env(var):
        v = os.environ.get(var)
        return float(v) if v not in (None, "") else None

    if _env("ROMEO_EMPATIA") is not None:
        p["empatia"] = _env("ROMEO_EMPATIA")
    if _env("ROMEO_DECAIMIENTO") is not None:
        p["decaimiento"] = _env("ROMEO_DECAIMIENTO")
    if _env("ROMEO_LATENCIA") is not None:
        p["latencia"] = _env("ROMEO_LATENCIA")
    base = list(p["animo_base"])
    for i, var in enumerate(("ROMEO_ANIMO_V", "ROMEO_ANIMO_A", "ROMEO_ANIMO_D")):
        if _env(var) is not None:
            base[i] = _env(var)
    p["animo_base"] = tuple(base)
    return p, clave



# ══════════════════════════════════════════════════════════════════════
# PULSE -- de probabilidades a una coordenada VAD
# ══════════════════════════════════════════════════════════════════════

def confianza(probabilidades):
    """Que tan seguro esta un clasificador, por la entropia de Shannon
    normalizada: c = 1 - H(p).

    Si el FER reparte la probabilidad entre varias emociones (duda), la
    entropia sube y la confianza baja. Si la concentra en una, al reves.
    Asi una lectura dudosa pesa menos en la fusion, en vez de valer igual
    que una clara."""
    ps = [p for p in probabilidades if p > 0.0]
    if len(ps) < 2:
        return 1.0
    h = -sum(p * math.log(p) for p in ps) / math.log(len(probabilidades))
    return max(0.0, min(1.0, 1.0 - h))


def a_vad(probabilidades, etiquetas):
    """Convierte la distribucion del clasificador a una coordenada VAD:

        S_vad = sum(p_i * VAD_i)

    Es la media de los anclajes pesada por la probabilidad de cada
    categoria -- por eso hace falta la distribucion ENTERA y no solo la
    etiqueta ganadora: "60% triste, 40% neutral" cae a medio camino entre
    ambas, que es justo lo que se quiere representar."""
    v = a = d = 0.0
    for p, etiqueta in zip(probabilidades, etiquetas):
        av, aa, ad = ANCLAJES.get(etiqueta, (0.0, 0.0, 0.0))
        v += p * av
        a += p * aa
        d += p * ad
    return (v, a, d)


def fusionar(lecturas):
    """Combina varias modalidades (FER, TER, SER...) pesando cada una por
    su confianza:

        S_u = sum(c_m * S_m) / sum(c_m)

    `lecturas` es una lista de (vector_vad, confianza). Con una sola
    modalidad devuelve esa misma -- que es el caso de Romeo hoy, que solo
    tiene FER. El dia que haya TER esto ya funciona sin tocarlo."""
    lecturas = [(s, c) for s, c in lecturas if s is not None and c > 0.0]
    if not lecturas:
        return None
    total = sum(c for _, c in lecturas)
    if total <= 0.0:
        return None
    return tuple(
        sum(s[i] * c for s, c in lecturas) / total
        for i in range(3)
    )


def percibir(probabilidades, etiquetas):
    """PULSE de punta a punta para UNA modalidad: de la distribucion que
    da el clasificador al vector del usuario, con su confianza.

    Devuelve (vector_vad, confianza)."""
    return a_vad(probabilidades, etiquetas), confianza(probabilidades)


# EL ROBOT TRABAJA CON CUATRO EMOCIONES. Son las unicas que la camara sabe
# distinguir (_emotion_detector.ETIQUETAS_RELEVANTES) y las unicas que
# tienen cara (_CARAS, aqui arriba). Las otras cuatro del FER+ se agrupan
# en la mas cercana en vez de descartarse, para no tirar la señal:
#
#     miedo     -> tristeza   negativa; descartarlo dejaba la activacion
#                             en -0.14 con miedo claro, o sea pasaba de
#                             "alerta" a "apagado"
#     asco      -> enojo      negativa y activada
#     sorpresa  -> felicidad  positiva y activada
#     desprecio -> enojo      negativa, dominante
#
# El SER y el TER ya llegan agrupados (sus propias tablas de traduccion);
# esto cubre al FER, que manda sus ocho clases tal cual, y de paso deja el
# criterio en UN solo sitio para las tres fuentes.
#
# Medido: en las cuatro que se conservan el VAD se mueve 0.04-0.09, por
# debajo del ruido del propio sistema.
_A_CUATRO = {
    "miedo": "tristeza",
    "asco": "enojo",
    "desprecio": "enojo",
    "sorpresa": "felicidad",
}


def agrupar_en_cuatro(probabilidades, etiquetas):
    """(etiquetas, probabilidades) con solo las cuatro del robot.

    Suma las que caen en la misma clave. Deja pasar tal cual lo que ya
    viene agrupado."""
    acumulado = {}
    for etiqueta, p in zip(etiquetas, probabilidades):
        clave = _A_CUATRO.get(etiqueta, etiqueta)
        acumulado[clave] = acumulado.get(clave, 0.0) + float(p)
    claves = sorted(acumulado)
    return claves, [acumulado[k] for k in claves]


def percibir_categorias(probabilidades, etiquetas):
    """PULSE para una fuente CAT (el FER de la ventana de percepcion):
    confianza por entropia (tesis, ec. 4.1-4.2) y conversion CAT -> VAD
    pesada por TODA la distribucion (ec. 4.3).

    Devuelve (vector_vad, confianza, dominante), donde `dominante` es la
    etiqueta mas probable. (None, 0.0, None) si la distribucion viene
    vacia o no coincide con las etiquetas."""
    probs = [float(p) for p in (probabilidades or [])]
    if not probs or len(probs) != len(etiquetas) or sum(probs) <= 0.0:
        return None, 0.0, None
    etiquetas, probs = agrupar_en_cuatro(probs, etiquetas)
    total = sum(probs)
    probs = [p / total for p in probs]
    vector, c = percibir(probs, etiquetas)
    dominante = etiquetas[max(range(len(probs)), key=probs.__getitem__)]
    return vector, c, dominante


# Lo que devuelve DetectEmotion.srv, traducido a las claves de ANCLAJES.
# El detector ya colapsa sus ocho categorias a estas cuatro antes de
# publicarlas (ETIQUETAS_RELEVANTES en _emotion_detector.py).
_DESDE_SERVICIO = {
    "Feliz": "felicidad",
    "Triste": "tristeza",
    "Enojado": "enojo",
    "Neutral": "neutral",
}


def percibir_del_texto(emocion, confianza_ter):
    """PULSE a partir de lo que devuelve ter.analizar().

    Mismo criterio que percibir_del_servicio: se escala el anclaje por la
    confianza, asi que un "estoy fatal" mueve el animo mas que un "bien"
    suelto. La diferencia es que aca las emociones son las OCHO de
    ANCLAJES y no las cuatro del detector facial, porque el texto si
    distingue miedo o sorpresa aunque Romeo no sepa ponerles cara.

    EL TEXTO PESA MENOS QUE LA CARA cuando hay varias fuentes: ter.analizar()
    tope su confianza en CONFIANZA_MAXIMA (0.6) mientras el detector facial
    llega a 0.9+, y fusionar() pondera por esa confianza. Con el texto como
    unica fuente, empatizar() aplica la empatia del perfil entera (tesis,
    ec. 4.5).

    Devuelve (vector_vad, confianza), o (None, 0.0) si no hubo evidencia."""
    if emocion is None or emocion not in ANCLAJES:
        return None, 0.0
    c = max(0.0, min(1.0, float(confianza_ter or 0.0)))
    if c <= 0.0:
        return None, 0.0
    # El vector es el anclaje INTACTO: dice que emocion es. La confianza
    # (atenuada por _PESO_TEXTO, porque el texto es menos fiable que la
    # cara) viaja aparte y la usan fusionar() y empatizar() para decidir
    # cuanto creersela. Ver la nota en empatizar().
    return ANCLAJES[emocion], c


def percibir_del_servicio(emocion, confianza_srv):
    """PULSE a partir de lo que entrega DetectEmotion.srv, que son solo
    (emocion, confianza) y no la distribucion entera.

    POR QUE NO SE USA percibir() DIRECTAMENTE. La distribucion completa
    existe, pero vive en el proceso de romeo_drivers: el .srv transporta
    unicamente la etiqueta ganadora y su probabilidad. Pasarla entera
    obligaria a cambiar romeo_interfaces y recompilarlo, y el beneficio no
    lo justifica -- la confianza, que es la mitad del trabajo de PULSE, ya
    viene en la respuesta, asi que no hace falta recalcular la entropia.

    Se devuelve el anclaje de la emocion detectada junto con la confianza
    que da el servicio, que es la que modula cuanto mueve el animo (ver
    empatizar): una deteccion al 40% lo mueve menos que una al 95%, que es
    el efecto que la entropia buscaba.

    Devuelve (vector_vad, confianza) o (None, 0.0) si no hubo deteccion."""
    clave = _DESDE_SERVICIO.get((emocion or "").strip())
    if clave is None:
        return None, 0.0
    c = max(0.0, min(1.0, float(confianza_srv or 0.0)))
    if c <= 0.0:
        return None, 0.0
    return ANCLAJES[clave], c


# ══════════════════════════════════════════════════════════════════════
# ROSE -- el animo propio de Romeo
# ══════════════════════════════════════════════════════════════════════

class EstadoAfectivo:
    """El estado interno de Romeo, que persiste entre turnos.

    Se instancia una vez por sesion. El orquestador le pasa lo que percibe
    (empatizar) y le pregunta como esta (vector, emocion_mas_cercana)."""

    def __init__(self, empatia=None, animo_base=None, decaimiento=None,
                 latencia=None, perfil=None, paso=None):
        # Sin perfil ni parametros sueltos se comporta como siempre: los
        # valores de modulo (OPTIMISTIC + variables de entorno). Con
        # `perfil`, se parte de ese perfil; los parametros sueltos que se
        # pasen igual lo pisan.
        if perfil is not None:
            p, self.perfil = parametros_de_perfil(perfil)
        else:
            p = {"empatia": EMPATIA, "animo_base": ANIMO_BASE,
                 "decaimiento": DECAIMIENTO, "latencia": LATENCIA}
            self.perfil = "PERSONALIZADO"
        self.empatia = p["empatia"] if empatia is None else empatia
        self.animo_base = tuple(p["animo_base"] if animo_base is None else animo_base)
        self.decaimiento = p["decaimiento"] if decaimiento is None else decaimiento
        self.latencia = p["latencia"] if latencia is None else latencia
        self.paso = PASO_DECAIMIENTO if paso is None else paso
        self.vector = self.animo_base
        self._ultimo_estimulo = None
        # Cuantos pasos de decaimiento ya se aplicaron desde el ultimo
        # estimulo (ver decaer()).
        self._decaido_hasta = None

    def reiniciar(self):
        """Vuelve al animo base. Se llama al empezar una sesion nueva: el
        usuario siguiente no hereda el humor que dejo el anterior."""
        self.vector = self.animo_base
        self._ultimo_estimulo = None
        self._decaido_hasta = None

    def empatizar(self, vector_usuario, confianza=1.0):
        """Mueve el estado de Romeo hacia el del usuario, segun la empatia:

            s_r' = e * s_u + (1 - e) * s_r

        Con e=0.4 toma el 40% de lo que siente el usuario y conserva el
        60% de lo suyo, asi que acompaña sin hundirse.

        LA CONFIANZA ENTRA AQUI, NO EN EL VECTOR. Antes las funciones
        percibir_* escalaban el anclaje por la confianza, y eso estaba mal:
        mezclaba CUANTO creer la lectura con QUE emocion es. Una deteccion
        floja de felicidad -- anclaje (0.76, 0.48, 0.35) por 0.09 --
        quedaba en (0.07, 0.04, 0.03), que es practicamente el origen, o
        sea NEUTRAL. Resultado: decir "estaba feliz" BAJABA el animo de
        Romeo en vez de subirlo, porque interpolaba hacia neutral desde una
        base mas positiva.

        Ahora el vector conserva el anclaje entero (la direccion).

        Y LA EMPATIA SE APLICA ENTERA, como en la tesis (ec. 4.5): la
        confianza ya hizo su trabajo antes, en fusionar(), pesando unas
        fuentes frente a otras. Antes se multiplicaba ademas aca, y con el
        TER (confianza ~0.3) el perfil OPTIMISTIC movia el 12 % en vez del
        40 %: el animo nunca salia de la zona neutra. `confianza` se acepta
        por compatibilidad pero solo descarta lecturas sin confianza (0)."""
        if vector_usuario is None or float(confianza) <= 0.0:
            return self.vector
        e = self.empatia
        if e <= 0.0:
            return self.vector
        self.vector = tuple(
            e * vector_usuario[i] + (1.0 - e) * self.vector[i]
            for i in range(3)
        )
        self._ultimo_estimulo = time.monotonic()
        # Un estimulo nuevo reinicia la cuenta (tesis, fig. 18): vuelve a
        # sostenerse `latencia` segundos antes de decaer.
        self._decaido_hasta = None
        return self.vector

    def decaer(self, ahora=None):
        """Vuelve de a poco al animo base, segun el TIEMPO transcurrido:

            s_r(t+dt) = (1 - d) * s_r(t) + d * b

        aplicado una vez al terminar la latencia y otra por cada `paso`
        segundos mas. Como es una interpolacion lineal repetida, n pasos
        equivalen a:

            s_r = b + (1 - d)^n * (s_r - b)

        No hace nada hasta que pasaron `latencia` segundos del ultimo
        estimulo: la emocion tiene que durar lo suficiente para que se note
        en la voz y la cara antes de empezar a diluirse.

        Se puede llamar todas las veces que se quiera: lleva la cuenta de
        cuantos pasos ya aplico desde el ultimo estimulo
        (`_decaido_hasta`), asi que dos llamadas seguidas no descuentan el
        mismo tramo dos veces. paso <= 0 vuelve al comportamiento viejo: un
        paso por llamada."""
        if self._ultimo_estimulo is None:
            return self.vector
        ahora = time.monotonic() if ahora is None else ahora
        inicio = self._ultimo_estimulo + self.latencia
        if ahora < inicio:
            return self.vector
        if self.paso > 0:
            debidos = int((ahora - inicio) // self.paso) + 1
        else:
            debidos = (self._decaido_hasta or 0) + 1
        pasos = debidos - (self._decaido_hasta or 0)
        if pasos <= 0:
            return self.vector
        self._decaido_hasta = debidos
        factor = (1.0 - self.decaimiento) ** pasos
        self.vector = tuple(
            self.animo_base[i] + factor * (self.vector[i] - self.animo_base[i])
            for i in range(3)
        )
        if self._cerca_del_base():
            self.vector = self.animo_base
            self._ultimo_estimulo = None
            self._decaido_hasta = None
        return self.vector

    def _cerca_del_base(self, tol=0.01):
        return all(abs(self.vector[i] - self.animo_base[i]) < tol
                   for i in range(3))

    def cara(self):
        """La cara de reposo que corresponde al animo actual.

        Devuelve una de las cuatro claves que entiende FaceCommand, por
        distancia euclidiana a los anclajes de esas mismas cuatro.

        OJO CON DONDE SE USA. Las caras de Romeo estan casi todas ocupadas
        por estados de la conversacion -- 'speaking' mientras habla,
        'countdown' mientras espera, 'content' mientras escucha -- y esas
        no se pueden sustituir sin que el robot deje de comunicar lo que
        esta haciendo. Esto es para UN solo sitio: el 'content' de reposo,
        cuando Romeo escucha y la cara no esta diciendo nada mas. Ahi, en
        vez de la cara neutra de siempre, puede mostrar como se siente.

        SE MIDE RESPECTO AL ANIMO BASE, NO AL ORIGEN. El animo base es
        positivo ([0.40, 0.30, 0.20]), asi que en coordenadas absolutas
        esta a 0.54 del origen y su cara mas cercana es 'happy': Romeo se
        quedaba sonriendo en reposo permanente, todo el rato, que es
        precisamente lo que no se quiere. La base es su estado NORMAL, y
        lo normal se ve con la cara neutra; lo que merece una cara es
        APARTARSE de ella. Por eso la distancia se mide desde la base: si
        el animo esta donde suele estar -> 'content', y solo cuando algo
        lo mueve de verdad aparece happy/sad/angry.

        La zona neutra es ancha a proposito: los cuatro anclajes estan
        lejos entre si, asi que con el animo cerca de la base el ganador
        saldria casi por azar y la cara parpadearia entre dos sin motivo."""
        direccion, magnitud = self.desviacion()
        return "content" if magnitud < _RADIO_NEUTRAL else direccion

    def desviacion(self):
        """Hacia donde y cuanto se aparto el animo de su base.

        Devuelve (cara, magnitud): la cara de las tres expresivas (happy,
        sad, angry) cuya direccion se parece mas al desplazamiento, y la
        longitud de ese desplazamiento en el espacio VAD. Con el animo
        exactamente en la base devuelve ("content", 0.0).

        Lo usan cara() y el AEB (aeb.py), que necesita ademas la magnitud
        para decidir la intensidad (L1-L4)."""
        dv = tuple(self.vector[i] - self.animo_base[i] for i in range(3))
        magnitud = math.sqrt(sum(x * x for x in dv))
        if magnitud == 0.0:
            return "content", 0.0

        # La cara se elige por la DIRECCION del movimiento, no por donde
        # acabo el vector. Comparar la desviacion con los anclajes tal cual
        # no funciona: desplazarse [+0.36,+0.18,+0.15] es inequivocamente
        # alegre, pero queda lejos del anclaje de felicidad ([0.76,0.48,
        # 0.35]) y ganaba 'content' igual. Como ademas la base ya es
        # positiva, el margen hacia arriba es mucho menor que hacia abajo
        # (0.43 contra 1.29), asi que con distancias absolutas 'happy' no
        # salia nunca, con ningun umbral.
        #
        # Normalizando ambos vectores se compara solo hacia donde apuntan,
        # y las tres caras quedan igual de alcanzables.
        dv_u = tuple(x / magnitud for x in dv)
        mejor, mejor_cos = "content", -2.0
        for nombre, ancla in _CARAS.items():
            if nombre == "content":
                continue
            na = math.sqrt(sum(x * x for x in ancla))
            cos = sum(dv_u[i] * ancla[i] / na for i in range(3))
            if cos > mejor_cos:
                mejor, mejor_cos = nombre, cos
        return mejor, magnitud

    def __repr__(self):
        v, a, d = self.vector
        return (f"<EstadoAfectivo v={v:+.2f} a={a:+.2f} d={d:+.2f} "
                f"cara={self.cara()}>")
