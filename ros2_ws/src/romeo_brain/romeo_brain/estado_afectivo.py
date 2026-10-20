# -*- coding: utf-8 -*-
"""El estado emocional de Romeo, en coordenadas continuas.

Implementa los tres motores de la capa afectiva de Rocha, Muchaluat-Saade y
Favela (UFF/CICESE, 2026) -- el framework del robot FRED:

    PULSE   convierte lo que ven los clasificadores (FER/TER/SER) a
            coordenadas VAD y los fusiona pesando cada uno por su confianza
    ROSE    mantiene el animo PROPIO de Romeo: se contagia del usuario
            segun un factor de empatia, y con el tiempo vuelve a su base

El tercer motor del framework (AEB, que traduce la coordenada a un gesto
concreto por distancia euclidiana) NO esta implementado: Romeo ya decide
que cara poner con agente_trivia.elegir_cara_pregunta() y el resto de la
logica que ya tenia, y meter otra via de decision encima solo serviria para
que las dos se pisaran. El estado de aca queda disponible por si algun dia
se quiere usar.

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
# enojo salian en uno. Con 0.25 las tres aparecen en dos o tres turnos: ni
# tan sensible que la cara cambie por una lectura suelta, ni tan sordo que
# la alegria no llegue nunca.
_RADIO_NEUTRAL = float(os.environ.get("ROMEO_RADIO_NEUTRAL", "0.25"))

# Cuanto pesa el TER (texto) frente al FER (cara). Ver percibir_del_texto:
# el texto es la modalidad menos fiable de las dos -- ironia, errores del
# STT, frases sobre terceros -- asi que mueve el animo mas despacio.
#
# 0.30 esta medido, no elegido a ojo: por encima (0.4-0.5) un solo "estoy
# triste" ya cambiaba la cara, y por debajo (0.2-0.25) el comportamiento es
# el mismo que con 0.30 pero con menos margen. Con 0.30 hace falta un
# segundo mensaje concordante, que es lo que distingue un estado de animo
# de un comentario suelto.
_PESO_TEXTO = float(os.environ.get("ROMEO_PESO_TEXTO", "0.30"))


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
DECAIMIENTO = float(os.environ.get("ROMEO_DECAIMIENTO", "0.5"))

# LATENCIA (l): segundos que sostiene la emocion antes de empezar a decaer.
# Sin esto el estado se evaporaria antes de que llegue a notarse en la voz.
LATENCIA = float(os.environ.get("ROMEO_LATENCIA", "3.0"))




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

    EL TEXTO PESA MENOS QUE LA CARA. La confianza que devuelve el TER ya
    esta topada, pero aun asi un solo "estoy triste" bastaba para cambiar
    la cara de Romeo, y eso es demasiado: la frase puede ser ironica, estar
    mal transcrita por el STT, o referirse a otra persona ("mi hermano esta
    triste"). Un emparejador de palabras no da para reaccionar al primer
    mensaje. Con _PESO_TEXTO el animo se mueve despacio y hacen falta dos o
    tres mensajes concordantes, que es lo que de verdad indica un estado.

    Devuelve (vector_vad, confianza), o (None, 0.0) si no hubo evidencia."""
    if emocion is None or emocion not in ANCLAJES:
        return None, 0.0
    c = max(0.0, min(1.0, float(confianza_ter or 0.0)))
    if c <= 0.0:
        return None, 0.0
    c *= _PESO_TEXTO
    av, aa, ad = ANCLAJES[emocion]
    return (av * c, aa * c, ad * c), c


def percibir_del_servicio(emocion, confianza_srv):
    """PULSE a partir de lo que entrega DetectEmotion.srv, que son solo
    (emocion, confianza) y no la distribucion entera.

    POR QUE NO SE USA percibir() DIRECTAMENTE. La distribucion completa
    existe, pero vive en el proceso de romeo_drivers: el .srv transporta
    unicamente la etiqueta ganadora y su probabilidad. Pasarla entera
    obligaria a cambiar romeo_interfaces y recompilarlo, y el beneficio no
    lo justifica -- la confianza, que es la mitad del trabajo de PULSE, ya
    viene en la respuesta, asi que no hace falta recalcular la entropia.

    Lo que se hace es escalar el anclaje por la confianza: una deteccion al
    40% mueve el estado menos que una al 95%, que es exactamente el efecto
    que la entropia buscaba. El resto de la probabilidad se reparte hacia
    el origen (neutral), igual que haria la suma ponderada de a_vad() si se
    tuviera la distribucion entera.

    Devuelve (vector_vad, confianza) o (None, 0.0) si no hubo deteccion."""
    clave = _DESDE_SERVICIO.get((emocion or "").strip())
    if clave is None:
        return None, 0.0
    c = max(0.0, min(1.0, float(confianza_srv or 0.0)))
    av, aa, ad = ANCLAJES[clave]
    return (av * c, aa * c, ad * c), c


# ══════════════════════════════════════════════════════════════════════
# ROSE -- el animo propio de Romeo
# ══════════════════════════════════════════════════════════════════════

class EstadoAfectivo:
    """El estado interno de Romeo, que persiste entre turnos.

    Se instancia una vez por sesion. El orquestador le pasa lo que percibe
    (empatizar) y le pregunta como esta (vector, emocion_mas_cercana)."""

    def __init__(self, empatia=None, animo_base=None, decaimiento=None,
                 latencia=None):
        self.empatia = EMPATIA if empatia is None else empatia
        self.animo_base = tuple(ANIMO_BASE if animo_base is None else animo_base)
        self.decaimiento = DECAIMIENTO if decaimiento is None else decaimiento
        self.latencia = LATENCIA if latencia is None else latencia
        self.vector = self.animo_base
        self._ultimo_estimulo = None

    def reiniciar(self):
        """Vuelve al animo base. Se llama al empezar una sesion nueva: el
        usuario siguiente no hereda el humor que dejo el anterior."""
        self.vector = self.animo_base
        self._ultimo_estimulo = None

    def empatizar(self, vector_usuario):
        """Mueve el estado de Romeo hacia el del usuario, segun la empatia:

            s_r' = e * s_u + (1 - e) * s_r

        Con e=0.4 toma el 40% de lo que siente el usuario y conserva el
        60% de lo suyo, asi que acompaña sin hundirse."""
        if vector_usuario is None:
            return self.vector
        e = self.empatia
        self.vector = tuple(
            e * vector_usuario[i] + (1.0 - e) * self.vector[i]
            for i in range(3)
        )
        self._ultimo_estimulo = time.monotonic()
        return self.vector

    def decaer(self, ahora=None):
        """Vuelve de a poco al animo base:

            s_r(t+dt) = (1 - d) * s_r(t) + d * b

        No hace nada hasta que pasaron `latencia` segundos del ultimo
        estimulo: la emocion tiene que durar lo suficiente para que se note
        en la voz y la cara antes de empezar a diluirse."""
        if self._ultimo_estimulo is None:
            return self.vector
        ahora = time.monotonic() if ahora is None else ahora
        if ahora - self._ultimo_estimulo < self.latencia:
            return self.vector
        d = self.decaimiento
        self.vector = tuple(
            (1.0 - d) * self.vector[i] + d * self.animo_base[i]
            for i in range(3)
        )
        if self._cerca_del_base():
            self._ultimo_estimulo = None
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
        # Desviacion respecto al animo base.
        dv = tuple(self.vector[i] - self.animo_base[i] for i in range(3))
        magnitud = math.sqrt(sum(x * x for x in dv))
        if magnitud < _RADIO_NEUTRAL:
            return "content"

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
        return mejor

    def __repr__(self):
        v, a, d = self.vector
        return (f"<EstadoAfectivo v={v:+.2f} a={a:+.2f} d={d:+.2f} "
                f"cara={self.cara()}>")
