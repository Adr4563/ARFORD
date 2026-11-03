# -*- coding: utf-8 -*-
"""Ordenes directas: "ve hacia adelante", "gira 360", "pon cara feliz".

Lo que el usuario pide que Romeo HAGA, no de lo que quiere hablar. Se
resuelve antes de llegar al chat, por tres razones:

  - el LLM no mueve el carrito ni cambia la cara, solo escribe texto;
  - el modelo esta afinado en EmpatheticDialogues y a "gira" contesta
    algo empatico, no gira;
  - una orden mal entendida mueve un robot de verdad, asi que conviene
    que la decida una tabla que se puede leer entera, y no un modelo.

POR QUE PALABRAS CLAVE Y NO EL LLM. Cuesta cero, funciona sin red, y
aqui ademas es mas seguro -- "no te muevas"
no debe acabar en un desplazamiento porque el modelo vio "muevas".

QUE NO HACE. No hay diagonales (FL/FR/BL/BR): el firmware las soporta y
la pagina de control las ofrece, pero por voz no se piden. Tampoco hay
giros de 90 ni 180: el firmware solo sabe girar continuo (RL/RR) o la
vuelta entera (ROTATE360).
"""
import re
import unicodedata

# Comandos de MotionCommand.msg. Los de giro continuo (RL/RR) se paran
# solos con el watchdog del carrito a los 3 s.
ADELANTE = "F"
ATRAS = "B"
GIRO_IZQUIERDA = "RL"
GIRO_DERECHA = "RR"
VUELTA_ENTERA = "ROTATE360"
PARAR = "S"

# Las caras de data/faces/. 'speaking' y 'countdown' no estan: las usa el
# robot para decir que esta hablando o esperando, y dejarlas pedir a mano
# solo confunde lo que la cara significa.
# Las claves son RAICES, no palabras enteras: se buscan con \b al
# principio pero no al final, asi "enoja" cubre "enojado" y "enojada" sin
# repetir cada genero. Por eso 'neutral' va antes que 'normal' y ninguna
# raiz es prefijo de otra con distinta cara.
_CARAS = {
    "feliz": "happy", "alegre": "happy", "content": "happy",
    "happy": "happy",
    "triste": "sad", "sad": "sad",
    "enoja": "angry", "enfada": "angry", "molest": "angry",
    "bravo": "angry", "angry": "angry",
    "neutral": "content", "normal": "content", "seri": "content",
}

# Verbos que piden una cara, CON o SIN la palabra "cara" en medio: "pon
# cara de feliz", "ponte triste", "muestra tu cara enojada". La emocion
# se busca despues del verbo, asi que "hoy me siento feliz" no entra --
# no hay verbo de orden -- pero "ponte feliz" si.
_PIDE_CARA = re.compile(
    r"\b(pon|ponte|pongas|muestra|muestrame|haz|hazme|cambia)\b", re.I)

# Giro completo. "360" solo ya es inequivoco; "da una vuelta" tambien.
_VUELTA = re.compile(
    r"\b(360|trescientos sesenta|vuelta completa|vuelta entera|"
    r"da una vuelta|date una vuelta|gira completo|giro completo)\b", re.I)

# Desplazamientos. El orden importa: "hacia atras" tiene que ganar a
# "hacia adelante" cuando el STT mete las dos (no pasa, pero el coste de
# ordenarlo bien es cero).
_MOVIMIENTOS = [
    # Los verbos llevan las terminaciones que importan en vez de \b, para
    # cubrir imperativo e infinitivo de una vez ("¿podrias avanzar?" es
    # tan orden como "avanza") sin tragarse la conversacion: "caminar" a
    # secas NO es una orden ("me gusta caminar por el parque"), asi que
    # solo cuenta con un "hacia adelante" al lado, que ya cae en la
    # primera alternativa.
    # Las terminaciones van de la mas larga a la mas corta: la alternancia
    # de re devuelve la PRIMERA que casa, asi que con (a|as) "avanzas"
    # casaria por "avanza" y el \b final fallaria contra la "s" suelta.
    # "atras" solo ya basta: las variantes ("para atras", "hacia atras",
    # "marcha atras") lo contienen.
    (re.compile(r"\b(atras|retroced(es|as|er|e)|reversa|recula)\b",
                re.I), ATRAS),
    # "adelant(e|a)" porque el STT se come la vocal final: "ve adelante"
    # llega como "hace adelanta" mas veces de las que parece (visto en el
    # robot). La raiz sola basta -- "adelante" y "adelanta" solo salen en
    # una orden; nadie dice "adelanta" en una conversacion con Romeo.
    #
    # avanZar pero avanCes: el subjuntivo cambia la z por c. "avance" a
    # secas NO entra: _limpio() quita las tildes, asi que seria lo mismo
    # que "avance" de "avance mucho en la tarea", que no es una orden.
    (re.compile(r"\b(adelant(e|a)|avanz(as|ar|a)|avances)\b", re.I),
     ADELANTE),
    (re.compile(r"\b(gira a la izquierda|giro a la izquierda|"
                r"gira izquierda|vuelta a la izquierda|"
                r"voltea a la izquierda)\b", re.I), GIRO_IZQUIERDA),
    (re.compile(r"\b(gira a la derecha|giro a la derecha|gira derecha|"
                r"vuelta a la derecha|voltea a la derecha)\b", re.I),
     GIRO_DERECHA),
    (re.compile(r"\b(para|parate|detente|quieto|alto|frena|no te muevas)\b",
                re.I), PARAR),
]

# Frases que NIEGAN la orden. "no te muevas" no es un desplazamiento, y
# "no gires" no es un giro. Se mira antes que nada salvo el PARAR, que
# justamente es lo que "no te muevas" significa.
_NEGADO = re.compile(
    r"\bno\b.{0,12}\b(gires|te muevas|avances|retrocedas|vayas|"
    r"camines|te des)\b", re.I)


def _limpio(texto):
    """Minusculas y sin tildes: el STT transcribe 'atrás' acentuado y las
    tablas de aqui van sin tilde."""
    sin = unicodedata.normalize("NFD", (texto or "").lower())
    return "".join(c for c in sin if unicodedata.category(c) != "Mn")


def interpretar(texto):
    """Que orden directa pide este mensaje, o None si no pide ninguna.

    Devuelve un dict con 'tipo' ("movimiento" o "cara") y 'valor' (el
    comando de MotionCommand o el nombre de la cara). None significa que
    el mensaje es conversacion y le toca al chat."""
    t = _limpio(texto)
    if not t:
        return None

    # "no te muevas" es un PARAR, no un movimiento negado; cualquier otra
    # negacion anula la orden.
    if _NEGADO.search(t) and "no te muevas" not in t:
        return None

    m = _PIDE_CARA.search(t)
    if m:
        # La cara va despues del verbo: "pon cara de feliz".
        resto = t[m.end():]
        for palabra, cara in _CARAS.items():
            # Sin \b al final: asi "enojada" casa con "enojado" y
            # "contenta" con "contento", sin repetir cada genero.
            if re.search(rf"\b{palabra}", resto):
                return {"tipo": "cara", "valor": cara}

    if _VUELTA.search(t):
        return {"tipo": "movimiento", "valor": VUELTA_ENTERA}

    for patron, comando in _MOVIMIENTOS:
        if patron.search(t):
            return {"tipo": "movimiento", "valor": comando}

    return None
