# -*- coding: utf-8 -*-
"""agente_trivia -- todo lo del juego de preguntas y del juego de caras.

Reune lo que antes estaba repartido en agent_corrector.py (evaluar la
respuesta del usuario) y agent_behavior.py (elegir cara, musica y
desplazamiento segun la pregunta). Son dos caras del mismo trabajo y nunca
se usan por separado: el orquestador llamaba a los dos en el mismo turno.

No hay ningun LLM aca: la correccion es determinista (comparacion
normalizada de texto y numeros) y la expresion sale de los campos de la
propia pregunta en preguntas.jsonl.
"""

# ─── Correccion de respuestas (venia de agent_corrector.py) ─────────
import difflib
import re
import unicodedata

_ARTICULOS = {"el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al"}
_UMBRAL_SOLAPAMIENTO = 0.6
_UMBRAL_FUZZY = 0.8


def _normalizar(s):
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[¿?¡!.,;:]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _palabras_significativas(s):
    return [w for w in _normalizar(s).split() if w not in _ARTICULOS and len(w) > 1]


def _es_numerico(s):
    return bool(re.fullmatch(r"-?\d+([.,]\d+)?", s.strip()))


def _extraer_numeros(s):
    return [float(n.replace(",", ".")) for n in re.findall(r"-?\d+(?:[.,]\d+)?", s)]


def evaluar_respuesta(esperada, respuesta_usuario):
    """Compara la respuesta del usuario con la esperada. Devuelve True/False."""
    esperada = (esperada or "").strip()
    respuesta_usuario = (respuesta_usuario or "").strip()
    if not esperada or not respuesta_usuario:
        return False

    if _es_numerico(esperada):
        numero_esperado = float(esperada.replace(",", "."))
        return numero_esperado in _extraer_numeros(respuesta_usuario)

    esp_norm = _normalizar(esperada)
    resp_norm = _normalizar(respuesta_usuario)

    if esp_norm in resp_norm:
        return True

    palabras_esp = _palabras_significativas(esperada)
    palabras_resp = set(_palabras_significativas(respuesta_usuario))
    if palabras_esp:
        coincididas = sum(
            1 for p in palabras_esp
            if p in palabras_resp or any(
                difflib.SequenceMatcher(None, p, w).ratio() >= 0.84 for w in palabras_resp
            )
        )
        if coincididas / len(palabras_esp) >= _UMBRAL_SOLAPAMIENTO:
            return True

    return difflib.SequenceMatcher(None, esp_norm, resp_norm).ratio() >= _UMBRAL_FUZZY


# ─── Expresion: cara, musica, motores (venia de agent_behavior.py) ──
# El dataset usa nombres en español (columna 'cara'); FaceCommand.face_name
# espera las claves en inglés de faces/*.mp4|*.gif. "Neutral" mapea a
# "content" porque es la misma cara de reposo -- no hay un asset de
# "neutral" aparte, igual que en el original.
_CARA_A_DISPLAY = {
    "Feliz": "happy",
    "Triste": "sad",
    "Enojado": "angry",
    "Neutral": "content",
}

# Traduce el texto en español de la columna 'desplazamiento' del dataset al
# comando que entiende MotionCommand.msg (que a su vez es casi 1:1 con el
# protocolo del firmware -- ver romeo_interfaces/msg/MotionCommand.msg).
# Esta tabla vivía en Clients/Carrito_Client.py en el proyecto original;
# se mueve acá porque es lógica de NEGOCIO (qué significa "Izquierda" en
# el dataset), no de protocolo de hardware -- moving_control_node ya no sabe nada
# de español, solo de F/B/SL/SR/RL/RR/S/ROTATE360.
_DIRECCION_A_COMANDO = {
    "adelante": "F",
    "atrás": "B",
    "atras": "B",
    "izquierda": "SL",
    "derecha": "SR",
}


def elegir_cara_pregunta(pregunta, acerto):
    """Cara para el veredicto de una pregunta CON respuesta_esperada."""
    campo = "cara_respuesta_buena" if acerto else "cara_respuesta_mala"
    valor = (pregunta.get(campo) or "").strip()
    return _CARA_A_DISPLAY.get(valor)


def cara_para_emocion(nombre):
    """Traduce Feliz/Triste/Enojado/Neutral (columna 'cara') a la clave que
    entiende FaceCommand -- usado por el Juego de emociones/imitación para
    mostrar la cara ANTES de pedirle al usuario que la imite."""
    return _CARA_A_DISPLAY.get((nombre or "").strip())


def expresar_musica(pregunta, bridge, esperar=False, logger=None):
    """Si la pregunta trae algo en 'musical', le pide al `bridge` que la
    reproduzca (PlayMusic.srv vía communication_node). Vacío/ausente -> no
    hace nada, devuelve None. `esperar=True` para Reconocimiento Musical
    (la música es el enunciado, hay que escucharla completa antes de
    responder) -- ver la nota en romeo_interfaces/srv/PlayMusic.srv."""
    musica = (pregunta.get("musical") or "").strip()
    if not musica:
        return None
    if logger is not None:
        logger.info(f"[música] {musica}")
    bridge.reproducir_musica(musica, esperar=esperar)
    return musica


def expresar_desplazamiento(pregunta, bridge, logger=None):
    """Si la pregunta trae algo en 'desplazamiento', le pide al `bridge`
    que publique el MotionCommand correspondiente. Vacío/ausente -> no hace
    nada, devuelve None."""
    desplazamiento = (pregunta.get("desplazamiento") or "").strip()
    if not desplazamiento:
        return None
    if logger is not None:
        logger.info(f"[desplazamiento] {desplazamiento}")

    if desplazamiento.lower().startswith("girar"):
        comando = "ROTATE360"
    else:
        comando = _DIRECCION_A_COMANDO.get(desplazamiento.strip().lower())
    if not comando:
        if logger is not None:
            logger.warning(f"[desplazamiento] valor sin comando mapeado: {desplazamiento!r}")
        return None

    bridge.mover(comando)
    return desplazamiento
