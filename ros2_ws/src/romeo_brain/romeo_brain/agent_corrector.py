"""Agente corrector: decide si la respuesta del usuario a una pregunta de
Trivia es correcta o no -- SIN LLM. Portado tal cual de
Agents/Agent_Corrector.py del proyecto original (comparación determinista:
numérica exacta, o normalización + substring + solapamiento de palabras
clave + fuzzy). Sin ningún cambio de lógica -- ver ese archivo para el
benchmark completo (98.4% accuracy, 100% recall en incorrectas).

orchestrator_node llama a evaluar_respuesta() directo, como función Python
normal -- sin hardware, sin red, no necesita ser servicio ROS2."""
import difflib
import re
import unicodedata

from . import perf_monitor

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


@perf_monitor.medir("agent_corrector")
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
