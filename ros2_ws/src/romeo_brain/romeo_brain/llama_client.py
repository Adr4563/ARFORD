"""Cliente HTTP hacia Ollama. Ollama NO es hardware (es un servicio de red,
aunque corra local) así que se queda como librería Python normal dentro de
romeo_brain, sin exponerse por ROS2 -- ver la nota del plan de migración
sobre "qué es nodo vs qué es librería" y la sección de romeo_interfaces
sobre por qué no hay una action GenerateResponse.

El nombre del modelo (CHAT_MODEL) viene de variables de entorno --
orchestrator_node además las expone como parámetros ROS2 declarados, ver la
nota en ese archivo.

Sin streaming: se pedía `stream: True` y se iba acumulando cada delta para
devolver el texto entero al final, que es justo lo que da una llamada
normal. El callback `on_token` que justificaba el streaming no lo usaba
nadie -- Speak.srv es request/response completo, así que el turno no puede
ir diciendo la frase mientras llega. Parsear SSE a mano solo añadía código
y un camino más donde fallar."""

import os

import requests

CHAT_SERVER_HOST = os.environ.get("CHAT_SERVER_HOST", "http://localhost:11434")

CHAT_MODEL = os.environ.get("CHAT_MODEL", "arford-empatico-full")


def generar_respuesta(mensajes, temperature=0.3, max_tokens=50, modelo=None):
    """Pide una respuesta al modelo y la devuelve completa."""
    resp = requests.post(
        f"{CHAT_SERVER_HOST}/v1/chat/completions",
        json={
            "model": modelo or CHAT_MODEL,
            "messages": mensajes,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    resp.encoding = "utf-8"
    return resp.json()["choices"][0]["message"]["content"]
