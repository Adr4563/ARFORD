"""Cliente HTTP hacia Ollama -- portado tal cual de Clients/Llama_Client.py
del proyecto original, sin ningún cambio de lógica ni de protocolo. Ollama
NO es hardware (es un servicio de red, aunque corra local) así que se
queda como librería Python normal dentro de lora_brain, sin exponerse por
ROS2 -- ver la nota del plan de migración sobre "qué es nodo vs qué es
librería" y la sección de lora_interfaces sobre por qué no hay una action
GenerateResponse.

Los nombres de modelo (CHAT_MODEL/TRIVIA_MODEL/SALIDA_TRIVIA_MODEL) siguen
viniendo de variables de entorno, igual que el original -- orchestrator_node
además los expone como parámetros ROS2 declarados, ver la nota en ese
archivo."""

import json
import os

import requests

from . import perf_monitor

CHAT_SERVER_HOST = os.environ.get("CHAT_SERVER_HOST", "http://localhost:11434")

CHAT_MODEL = os.environ.get("CHAT_MODEL", "lora-chat-libre-v4")
TRIVIA_MODEL = os.environ.get("TRIVIA_MODEL", "lora-trivia")
SALIDA_TRIVIA_MODEL = os.environ.get("SALIDA_TRIVIA_MODEL", "lora-salida-trivia-v2")

_SALIDA_TRIVIA_SYSTEM = (
    "Sos un clasificador. Te dan la pregunta de una trivia y el mensaje del "
    "usuario. Respondé EXACTAMENTE una palabra: RESPUESTA si el usuario "
    "intenta responder esa pregunta (aunque esté mal, o sea una opinión "
    "larga si la pregunta la pide), o SALIR si se puso a hablar de otra "
    "cosa (un tema personal, un problema, charla sin relación, o pide "
    "explícitamente parar/cambiar de tema)."
)


@perf_monitor.medir("llama_salida_trivia")
def clasificar_salida_trivia(pregunta, mensaje_usuario):
    """True si el usuario se está yendo de la trivia, False si está
    intentando responder la pregunta, None si no se pudo decidir (Ollama
    caído, modelo sin importar, respuesta inesperada) -- el caller
    (orchestrator_node) cae a las listas de palabras clave cuando pasa
    eso."""
    mensajes = [
        {"role": "system", "content": _SALIDA_TRIVIA_SYSTEM},
        {"role": "user",
         "content": f"Pregunta: {pregunta}\nUsuario: {mensaje_usuario}\n¿RESPUESTA o SALIR?"},
    ]
    try:
        resp = requests.post(
            f"{CHAT_SERVER_HOST}/v1/chat/completions",
            json={
                "model": SALIDA_TRIVIA_MODEL,
                "messages": mensajes,
                "temperature": 0,
                "max_tokens": 5,
                "stream": False,
            },
            timeout=30,
        )
        resp.raise_for_status()
        resp.encoding = "utf-8"
        veredicto = resp.json()["choices"][0]["message"]["content"].strip().upper()
    except (requests.RequestException, ValueError, KeyError, IndexError):
        return None
    if veredicto.startswith("SALIR"):
        return True
    if veredicto.startswith("RESPUESTA"):
        return False
    return None


def generar_respuesta(mensajes, temperature=0.3, max_tokens=50, on_token=None, modelo=None):
    """Llama al modelo con streaming. `on_token` sigue soportado (útil si
    algún caller interno de lora_brain quiere ir logueando tokens), pero
    orchestrator_node no lo expone hacia afuera por ROS2 -- Speak.srv es
    request/response completo, no streaming, ver la nota en
    lora_interfaces/srv/Speak.srv."""
    modelo_usado = modelo or CHAT_MODEL
    with perf_monitor.medir_bloque(f"llama_generar:{modelo_usado}"):
        resp = requests.post(
            f"{CHAT_SERVER_HOST}/v1/chat/completions",
            json={
                "model": modelo_usado,
                "messages": mensajes,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            },
            timeout=120,
            stream=True,
        )
        resp.raise_for_status()
        resp.encoding = "utf-8"
        texto = []
        for linea in resp.iter_lines(decode_unicode=True):
            if not linea or not linea.startswith("data: "):
                continue
            payload = linea[len("data: "):]
            if payload == "[DONE]":
                break
            delta = json.loads(payload)["choices"][0]["delta"].get("content", "")
            if delta:
                texto.append(delta)
                if on_token:
                    on_token(delta)
        return "".join(texto)
