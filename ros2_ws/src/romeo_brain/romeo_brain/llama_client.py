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
import re
import time

import requests

CHAT_SERVER_HOST = os.environ.get("CHAT_SERVER_HOST", "http://localhost:11434")

CHAT_MODEL = os.environ.get("CHAT_MODEL", "llama3.2:1b")

# Tope de tokens por respuesta. Con 50 las frases se cortaban a la mitad
# ("...y es lo mas importante que estoy con"), porque 50 tokens son unas 35
# palabras y el modelo no sabe que tiene que terminar antes.
#
# 120 deja decir dos o tres frases completas sin que la respuesta se vuelva
# un discurso: la voz tarda ~1 s cada 4 palabras, asi que una respuesta larga
# se hace eterna de escuchar.
MAX_TOKENS = int(os.environ.get("ROMEO_MAX_TOKENS", "120"))


# Por debajo de esto la primera frase es solo una exclamacion ("¡Hola!",
# "¡Claro!") y se le añade la siguiente para que diga algo.
_FRASE_MINIMA = 25


def primera_frase(texto):
    """Lo que Romeo dice en voz alta: UNA frase.

    El modelo pequeño se alarga: cierra con un parrafo de relleno ("Si deseas
    hablar más... no dudes en pedirlo") o, en la trivia, copia el formato del
    prompt ("Pregunta: ... La persona dijo: ..."). Se corta en el primer
    salto de linea y despues en el primer final de frase."""
    texto = (texto or "").strip().split("\n")[0].strip()
    partes = re.split(r"(?<=[.!?])\s+", texto)
    frase = partes[0]
    if len(frase) < _FRASE_MINIMA and len(partes) > 1:
        frase = f"{frase} {partes[1]}"
    return frase.strip()


def generar_respuesta(mensajes, temperature=0.3, max_tokens=MAX_TOKENS, modelo=None):
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


# Cuanto se espera a que un modelo termine de cargar. Es generoso porque es
# precisamente el caso lento: cargar de disco a RAM.
TIMEOUT_PRECARGA = float(os.environ.get("ROMEO_TIMEOUT_PRECARGA", "180"))

# Cuantos tokens se le piden al precargar. Ver la nota en precargar(): con
# 1 no basta, hay que hacerle generar de verdad.
TOKENS_PRECARGA = int(os.environ.get("ROMEO_TOKENS_PRECARGA", "30"))


def precargar(modelos, logger=None):
    """Hace que Ollama cargue estos modelos en RAM antes del primer turno.

    POR QUE. Medido: la PRIMERA llamada a un modelo tarda ~5 s porque
    Ollama lo lee de disco; las siguientes, 55-400 ms. Sin esto el usuario
    paga esa carga en mitad de la conversacion -- y como el corrector de
    trivia es otro modelo distinto del chat, se paga DOS veces, cada una
    en el peor momento: cuando acaba de hablarle al robot.

    Alternar entre modelos ya cargados NO cuesta (medido: 38-71 ms), asi
    que basta con tocarlos una vez cada uno al arrancar.

    Pide TOKENS_PRECARGA tokens y no uno solo. Medido: con max_tokens=1
    la precarga tarda 64 ms pero el primer uso real todavia cuesta 517 ms,
    porque cargar los pesos y arrancar la generacion son cosas distintas.
    Generando ~30 tokens la precarga sube a 2 s -- que se pagan al
    arrancar, sin nadie esperando -- y el primer turno baja a 51 ms.

    Los fallos no interrumpen el arranque: si el servidor no esta, el robot
    ya sabe degradar en cada turno.
    """
    for modelo in dict.fromkeys(m for m in modelos if m):
        try:
            inicio = time.monotonic()
            requests.post(
                f"{CHAT_SERVER_HOST}/v1/chat/completions",
                json={
                    "model": modelo,
                    "messages": [{"role": "user", "content": "hola"}],
                    "max_tokens": TOKENS_PRECARGA,
                    "stream": False,
                },
                timeout=TIMEOUT_PRECARGA,
            ).raise_for_status()
            msg = f"[llm] {modelo} precargado en {time.monotonic() - inicio:.1f} s"
        except Exception as e:  # noqa: BLE001 - arrancar igual, ya se degrada
            msg = f"[llm] no se pudo precargar {modelo}: {str(e)[:70]}"
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)
