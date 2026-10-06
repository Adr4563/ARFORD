#!/usr/bin/env python3
"""API HTTP de reconocimiento de voz en español para Lora.

Expone el motor de `_stt_engine.py` como servicio, para que cualquier cliente
-- la placa ESP32-S3, `communication_node`, o un simple `curl` -- le mande audio
y reciba texto, sin cargar el modelo por su cuenta.

Se usa Flask porque YA es dependencia de este paquete (`_web_bridge.py` levanta
la página web con Flask + SSE en el 8081). Este servicio escucha en el 8082 para
no chocar con aquél.

Dos decisiones importantes:

1. EL MODELO SE CARGA AL ARRANCAR, no en la primera petición. Cargar `tiny`
   cuesta ~1.4s en la Pi; pagarlo en la primera transcripción se notaría como un
   tirón raro justo al empezar a hablar.
2. LAS TRANSCRIPCIONES SE SERIALIZAN con un lock. La Pi tiene 4 núcleos que ya
   comparte con Ollama, y está medido que meter más hilos EMPEORA la latencia
   (ver la tabla en `_stt_engine.py`). Atender dos peticiones a la vez haría más
   lentas a las dos, así que se atienden en fila.

Arrancar en la Pi::

    ~/.lora/venv/bin/python stt_server.py

Probar desde cualquier máquina de la red::

    curl -s http://<ip-de-la-pi>:8082/salud

    curl -s -X POST --data-binary @prueba_es.wav \
         -H "Content-Type: audio/wav" \
         http://<ip-de-la-pi>:8082/transcribir

El audio tiene que ser WAV mono de 16 bits. Si tienes otro formato, la Pi tiene
ffmpeg::

    ffmpeg -i entrada.mp3 -ar 16000 -ac 1 salida.wav

Variables de entorno: las mismas de `_stt_engine.py` (LORA_STT_TAM,
LORA_STT_HILOS, LORA_STT_IDIOMA, LORA_STT_MODELOS_DIR) más:

    LORA_STT_PUERTO   puerto de escucha   (default 8082)
    LORA_STT_HOST     interfaz            (default 0.0.0.0)
"""

import io
import os
import sys
import threading
import time
import wave

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402
from flask import Flask, jsonify, request  # noqa: E402

from lora_drivers._stt_engine import HILOS, IDIOMA, TAM, ReconocedorVoz  # noqa: E402

PUERTO = int(os.environ.get("LORA_STT_PUERTO", "8082"))
HOST = os.environ.get("LORA_STT_HOST", "0.0.0.0")

app = Flask(__name__)
reconocedor = ReconocedorVoz()

# Ver el punto 2 del docstring: transcribir de a una.
_lock = threading.Lock()


def _wav_a_samples(crudo):
    """Convierte bytes de un WAV mono 16 bits a (samples float32, sample_rate).

    Devuelve (None, motivo) si el audio no sirve, para poder responder un 400
    con una explicación útil en vez de un 500 sin contexto.
    """
    try:
        with wave.open(io.BytesIO(crudo), "rb") as w:
            canales = w.getnchannels()
            ancho = w.getsampwidth()
            sr = w.getframerate()
            if canales != 1:
                return None, f"el audio tiene {canales} canales; se espera mono"
            if ancho != 2:
                return None, f"el audio es de {ancho * 8} bits; se esperan 16"
            marcos = w.readframes(w.getnframes())
    except wave.Error as e:
        return None, f"no es un WAV válido: {e}"

    samples = np.frombuffer(marcos, dtype=np.int16).astype("float32") / 32768.0
    return (samples, sr), None


@app.get("/salud")
def salud():
    """Estado del servicio. Útil para comprobar que el modelo cargó de verdad."""
    return jsonify(
        {
            "listo": reconocedor.listo,
            "modelo": f"whisper-{reconocedor.tam}",
            "hilos": reconocedor.hilos,
            "idioma": reconocedor.idioma,
            "segundos_carga": round(reconocedor.segundos_carga, 2),
        }
    )


@app.post("/transcribir")
def transcribir():
    """Recibe un WAV y devuelve el texto.

    Acepta el audio de dos formas: como cuerpo crudo (`--data-binary`) o como
    campo `audio` de un formulario multipart, para que sirva tanto a un cliente
    embebido simple como a una subida desde el navegador.
    """
    if not reconocedor.listo:
        return jsonify({"error": "el modelo no está cargado"}), 503

    if "audio" in request.files:
        crudo = request.files["audio"].read()
    else:
        crudo = request.get_data()

    if not crudo:
        return jsonify({"error": "no llegó audio"}), 400

    datos, motivo = _wav_a_samples(crudo)
    if datos is None:
        return jsonify({"error": motivo}), 400

    samples, sr = datos
    duracion = len(samples) / float(sr)

    t0 = time.perf_counter()
    with _lock:
        texto = reconocedor.transcribir(samples, sr)
    tardanza = time.perf_counter() - t0

    return jsonify(
        {
            "texto": texto,
            "duracion_audio": round(duracion, 2),
            "segundos_proceso": round(tardanza, 2),
            "rtf": round(tardanza / duracion, 2) if duracion else None,
        }
    )


def main():
    print(f"[STT] cargando whisper-{TAM} ({HILOS} hilos, idioma={IDIOMA})...")
    if not reconocedor.cargar():
        # Falla ruidosa a propósito: un servicio que arranca sin modelo solo
        # sirve para devolver 503 a todo el mundo. Mejor no arrancar.
        print("[STT] no se pudo cargar el modelo; el servicio no arranca")
        return 1

    print(f"[STT] escuchando en http://{HOST}:{PUERTO}")
    print(f"[STT]   GET  /salud")
    print(f"[STT]   POST /transcribir   (cuerpo: WAV mono 16 bits)")
    # threaded=False refuerza la serialización del lock y evita que el servidor
    # de desarrollo reparta peticiones entre hilos que se pisarían la CPU.
    app.run(host=HOST, port=PUERTO, threaded=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
