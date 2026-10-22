#!/usr/bin/env python3
"""API HTTP de síntesis de voz (TTS) para Romeo.

Es la pareja de `stt_server.py`: corre en la PC y `communication_node` en la Pi
le pide la voz con el parámetro `tts_servidor` (ver `_voice_backends.py`,
`_hablar_remoto`). Sintetizar en la Pi cuesta ~5 s por frase; en la PC, una
fracción de eso. Lo que viaja por la red es el WAV ya sintetizado.

Usa el MISMO motor que la Pi -- VITS con sherpa-onnx y la voz
`es_MX-claude-high` -- con los mismos ajustes de `romeo_params.yaml`, así que
Romeo suena igual sintetice donde sintetice. El cambio de tono no se hace acá:
lo aplica mpv en la Pi al reproducir (`voz_tono_semitonos`).

Contrato que espera el cliente::

    GET /hablar?texto=...   ->  audio/wav (mono, 16 bits)
    GET /salud              ->  JSON con el estado

Arrancar en la PC::

    python tts_server.py

Probar::

    curl -s http://<ip-de-la-pc>:8083/salud
    curl -s -o prueba.wav "http://<ip-de-la-pc>:8083/hablar?texto=Hola%20soy%20Romeo"

Variables de entorno:

    ROMEO_TTS_VITS_DIR   carpeta de la voz   (default ~/.romeo/modelos/vits-piper-es_MX-claude-high)
    ROMEO_TTS_RUIDO      noise_scale         (default 0.667, nativo de Piper)
    ROMEO_TTS_RUIDO_W    noise_scale_w       (default 0.8, nativo de Piper)
    ROMEO_TTS_DURACION   length_scale        (default 1.0, velocidad natural)
    ROMEO_TTS_HILOS      hilos de onnxruntime (default 4)
    ROMEO_TTS_PUERTO     puerto de escucha   (default 8083)
    ROMEO_TTS_HOST       interfaz            (default 0.0.0.0)
"""

import glob
import io
import os
import threading
import time
import wave

import numpy as np
from flask import Flask, Response, jsonify, request

VITS_DIR = os.path.expanduser(os.environ.get(
    "ROMEO_TTS_VITS_DIR", "~/.romeo/modelos/vits-piper-es_MX-claude-high"))
RUIDO = float(os.environ.get("ROMEO_TTS_RUIDO", "0.667"))
RUIDO_W = float(os.environ.get("ROMEO_TTS_RUIDO_W", "0.8"))
DURACION = float(os.environ.get("ROMEO_TTS_DURACION", "1.0"))
HILOS = int(os.environ.get("ROMEO_TTS_HILOS", "4"))
PUERTO = int(os.environ.get("ROMEO_TTS_PUERTO", "8083"))
HOST = os.environ.get("ROMEO_TTS_HOST", "0.0.0.0")

app = Flask(__name__)
_tts = None
_segundos_carga = 0.0
# Una síntesis a la vez, mismo motivo que el lock de stt_server.py: dos a la
# vez se reparten la CPU y salen más lentas las dos.
_lock = threading.Lock()


def cargar():
    """Carga la voz VITS. Devuelve True si quedó lista."""
    global _tts, _segundos_carga
    import sherpa_onnx

    modelos = sorted(glob.glob(os.path.join(VITS_DIR, "*.onnx")))
    if not modelos:
        print(f"[TTS] no hay ningún .onnx en {VITS_DIR}")
        return False
    t0 = time.perf_counter()
    cfg = sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model=modelos[0], tokens=os.path.join(VITS_DIR, "tokens.txt"),
            data_dir=os.path.join(VITS_DIR, "espeak-ng-data"),
            noise_scale=RUIDO, noise_scale_w=RUIDO_W, length_scale=DURACION),
        num_threads=HILOS))
    _tts = sherpa_onnx.OfflineTts(cfg)
    _segundos_carga = time.perf_counter() - t0
    print(f"[TTS] voz {os.path.basename(modelos[0])} lista en "
          f"{_segundos_carga:.1f}s ({HILOS} hilos)")
    return True


def _a_wav(muestras, sr):
    """float32 -> bytes de un WAV mono de 16 bits.

    Normaliza el pico a 0.85, igual que _voice_backends._hablar_vits(): deja
    margen para que el remuestreo y el cambio de tono de mpv no recorten."""
    muestras = muestras * (0.85 / max(1e-6, float(np.abs(muestras).max())))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((muestras * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


@app.get("/salud")
def salud():
    return jsonify({
        "listo": _tts is not None,
        "voz": os.path.basename(VITS_DIR),
        "hilos": HILOS,
        "segundos_carga": round(_segundos_carga, 2),
    })


@app.get("/hablar")
def hablar():
    texto = (request.args.get("texto") or "").strip()
    if not texto:
        return jsonify({"error": "falta el parámetro texto"}), 400
    if _tts is None:
        return jsonify({"error": "la voz no está cargada"}), 503

    t0 = time.perf_counter()
    with _lock:
        audio = _tts.generate(texto, sid=0)
    muestras = np.asarray(audio.samples, dtype=np.float32)
    if muestras.size == 0:
        # El cliente trata un cuerpo vacío como fallo y sintetiza en local.
        return Response(b"", mimetype="audio/wav")
    datos = _a_wav(muestras, audio.sample_rate)
    print(f"[TTS] {time.perf_counter() - t0:.2f}s  "
          f"{muestras.size / audio.sample_rate:.1f}s de audio  {texto[:60]!r}")
    return Response(datos, mimetype="audio/wav")


def main():
    # Falla ruidosa, igual que stt_server.py: sin voz solo devolvería 503.
    if not cargar():
        print("[TTS] no se pudo cargar la voz; el servicio no arranca")
        return 1
    print(f"[TTS] escuchando en http://{HOST}:{PUERTO}")
    print("[TTS]   GET /salud")
    print("[TTS]   GET /hablar?texto=...")
    app.run(host=HOST, port=PUERTO, threaded=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
