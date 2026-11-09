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

Dos motores, uno a la vez (ROMEO_TTS_MOTOR):

    vits        (default) la voz de ROMEO: es_MX-claude-high, igual que la Pi.
    supertonic  la voz infantil M5 de Supertonic (ver modelos/voz-supertonic-m5/).
                Sube el tono +ROMEO_TTS_SEMITONOS y remuestrea, para que suene
                mas aguda sin cambiar la duracion (preset «Infantil»).

Variables de entorno:

    ROMEO_TTS_MOTOR      vits | supertonic   (default vits)
    ROMEO_TTS_VITS_DIR   carpeta de la voz   (default ~/.romeo/modelos/vits-piper-es_MX-claude-high)
    ROMEO_TTS_RUIDO      noise_scale         (default 0.667, nativo de Piper)
    ROMEO_TTS_RUIDO_W    noise_scale_w       (default 0.8, nativo de Piper)
    ROMEO_TTS_DURACION   length_scale        (default 1.0, velocidad natural)
    ROMEO_TTS_HILOS      hilos de onnxruntime (default 4)
    ROMEO_TTS_PUERTO     puerto de escucha   (default 8083)
    ROMEO_TTS_HOST       interfaz            (default 0.0.0.0)

  Solo con ROMEO_TTS_MOTOR=supertonic:
    ROMEO_TTS_SUPERTONIC_DIR  carpeta del modelo (default ~/ARFORD/modelos/voz-supertonic-m5
                              o modelos/voz-supertonic-m5 junto a este script)
    ROMEO_TTS_SID             voz de Supertonic   (default 4 = M5)
    ROMEO_TTS_SEMITONOS       cuanto sube el tono (default 5, preset Infantil)
    ROMEO_TTS_PASOS           num_steps           (default 12)
"""

import glob
import io
import os
import threading
import time
import wave

import numpy as np
from flask import Flask, Response, jsonify, request

MOTOR = os.environ.get("ROMEO_TTS_MOTOR", "vits").strip().lower()
VITS_DIR = os.path.expanduser(os.environ.get(
    "ROMEO_TTS_VITS_DIR", "~/.romeo/modelos/vits-piper-es_MX-claude-high"))
RUIDO = float(os.environ.get("ROMEO_TTS_RUIDO", "0.667"))
RUIDO_W = float(os.environ.get("ROMEO_TTS_RUIDO_W", "0.8"))
DURACION = float(os.environ.get("ROMEO_TTS_DURACION", "1.0"))
HILOS = int(os.environ.get("ROMEO_TTS_HILOS", "4"))
PUERTO = int(os.environ.get("ROMEO_TTS_PUERTO", "8083"))
HOST = os.environ.get("ROMEO_TTS_HOST", "0.0.0.0")

# Supertonic (solo con ROMEO_TTS_MOTOR=supertonic). La carpeta por defecto es
# la del repo; se prueba ~/ARFORD y la de junto al script, para que funcione
# tanto en la PC como en la Pi sin configurar nada.
def _supertonic_dir():
    env = os.environ.get("ROMEO_TTS_SUPERTONIC_DIR")
    if env:
        return os.path.expanduser(env)
    aqui = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.expanduser("~/ARFORD/modelos/voz-supertonic-m5"),
                 os.path.join(aqui, "..", "..", "..", "..", "modelos",
                              "voz-supertonic-m5")):
        if os.path.isdir(cand):
            return os.path.abspath(cand)
    return os.path.expanduser("~/ARFORD/modelos/voz-supertonic-m5")

SUPERTONIC_DIR = _supertonic_dir()
SID = int(os.environ.get("ROMEO_TTS_SID", "4"))
SEMITONOS = float(os.environ.get("ROMEO_TTS_SEMITONOS", "5"))
PASOS = int(os.environ.get("ROMEO_TTS_PASOS", "12"))

app = Flask(__name__)
_tts = None
_segundos_carga = 0.0
# Nombre de la voz activa, para /salud.
_voz_nombre = os.path.basename(VITS_DIR)
# Una síntesis a la vez, mismo motivo que el lock de stt_server.py: dos a la
# vez se reparten la CPU y salen más lentas las dos.
_lock = threading.Lock()


def cargar():
    """Carga el motor activo (MOTOR). Devuelve True si quedó lista."""
    if MOTOR == "supertonic":
        return _cargar_supertonic()
    return _cargar_vits()


def _cargar_vits():
    """Carga la voz VITS (es_MX-claude-high). Devuelve True si quedó lista."""
    global _tts, _segundos_carga, _voz_nombre
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
    _voz_nombre = os.path.basename(modelos[0])
    _segundos_carga = time.perf_counter() - t0
    print(f"[TTS] voz {_voz_nombre} lista en "
          f"{_segundos_carga:.1f}s ({HILOS} hilos)")
    return True


def _cargar_supertonic():
    """Carga la voz infantil Supertonic M5. Devuelve True si quedó lista.

    Los 6 archivos del modelo tienen que estar en SUPERTONIC_DIR (ver el
    README de modelos/voz-supertonic-m5). El tono no se sube aquí: se aplica
    en cada síntesis (ver _generar), igual que en el README."""
    global _tts, _segundos_carga, _voz_nombre
    import sherpa_onnx

    d = SUPERTONIC_DIR
    req = ["duration_predictor.int8.onnx", "text_encoder.int8.onnx",
           "vector_estimator.int8.onnx", "vocoder.int8.onnx",
           "tts.json", "unicode_indexer.bin", "voice.bin"]
    faltan = [f for f in req if not os.path.isfile(os.path.join(d, f))]
    if faltan:
        print(f"[TTS] faltan archivos de Supertonic en {d}: {faltan}")
        return False
    t0 = time.perf_counter()
    cfg = sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
        supertonic=sherpa_onnx.OfflineTtsSupertonicModelConfig(
            duration_predictor=os.path.join(d, "duration_predictor.int8.onnx"),
            text_encoder=os.path.join(d, "text_encoder.int8.onnx"),
            vector_estimator=os.path.join(d, "vector_estimator.int8.onnx"),
            vocoder=os.path.join(d, "vocoder.int8.onnx"),
            tts_json=os.path.join(d, "tts.json"),
            unicode_indexer=os.path.join(d, "unicode_indexer.bin"),
            voice_style=os.path.join(d, "voice.bin")),
        num_threads=HILOS))
    _tts = sherpa_onnx.OfflineTts(cfg)
    _voz_nombre = f"supertonic-M5 (sid={SID}, +{SEMITONOS:g} semitonos)"
    _segundos_carga = time.perf_counter() - t0
    print(f"[TTS] voz {_voz_nombre} lista en "
          f"{_segundos_carga:.1f}s ({HILOS} hilos)")
    return True


def _subir_tono(muestras, semitonos):
    """Sube el tono `semitonos` sin cambiar la duración: la síntesis ya salió
    más lenta (speed = 1/f en _generar), así que al remuestrear a la longitud
    original el tono sube y la duración vuelve a ser la de siempre. Igual que
    el README de la voz."""
    if not semitonos:
        return muestras
    f = 2 ** (semitonos / 12.0)
    n = int(muestras.size / f)
    if n < 2:
        return muestras
    espectro = np.fft.rfft(muestras)[:n // 2 + 1]
    return (np.fft.irfft(espectro, n) * (n / muestras.size)).astype(np.float32)


def _generar(texto):
    """Sintetiza `texto` con el motor activo -> (muestras float32, sample_rate).

    Para Supertonic aplica el preset Infantil: sintetiza más lento (speed=1/f)
    y después _subir_tono devuelve la duración normal con el tono más agudo."""
    if MOTOR == "supertonic":
        import sherpa_onnx
        f = 2 ** (SEMITONOS / 12.0)
        g = sherpa_onnx.GenerationConfig()
        g.sid, g.speed, g.num_steps, g.extra = SID, 1.0 / f, PASOS, {"lang": "es"}
        audio = _tts.generate(texto, g)
        muestras = np.asarray(audio.samples, dtype=np.float32)
        muestras = _subir_tono(muestras, SEMITONOS)
        return muestras, audio.sample_rate
    audio = _tts.generate(texto, sid=0)
    return np.asarray(audio.samples, dtype=np.float32), audio.sample_rate


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
        "motor": MOTOR,
        "voz": _voz_nombre,
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
        muestras, sr = _generar(texto)
    if muestras.size == 0:
        # El cliente trata un cuerpo vacío como fallo y sintetiza en local.
        return Response(b"", mimetype="audio/wav")
    datos = _a_wav(muestras, sr)
    print(f"[TTS] {time.perf_counter() - t0:.2f}s  "
          f"{muestras.size / sr:.1f}s de audio  {texto[:60]!r}")
    return Response(datos, mimetype="audio/wav")


def main():
    # Falla ruidosa, igual que stt_server.py: sin voz solo devolvería 503.
    print(f"[TTS] motor: {MOTOR}")
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
