# -*- coding: utf-8 -*-
"""Compara los modelos de STT instalados con un WAV de voz real.

Se ejecuta EN LA RASPBERRY, que es donde estan los modelos y donde importa la
velocidad:

    scp comparar_stt.py pi@<ip>:/tmp/
    ssh pi@<ip> 'python3 /tmp/comparar_stt.py /tmp/voz.wav'

Mide, para cada modelo: lo que transcribe, cuanto tarda y el RTF (tiempo de
proceso / duracion del audio). RTF < 1 significa mas rapido que tiempo real.
"""
import os
import sys
import time
import wave

sys.path.insert(0, "/home/pi/ros2_ws/install/romeo_drivers/lib/python3.14/site-packages")
import numpy as np
import sherpa_onnx

WAV = sys.argv[1] if len(sys.argv) > 1 else "/tmp/voz.wav"
MODELOS = os.path.expanduser("~/.romeo/modelos")
ASR = f"{MODELOS}/asr"
HILOS = 2   # el optimo medido en la Pi, ver romeo_params.yaml


def whisper(dir_, tam):
    return dict(
        encoder=f"{dir_}/{tam}-encoder.int8.onnx",
        decoder=f"{dir_}/{tam}-decoder.int8.onnx",
        tokens=f"{dir_}/{tam}-tokens.txt",
        language="es", task="transcribe", num_threads=HILOS)


CANDIDATOS = [
    ("whisper-tiny", "whisper", whisper(f"{MODELOS}/sherpa-onnx-whisper-tiny", "tiny")),
    ("whisper-base", "whisper", whisper(f"{MODELOS}/sherpa-onnx-whisper-base", "base")),
    ("whisper-small", "whisper", whisper(f"{ASR}/sherpa-onnx-whisper-small", "small")),
    ("moonshine-base-es", "moonshine", dict(
        preprocessor=f"{ASR}/sherpa-onnx-moonshine-base-es-quantized-2026-02-27/preprocess.onnx",
        encoder=f"{ASR}/sherpa-onnx-moonshine-base-es-quantized-2026-02-27/encoder_model.ort",
        uncached_decoder=f"{ASR}/sherpa-onnx-moonshine-base-es-quantized-2026-02-27/decoder_model_merged.ort",
        cached_decoder=f"{ASR}/sherpa-onnx-moonshine-base-es-quantized-2026-02-27/decoder_model_merged.ort",
        tokens=f"{ASR}/sherpa-onnx-moonshine-base-es-quantized-2026-02-27/tokens.txt",
        num_threads=HILOS)),
    ("nemo-ctc-es", "nemo_ctc", dict(
        model=f"{ASR}/sherpa-onnx-nemo-fast-conformer-ctc-es-1424-int8/model.int8.onnx",
        tokens=f"{ASR}/sherpa-onnx-nemo-fast-conformer-ctc-es-1424-int8/tokens.txt",
        num_threads=HILOS)),
    ("nemo-transducer-es", "transducer", dict(
        encoder=f"{ASR}/sherpa-onnx-nemo-fast-conformer-transducer-es-1424-int8/encoder.int8.onnx",
        decoder=f"{ASR}/sherpa-onnx-nemo-fast-conformer-transducer-es-1424-int8/decoder.int8.onnx",
        joiner=f"{ASR}/sherpa-onnx-nemo-fast-conformer-transducer-es-1424-int8/joiner.int8.onnx",
        tokens=f"{ASR}/sherpa-onnx-nemo-fast-conformer-transducer-es-1424-int8/tokens.txt",
        num_threads=HILOS, model_type="nemo_transducer")),
    ("canary-180m", "canary", dict(
        encoder=f"{ASR}/sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/encoder.int8.onnx",
        decoder=f"{ASR}/sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/decoder.int8.onnx",
        tokens=f"{ASR}/sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/tokens.txt",
        src_lang="es", tgt_lang="es", num_threads=HILOS)),
]

CONSTRUCTOR = {
    "whisper": sherpa_onnx.OfflineRecognizer.from_whisper,
    "moonshine": sherpa_onnx.OfflineRecognizer.from_moonshine,
    "nemo_ctc": sherpa_onnx.OfflineRecognizer.from_nemo_ctc,
    "transducer": sherpa_onnx.OfflineRecognizer.from_transducer,
    "canary": getattr(sherpa_onnx.OfflineRecognizer, "from_canary", None),
}

with wave.open(WAV) as w:
    audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    sr = w.getframerate()
duracion = len(audio) / sr
print(f"Audio: {WAV}  ({duracion:.1f} s, {sr} Hz, pico {100 * abs(audio).max():.1f}%)")
print()

for nombre, tipo, kwargs in CANDIDATOS:
    constructor = CONSTRUCTOR.get(tipo)
    if constructor is None:
        print(f"  {nombre:<20} sin soporte en esta version de sherpa-onnx")
        continue
    falta = [v for k, v in kwargs.items()
             if isinstance(v, str) and ("/" in v) and not os.path.exists(v)]
    if falta:
        print(f"  {nombre:<20} falta {os.path.basename(falta[0])}")
        continue
    try:
        t0 = time.perf_counter()
        rec = constructor(**kwargs)
        carga = time.perf_counter() - t0

        t0 = time.perf_counter()
        s = rec.create_stream()
        s.accept_waveform(sr, audio)
        rec.decode_stream(s)
        tiempo = time.perf_counter() - t0

        print(f"  {nombre:<20} {tiempo:5.1f}s  RTF {tiempo / duracion:4.2f}  "
              f"(carga {carga:.1f}s)")
        print(f"  {'':<20} -> {s.result.text.strip()[:95]!r}")
    except Exception as e:
        print(f"  {nombre:<20} ERROR: {type(e).__name__}: {str(e)[:70]}")
    print()
