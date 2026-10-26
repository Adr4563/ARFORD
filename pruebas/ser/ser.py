#!/usr/bin/env python3
"""SER: emocion de la VOZ con el micro de la audio board.

    python3 ser.py                  # Enter, hablas 4 s, sale la emocion
    python3 ser.py --segundos 6
    python3 ser.py --wav frase.wav

Mismo modelo que usa stt_server.py en la PC
(somosnlp-hackathon-2022/wav2vec2-base-finetuned-sentiment-mesd, entrenado en
espanol de Mexico, 6 emociones), exportado a ONNX con
pruebas/stt-nemo/exportar_ser.py. Vive fuera del repo en ~/.romeo/modelos/ser/.

La tesis de Rocha no usa SER en sus escenarios (solo TER + FER): esto es
algo que agrega Romeo.

Independiente del robot y de la prueba de NeMo: solo numpy, onnxruntime y
sounddevice. Si el robot esta corriendo tiene el micro: paralo antes.
"""

import argparse
import json
import math
import os
import sys
import time
import wave

import numpy as np

DIR = os.environ.get("ROMEO_SER_DIR", os.path.expanduser("~/.romeo/modelos/ser"))
MIC = os.environ.get("ROMEO_MIC", "Board: USB Audio")   # = mic_dispositivo del robot
SR = 16000
TRADUCCION = {"Anger": "enojo", "Disgust": "asco", "Fear": "miedo",
              "Happiness": "felicidad", "Neutral": "neutral", "Sadness": "tristeza"}
ANCLAJES = {
    "neutral": (0.00, 0.00, 0.00), "felicidad": (0.76, 0.48, 0.35),
    "tristeza": (-0.63, -0.27, -0.33), "enojo": (-0.51, 0.59, 0.25),
    "asco": (-0.60, 0.35, 0.11), "miedo": (-0.64, 0.60, -0.43),
}


class Ser:

    def __init__(self, hilos=4):
        # 4 hilos: medido en la Pi con una frase de 3.1 s, 2 hilos 2.85 s,
        # 3 hilos 2.1 s, 4 hilos 1.85 s.
        import onnxruntime as ort
        with open(os.path.join(DIR, "ser.json"), encoding="utf-8") as f:
            meta = json.load(f)
        self.etiquetas = [TRADUCCION[meta["id2label"][str(i)]] for i in range(len(meta["id2label"]))]
        op = ort.SessionOptions()
        op.intra_op_num_threads = hilos
        self._sesion = ort.InferenceSession(os.path.join(DIR, "ser.onnx"), op)

    def clasificar(self, x):
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        # Lo que hace Wav2Vec2FeatureExtractor: media 0 y varianza 1. De paso
        # hace que el volumen bajo de la placa no importe.
        x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
        z = self._sesion.run(None, {"input_values": x[None, :]})[0][0]
        e = np.exp(z - z.max())
        return e / e.sum()


def confianza(p):
    h = -sum(v * math.log(v) for v in p if v > 0) / math.log(len(p))
    return 1.0 - h


def mostrar(ser, audio):
    pico = float(np.abs(audio).max()) if len(audio) else 0.0
    t0 = time.perf_counter()
    p = ser.clasificar(audio)
    ms = (time.perf_counter() - t0) * 1000
    orden = sorted(zip(ser.etiquetas, p), key=lambda kv: -kv[1])[:3]
    v, a, d = (sum(pi * ANCLAJES[e][k] for pi, e in zip(p, ser.etiquetas)) for k in range(3))
    print(f"  {', '.join(f'{e} {x:.0%}' for e, x in orden)}\n"
          f"      confianza {confianza(p):.2f}, VAD ({v:+.2f}, {a:+.2f}, {d:+.2f}), "
          f"{len(audio) / SR:.1f} s de audio, pico {pico:.3f}, {ms:.0f} ms", flush=True)


def leer_wav(ruta):
    with wave.open(ruta, "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != SR:
            sys.exit("Se espera WAV mono, 16 bits, 16 kHz (ffmpeg -i x -ar 16000 -ac 1 y.wav)")
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wav")
    ap.add_argument("--segundos", type=float, default=4.0)
    ap.add_argument("--mic", default=MIC)
    args = ap.parse_args()

    ser = Ser()
    if args.wav:
        mostrar(ser, leer_wav(args.wav))
        return 0

    import sounddevice as sd
    print(f"Enter para grabar {args.segundos:g} s (Ctrl+C para salir). "
          f"Di una frase con emocion: alegre, triste, enojado...", flush=True)
    try:
        while True:
            input()
            print("  grabando...", flush=True)
            audio = sd.rec(int(args.segundos * SR), samplerate=SR, channels=1,
                           dtype="float32", device=args.mic)
            sd.wait()
            mostrar(ser, audio.reshape(-1))
    except (KeyboardInterrupt, EOFError):
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
