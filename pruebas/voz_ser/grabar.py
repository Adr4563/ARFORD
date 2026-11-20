"""Graba audios actuados para comparar_ser.py.

Uso interactivo (original):  python grabar.py
Una emocion, sin input():    python grabar.py --disp 17 --emocion neutral
En esta PC la placa ESP32-S3 (usb uac) capta bien solo por el indice 17 (WDM-KS).
"""
import argparse
import os
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

SEG, TOMAS, FS = 5, 3, 16000
CARPETA = "audios/actuados"
FRASES = {"triste": "me siento muy mal hoy",
          "feliz": "estoy muy contento, qué buen día",
          "enojado": "estoy harto, esto me molesta mucho",
          "neutral": "hoy es miércoles y son las dos"}


def nivel(x):
    rms = float(np.sqrt(np.mean(x ** 2)))
    db = 20 * np.log10(rms + 1e-9)
    pico = float(np.max(np.abs(x)))
    sat = int((np.abs(x) > 0.99).sum())
    aviso = "  <-- BAJO, repetir mas cerca" if db < -50 else ""
    aviso += "  <-- SATURADO" if sat else ""
    return f"rms {db:6.1f} dBFS  pico {pico:.3f}  saturadas {sat}{aviso}"


def pitido(hz, seg=0.25, veces=1):
    """Aviso sonoro por la salida predeterminada (para hablar sin mirar la pantalla)."""
    n = np.arange(int(seg * 44100)) / 44100
    tono = (0.3 * np.sin(2 * np.pi * hz * n)).astype("float32")
    for _ in range(veces):
        sd.play(tono, 44100)
        sd.wait()
        time.sleep(0.1)


def tomar(disp, emocion, t):
    x = sd.rec(int(SEG * FS), samplerate=FS, channels=1, device=disp, dtype="float32")
    sd.wait()
    os.makedirs(CARPETA, exist_ok=True)
    ruta = f"{CARPETA}/{emocion}_{t}.wav"
    sf.write(ruta, x, FS)
    print(f"guardado {ruta}  {nivel(x[:, 0])}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disp", type=int)
    ap.add_argument("--emocion", choices=list(FRASES))
    ap.add_argument("--carpeta", default=CARPETA, help="p. ej. audios/persona2")
    a = ap.parse_args()
    globals()["CARPETA"] = a.carpeta

    if a.emocion is None:  # modo original interactivo
        print(sd.query_devices())
        disp = a.disp if a.disp is not None else int(input("Índice del micrófono: "))
        for emocion, frase in FRASES.items():
            for t in range(1, TOMAS + 1):
                input(f"[{emocion} {t}/{TOMAS}] di con ese tono: «{frase}» (Enter y habla)")
                tomar(disp, emocion, t)
        return

    frase = FRASES[a.emocion]
    for t in range(1, TOMAS + 1):
        print(f"\n[{a.emocion} {t}/{TOMAS}] frase: «{frase}»", flush=True)
        for c in (3, 2, 1):
            print(f"  {c}...", flush=True)
            time.sleep(1)
        print(f"  >>> HABLA AHORA ({SEG} s)", flush=True)
        pitido(880)                      # 1 pitido agudo = habla
        tomar(a.disp, a.emocion, t)
        pitido(440, 0.15, veces=2)       # 2 pitidos graves = fin de la toma
        time.sleep(1.5)
    print("\n>>> FIN de esta emoción", flush=True)


if __name__ == "__main__":
    main()
