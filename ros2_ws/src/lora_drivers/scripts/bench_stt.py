#!/usr/bin/env python3
"""Mide el STT en la Raspberry Pi antes de comprometerse con él.

Esta es la pieza de riesgo de toda la integración con la placa de audio: si
transcribir una frase tarda demasiado CON OLLAMA AL LADO, el resto del diseño
da igual. Por eso se mide primero, con el mismo criterio que se usó para sacar
el router y el corrector del LLM: manda la latencia medida, no la preferencia.

Uso en la Pi::

    # 1. Sin Ollama, para tener la línea base
    python3 bench_stt.py --tam base --repeticiones 5

    # 2. Con Ollama cargado, que es la condición real
    ollama run lora-chat-libre-v4 "hola" >/dev/null
    python3 bench_stt.py --tam base --repeticiones 5

    # 3. Comparar contra el modelo chico
    python3 bench_stt.py --tam tiny --repeticiones 5

Sin --wav, graba del micrófono. Con --wav, usa un archivo (mono, 16 bits),
que es la forma de comparar peras con peras entre corridas.

Qué mirar:
  - RTF (Real-Time Factor) = tiempo de proceso / duración del audio.
    RTF < 1 significa que transcribe más rápido de lo que dura el audio.
  - El tiempo absoluto por frase importa más que el RTF: el usuario espera ahí.
    Como referencia, en el turno del juego de emociones la voz ya se lleva 4.8s
    de 16.8s; si el STT suma varios segundos más, la conversación se rompe.
"""

import argparse
import os
import statistics
import sys
import time

# Permite correr el script sin instalar el paquete ROS 2.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lora_drivers._stt_engine import SAMPLE_RATE, ReconocedorVoz  # noqa: E402


def grabar(segundos):
    """Graba del micrófono y devuelve (samples, duración real)."""
    try:
        import sounddevice as sd
    except ImportError:
        print("Falta sounddevice: pip install sounddevice")
        return None, 0.0

    print(f"\nHabla ahora ({segundos}s)...")
    audio = sd.rec(
        int(segundos * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32"
    )
    sd.wait()
    print("Listo.\n")
    return audio.reshape(-1), float(segundos)


def duracion_wav(ruta):
    import wave

    with wave.open(ruta, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tam", default="base", help="tiny | base | small")
    p.add_argument("--hilos", type=int, default=2)
    p.add_argument("--idioma", default="es")
    p.add_argument("--repeticiones", type=int, default=3)
    p.add_argument("--segundos", type=float, default=4.0, help="duración a grabar")
    p.add_argument("--wav", help="usar este WAV en vez del micrófono")
    args = p.parse_args()

    print("=" * 60)
    print(f"Modelo whisper-{args.tam} | {args.hilos} hilos | idioma {args.idioma}")
    print(f"CPUs disponibles: {os.cpu_count()}")

    try:
        import subprocess

        ps = subprocess.run(
            ["pgrep", "-af", "ollama"], capture_output=True, text=True, timeout=5
        )
        print(f"Ollama corriendo: {'SÍ' if ps.stdout.strip() else 'no'}")
    except Exception:  # noqa: BLE001
        pass
    print("=" * 60)

    rec = ReconocedorVoz(tam=args.tam, hilos=args.hilos, idioma=args.idioma)
    if not rec.cargar():
        print("\nNo se pudo cargar el modelo. Revisa el docstring de _stt_engine.py")
        return 1
    print(f"Carga del modelo: {rec.segundos_carga:.2f}s\n")

    # Preparar el audio una sola vez, para que todas las repeticiones midan lo
    # mismo y la comparación sea honesta.
    if args.wav:
        if not os.path.exists(args.wav):
            print(f"No existe {args.wav}")
            return 1
        dur = duracion_wav(args.wav)
        samples = None
    else:
        samples, dur = grabar(args.segundos)
        if samples is None:
            return 1

    tiempos = []
    for i in range(1, args.repeticiones + 1):
        t0 = time.perf_counter()
        texto = rec.transcribir_wav(args.wav) if args.wav else rec.transcribir(samples)
        dt = time.perf_counter() - t0
        tiempos.append(dt)
        print(f"  {i}. {dt:6.2f}s  RTF {dt / dur:.2f}  ->  {texto!r}")

    print("\n" + "-" * 60)
    print(f"Audio de {dur:.1f}s")
    print(f"Mediana : {statistics.median(tiempos):.2f}s")
    print(f"Mín/Máx : {min(tiempos):.2f}s / {max(tiempos):.2f}s")
    print(f"RTF     : {statistics.median(tiempos) / dur:.2f}")
    print("-" * 60)
    print(
        "\nLa primera repetición suele ser la más lenta (cachés en frío);\n"
        "para decidir, mira la mediana y compárala con y sin Ollama cargado."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
