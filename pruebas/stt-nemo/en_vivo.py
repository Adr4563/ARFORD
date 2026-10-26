#!/usr/bin/env python3
"""Habla al micro de la audio board y NeMo (FastConformer espanol) transcribe
EN LA PI, sin la PC.

    python3 en_vivo.py               # micro en vivo, Ctrl+C para salir
    python3 en_vivo.py --wav a.wav   # un archivo WAV mono 16 bits
    python3 en_vivo.py --hilos 4

No es un motor nuevo: usa el mismo ReconocedorVoz del robot
(romeo_drivers/_stt_engine.py) con motor="nemo" y sin servidor, y el mismo
VAD (silero) para cortar las frases. Asi lo que se mide aca es exactamente
lo que haria communication_node con `stt_motor: nemo` y `stt_servidor: ""`.

NeMo SI es una red neuronal (no cumple "sin IA"). El modelo se baja una vez:

    cd ~/.romeo/modelos/asr
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-fast-conformer-transducer-es-1424-int8.tar.bz2
    tar xjf sherpa-onnx-nemo-fast-conformer-transducer-es-1424-int8.tar.bz2

Si el robot esta corriendo, tiene ocupado el micro: paralo antes.
"""

import argparse
import os
import sys
import time

AQUI = os.path.dirname(os.path.abspath(__file__))
# El paquete romeo_drivers esta en ros2_ws/src/ en el repo, y en src/ en la
# Pi (alli ~/ros2_ws ES el repo). Se prueban las dos.
for rel in ("../../ros2_ws/src/romeo_drivers", "../../src/romeo_drivers"):
    ruta = os.path.normpath(os.path.join(AQUI, rel))
    if os.path.isdir(os.path.join(ruta, "romeo_drivers")):
        sys.path.insert(0, ruta)
        break

from romeo_drivers._stt_engine import ReconocedorVoz  # noqa: E402
from ser_onnx import ClasificadorVozOnnx  # noqa: E402

# Mismo valor que mic_dispositivo en romeo_params.yaml.
MIC = os.environ.get("ROMEO_MIC", "Board: USB Audio")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wav")
    ap.add_argument("--mic", default=MIC)
    ap.add_argument("--hilos", type=int, default=2)
    ap.add_argument("--sin-ser", action="store_true", help="no carga el SER")
    # Medido en la Pi, frase de 3.1 s: 2 hilos 2.85 s, 3 hilos 2.1 s, 4 hilos
    # 1.85 s. La version int8 fue PEOR (3.2 s) y cambiaba la emocion.
    ap.add_argument("--hilos-ser", type=int, default=4)
    args = ap.parse_args()

    rec = ReconocedorVoz(motor="nemo", servidor="", hilos=args.hilos)
    print("Cargando NeMo...", flush=True)
    if not rec.cargar():
        return 1

    ser = None
    if not args.sin_ser:
        try:
            t0 = time.perf_counter()
            ser = ClasificadorVozOnnx(hilos=args.hilos_ser)
            print(f"[SER] listo en {time.perf_counter() - t0:.1f}s", flush=True)
        except Exception as e:  # noqa: BLE001 - sin SER, el STT sigue
            print(f"[SER] desactivado: {e}", flush=True)

    if args.wav:
        # Se pasa por transcribir_medido (abajo) para que tambien salga el SER.
        args_wav = args.wav
    else:
        args_wav = None

    # Los tiempos se miden aca envolviendo transcribir(), y no con
    # rec.ultima_frase_fin: el _stt_engine.py que esta en la Pi es anterior y
    # no tiene esos atributos.
    medida = {}
    original = rec.transcribir

    def transcribir_medido(samples, sample_rate=16000):
        medida["voz"] = len(samples) / sample_rate
        t0 = time.perf_counter()
        # El SER va sobre el audio CRUDO, antes de la ganancia del STT, como
        # en stt_server.py. Igual normaliza por su cuenta.
        medida["ser"] = ser.clasificar(samples, sample_rate) if ser else None
        medida["t_ser"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        texto = original(samples, sample_rate)
        medida["proceso"] = time.perf_counter() - t0
        return texto

    rec.transcribir = transcribir_medido

    def al_transcribir(texto):
        # Lo que espera el usuario: el silencio que el VAD necesita para dar
        # la frase por terminada, mas lo que tardan NeMo y el SER.
        silencio = getattr(sys.modules[type(rec).__module__], "MIN_SILENCIO", 0.0)
        total = medida["proceso"] + medida["t_ser"]
        print(f"  {texto}\n      {medida['voz']:.1f} s de voz, NeMo "
              f"{medida['proceso']:.2f} s, SER {medida['t_ser']:.2f} s, latencia "
              f"{total + silencio:.1f} s "
              f"(ganancia x{getattr(rec, 'ultima_ganancia', 1.0):.1f})", flush=True)
        if medida["ser"]:
            orden = sorted(zip(*medida["ser"]), key=lambda kv: -kv[1])[:3]
            print("      voz: " + ", ".join(f"{e} {p:.0%}" for e, p in orden), flush=True)
        elif ser:
            print("      voz: (frase muy corta para el SER)", flush=True)

    if args_wav:
        texto = rec.transcribir_wav(args_wav)
        al_transcribir(texto)
        return 0

    print("Habla (Ctrl+C para salir).\n", flush=True)
    try:
        rec.escuchar(al_transcribir, dispositivo=args.mic or None)
    except KeyboardInterrupt:
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
