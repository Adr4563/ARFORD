#!/usr/bin/env python3
"""Habla al micro de la audio board y PocketSphinx transcribe frase a frase.

    python3 en_vivo.py                     # CIEMPIESS, habla libre
    python3 en_vivo.py --modelo cmu
    python3 en_vivo.py --comandos          # solo las frases de comandos.txt

El audio lo graba `arecord` directo de la placa y va por tuberia a
`pocketsphinx live -`, que corta las frases solo. Por cada frase imprime el
texto y la LATENCIA: cuanto tardo en llegar el texto desde que terminaste
de hablar.

Si el robot esta corriendo, communication_node tiene ocupado el micro:
paralo antes. Ctrl+C para salir.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time

import modelos
from gramatica import crear_gramatica

AQUI = os.path.dirname(os.path.abspath(__file__))
# Mismo nombre ALSA que usa el robot para el parlante (audio_salida en
# romeo_params.yaml): CARD=Board es la audio board, no cambia entre arranques.
MIC = os.environ.get("ROMEO_MIC_ALSA", "plughw:CARD=Board,DEV=0")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--modelo", choices=sorted(modelos.MODELOS), default="ciempiess")
    ap.add_argument("--comandos", action="store_true",
                    help="solo reconoce las frases de comandos.txt (gramatica JSGF)")
    ap.add_argument("--mic", default=MIC)
    args = ap.parse_args()

    falta = modelos.faltantes(args.modelo)
    if falta:
        print(f"Falta {falta[0]}. Corre primero:  bash instalar_modelos.sh")
        return 1
    gram = crear_gramatica(args.modelo) if args.comandos else None

    print(f"Cargando {args.modelo} (~9 s en la Pi)...", flush=True)
    sphinx = modelos.lanzar(args.modelo, gram)
    grab = subprocess.Popen(
        ["arecord", "-q", "-D", args.mic, "-f", "S16_LE", "-r", "16000",
         "-c", "1", "-t", "raw"], stdout=subprocess.PIPE)

    # Los tiempos de pocketsphinx ("b", "d") cuentan segundos de AUDIO desde
    # el primer byte que recibio. Se marca cuando sale ese primer byte para
    # pasarlos a hora real y poder medir la latencia.
    inicio = {}

    def bombear():
        while True:
            datos = grab.stdout.read(3200)  # 100 ms
            if not datos:
                break
            inicio.setdefault("t", time.monotonic())
            try:
                sphinx.stdin.write(datos)
                sphinx.stdin.flush()
            except BrokenPipeError:
                break

    threading.Thread(target=bombear, daemon=True).start()
    print("Habla (Ctrl+C para salir). La primera frase suele salir peor: "
          "el modelo se esta ajustando a tu micro.\n", flush=True)
    try:
        for linea in sphinx.stdout:
            llegada = time.monotonic()
            r = json.loads(linea)
            fin_voz = inicio.get("t", llegada) + r["b"] + r["d"]
            texto = r["t"] or "(nada)"
            print(f"  {texto}\n      {r['d']:.1f} s de voz, latencia "
                  f"{llegada - fin_voz:.1f} s", flush=True)
    except KeyboardInterrupt:
        print()
    finally:
        grab.terminate()
        sphinx.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
