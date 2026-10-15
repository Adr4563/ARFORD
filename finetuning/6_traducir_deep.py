"""Traduce Empathetic Dialogues EN->ES con deep-translator (MyMemory).

No usa ningun modelo local: la traduccion la hace un servicio externo.
Escribe cada conversacion al terminarla, asi puedes ver el avance con:
    wc -l data/empatico_deep_es.jsonl
Si se corta, al relanzarlo continua donde quedo.

    python 6_traducir_deep.py            # 2000 conversaciones, 8 hilos
    python 6_traducir_deep.py 500 12     # 500 conversaciones, 12 hilos
"""
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from datasets import load_dataset
from deep_translator import MyMemoryTranslator

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
HILOS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
SALIDA = "data/empatico_deep_es.jsonl"

ds = load_dataset("Estwld/empathetic_dialogues_llm", split="train")
N = min(N, len(ds))

# --- Reanudacion ---
hechos = set()
if os.path.exists(SALIDA):
    with open(SALIDA, encoding="utf-8") as f:
        for linea in f:
            try:
                hechos.add(json.loads(linea)["conv_id"])
            except Exception:
                pass

pendientes = [i for i in range(N) if ds[i]["conv_id"] not in hechos]
print(f"Ya traducidas: {len(hechos)}  |  Pendientes: {len(pendientes)}  |  Hilos: {HILOS}")
if not pendientes:
    print("Nada que hacer.")
    sys.exit(0)

lock = threading.Lock()
# MyMemory permite 5 peticiones/s: nos quedamos en 4 para tener margen
ritmo = threading.Semaphore(1)
ultima = [0.0]
PAUSA = 0.25

salida = open(SALIDA, "a", encoding="utf-8")
hechas = [0]
fallos = [0]
t0 = time.perf_counter()
local = threading.local()


def traductor():
    # Un traductor por hilo: la libreria no es thread-safe
    if not hasattr(local, "t"):
        local.t = MyMemoryTranslator(source="en-GB", target="es-ES")
    return local.t


def traducir(texto):
    # MyMemory corta a 500 caracteres
    texto = texto.strip()[:499]
    for intento in range(5):
        try:
            with ritmo:
                espera = PAUSA - (time.perf_counter() - ultima[0])
                if espera > 0:
                    time.sleep(espera)
                ultima[0] = time.perf_counter()
            r = traductor().translate(texto)
            if r and r.strip():
                return r.strip()
        except Exception:
            time.sleep(5 * (intento + 1))
    return None


def procesa(i):
    fila = ds[i]
    msgs = []
    for m in fila["conversations"]:
        t = traducir(m["content"])
        if t is None:
            with lock:
                fallos[0] += 1
            return
        msgs.append({"role": m["role"], "content": t})

    registro = {"conv_id": fila["conv_id"], "emotion": fila["emotion"], "messages": msgs}
    with lock:
        salida.write(json.dumps(registro, ensure_ascii=False) + "\n")
        salida.flush()
        hechas[0] += 1
        if hechas[0] % 10 == 0:
            vel = hechas[0] / (time.perf_counter() - t0)
            falta = (len(pendientes) - hechas[0]) / vel / 60
            print(f"  {hechas[0]}/{len(pendientes)}  ({vel:.2f} conv/s, "
                  f"faltan {falta:.0f} min, {fallos[0]} fallos)", flush=True)


try:
    with ThreadPoolExecutor(max_workers=HILOS) as pool:
        list(pool.map(procesa, pendientes))
finally:
    salida.close()

total = sum(1 for _ in open(SALIDA, encoding="utf-8"))
print(f"\n{SALIDA}: {total} conversaciones  ({(time.perf_counter()-t0)/60:.1f} min, "
      f"{fallos[0]} fallos)")
