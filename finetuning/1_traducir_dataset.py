"""Traduce Empathetic Dialogues (Rashkin et al. 2019) al espanol.

Escribe a disco cada conversacion terminada, asi puedes ver el avance en
cualquier momento con:  wc -l data/empatico_es.jsonl

Si se corta, al volver a lanzarlo continua donde quedo.

    python 1_traducir_dataset.py              # 2000 conversaciones
    python 1_traducir_dataset.py 19533 24     # todas, 24 hilos
"""
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from datasets import load_dataset
from ollama import Client

MODELO = "llama3.2:3b"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
HILOS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
SALIDA = "data/empatico_es.jsonl"

SYSTEM = (
    "Traduce al espanol de forma natural y cercana. Devuelve SOLO la traduccion, "
    "sin comillas, sin notas, sin el texto original. Manten el tono emocional."
)

cliente = Client(host="http://127.0.0.1:11434")
ds = load_dataset("Estwld/empathetic_dialogues_llm", split="train")
N = min(N, len(ds))

# --- Reanudacion: que conv_id ya estan en el archivo ---
hechos = set()
if os.path.exists(SALIDA):
    with open(SALIDA, encoding="utf-8") as f:
        for linea in f:
            try:
                hechos.add(json.loads(linea)["conv_id"])
            except Exception:
                pass

pendientes = [i for i in range(N) if ds[i]["conv_id"] not in hechos]
print(f"Dataset: {len(ds)} conversaciones")
print(f"Ya traducidas: {len(hechos)}  |  Pendientes: {len(pendientes)}  |  Hilos: {HILOS}")
if not pendientes:
    print("Nada que hacer.")
    sys.exit(0)

lock = threading.Lock()
salida = open(SALIDA, "a", encoding="utf-8")
hechas = [0]
t0 = time.perf_counter()


def traducir(texto):
    r = cliente.chat(
        model=MODELO,
        options={"temperature": 0, "num_predict": 200},
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": texto}],
    )
    return r.message.content.strip()


def procesa(i):
    fila = ds[i]
    try:
        msgs = [{"role": m["role"], "content": traducir(m["content"])}
                for m in fila["conversations"]]
    except Exception as exc:
        print(f"  [error {i}] {str(exc)[:60]}", flush=True)
        return

    registro = {"conv_id": fila["conv_id"], "emotion": fila["emotion"], "messages": msgs}

    # Escribe y vacia el buffer: la linea queda en disco de inmediato
    with lock:
        salida.write(json.dumps(registro, ensure_ascii=False) + "\n")
        salida.flush()
        hechas[0] += 1
        if hechas[0] % 25 == 0:
            vel = hechas[0] / (time.perf_counter() - t0)
            falta = (len(pendientes) - hechas[0]) / vel / 60
            print(f"  {hechas[0]}/{len(pendientes)}  ({vel:.1f} conv/s, faltan {falta:.0f} min)",
                  flush=True)


try:
    with ThreadPoolExecutor(max_workers=HILOS) as pool:
        list(pool.map(procesa, pendientes))
finally:
    salida.close()

total = sum(1 for _ in open(SALIDA, encoding="utf-8"))
print(f"\nListo. {SALIDA} tiene {total} conversaciones ({(time.perf_counter()-t0)/60:.1f} min)")
