"""Descarga ALIA (SINAI) — dialogos clinicos ya escritos en espanol.

No necesita traduccion. Normaliza los dos subsets al formato ChatML y los
combina con el dataset traducido si existe.

    python 1b_dataset_es_nativo.py
"""
import json
import os

from datasets import load_dataset

REPO = "SINAI/ALIA-es-clinical-psychology-dialogues"
SALIDA = "data/alia_es.jsonl"
COMBINADO = "data/entrenamiento.jsonl"

filas = []

# --- TerapIA: ya viene como messages (role/content) ---
d = load_dataset(REPO, "TerapIA", split="train")
for fila in d:
    msgs = [m for m in fila["messages"] if m["role"] in ("user", "assistant")]
    if len(msgs) >= 2:
        filas.append({"fuente": "terapia", "messages": msgs})
print(f"TerapIA: {len(d)} filas -> {len(filas)} validas")

# --- clinic-roleplay: usa from/value y trae un system prompt ---
n = len(filas)
d = load_dataset(REPO, "clinic-roleplay", split="train")
for fila in d:
    msgs = [{"role": m["from"], "content": m["value"]}
            for m in fila["conversations"] if m["from"] in ("user", "assistant")]
    if len(msgs) >= 2:
        filas.append({"fuente": "clinica", "messages": msgs})
print(f"clinic-roleplay: {len(d)} filas -> {len(filas)-n} validas")

with open(SALIDA, "w", encoding="utf-8") as f:
    for fila in filas:
        f.write(json.dumps(fila, ensure_ascii=False) + "\n")
print(f"\nGuardado: {SALIDA}  ({len(filas)} conversaciones en espanol nativo)")

# --- Combina con el traducido ---
total = list(filas)
TRADUCIDO = "data/empatico_es.jsonl"
if os.path.exists(TRADUCIDO):
    with open(TRADUCIDO, encoding="utf-8") as f:
        for linea in f:
            try:
                r = json.loads(linea)
                total.append({"fuente": "empathetic", "messages": r["messages"]})
            except Exception:
                pass
    print(f"Traducido:  {len(total)-len(filas)} conversaciones")

with open(COMBINADO, "w", encoding="utf-8") as f:
    for fila in total:
        f.write(json.dumps(fila, ensure_ascii=False) + "\n")

print(f"\n=== {COMBINADO}: {len(total)} conversaciones ===")
for src in ("clinica", "terapia", "empathetic"):
    n = sum(1 for x in total if x["fuente"] == src)
    if n:
        print(f"  {src:12} {n:5}")

print("\nEjemplo (clinica):")
ej = next(x for x in total if x["fuente"] == "clinica")
for m in ej["messages"][:4]:
    print(f"  {m['role']}: {m['content'][:95]}")
