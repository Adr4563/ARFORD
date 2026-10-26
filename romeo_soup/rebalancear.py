"""Baja el porcentaje de respuestas que acaban en pregunta.

EL PROBLEMA. EmpatheticDialogues son turnos de *listener*: el 68% de las
respuestas llevan pregunta. Un modelo chico aprende ESE MOLDE antes que
el contenido, y acaba cerrando SIEMPRE igual -- en el log del robot,
romeo-empatico-v2 repetia incluso la pregunta que acababa de hacer.

QUE HACE. De las 1006 respuestas que son "afirmacion + pregunta", a una
parte se le quita la pregunta y se queda solo la afirmacion. No se borra
ningun par: el dataset conserva su tamaño y los 1825 ejemplos siguen
enseñando a responder, solo que ya no todos terminan preguntando.

Las que son SOLO pregunta (237) no se tocan: sin ella no queda nada.
Los ejemplos de identidad tampoco, porque su pregunta es la que devuelve
el turno ("Me llamo Romeo. ¿Y tu como te llamas?").

    python rebalancear.py            # deja ~40% con pregunta
    python rebalancear.py 0.5        # otro objetivo
"""
import json
import os
import random
import re
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
ORIGEN = os.path.join(AQUI, "data", "romeo_empatico_v2.jsonl")
DESTINO = os.path.join(AQUI, "data", "romeo_empatico_v3.jsonl")
OBJETIVO = float(sys.argv[1]) if len(sys.argv) > 1 else 0.40

# No tocar lo que enseña quien es Romeo: ahi la pregunta devuelve el turno.
IDENTIDAD = re.compile(r"\b(romeo|robot|me llamo|soy)\b", re.I)


def sin_pregunta(texto):
    """La respuesta sin su ultima pregunta, o None si no queda nada util."""
    partes = [p for p in re.split(r"(?<=[.!?])\s+", texto.strip()) if p]
    afirmaciones = [p for p in partes if "?" not in p]
    if not afirmaciones:
        return None
    recortada = " ".join(afirmaciones).strip()
    # Menos de esto ("Claro.") no dice nada por si solo.
    return recortada if len(recortada) >= 12 else None


filas = [json.loads(l) for l in open(ORIGEN, encoding="utf-8") if l.strip()]
con = [i for i, m in enumerate(filas) if "?" in m["messages"][2]["content"]]
print(f"entrada: {len(filas)} pares, {len(con)} con pregunta "
      f"({100 * len(con) / len(filas):.0f}%)")

# Candidatas: tienen pregunta, no son de identidad, y queda algo al cortar.
candidatas = []
for i in con:
    r = filas[i]["messages"][2]["content"]
    if IDENTIDAD.search(r):
        continue
    corte = sin_pregunta(r)
    if corte:
        candidatas.append((i, corte))

sobran = len(con) - int(len(filas) * OBJETIVO)
print(f"objetivo {OBJETIVO:.0%}: hay que quitar la pregunta a {sobran}")
print(f"candidatas (afirmacion + pregunta, sin identidad): {len(candidatas)}")

random.seed(42)
random.shuffle(candidatas)
cambiadas = 0
for i, corte in candidatas[:max(0, sobran)]:
    filas[i]["messages"][2]["content"] = corte
    cambiadas += 1

quedan = sum(1 for m in filas if "?" in m["messages"][2]["content"])
print(f"\nrecortadas: {cambiadas}")
print(f"salida: {len(filas)} pares, {quedan} con pregunta "
      f"({100 * quedan / len(filas):.0f}%)")

with open(DESTINO, "w", encoding="utf-8") as f:
    for m in filas:
        f.write(json.dumps(m, ensure_ascii=False) + "\n")
print(f"escrito: {DESTINO}")
