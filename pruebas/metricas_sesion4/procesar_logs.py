# -*- coding: utf-8 -*-
"""Reconstruye, DESDE CERO y desde el log completo, el flujo de cada turno de la
sesion 4 (optimista) y saca las metricas por turno.

Flujo de un turno (todo medido por los timestamps del propio log de Romeo):
  [mic] transcrito      -> el micro capturo y el STT transcribio
  [afecto] texto expresa -> TER (emocion del texto)
  [afecto] camara vio    -> FER (emocion de la cara)
  [afecto] voz suena     -> SER (emocion de la voz)
  [afecto] usuario (fusion) -> PULSE (fusion VAD)
  [afecto] v ..->..         -> ROSE (empatia + decaimiento)
  [aeb] ... -> emocion_Lx   -> AEB (comportamiento: cara + movimiento)
  Asistente: (respuesta)    -> la IA respondio

El tiempo del STT no esta en el log (solo "[mic] transcrito"), pero SI en
percepcion.csv (t_stt_ms, medido en la Pi). Se empareja por el texto transcrito.

Salida: turnos_reconstruidos.csv  con una fila por turno y los tiempos de cada
etapa, mas la emocion/nivel/cara/movimiento resultante.
"""
import re
import csv
import sys

LOG = sys.argv[1] if len(sys.argv) > 1 else "romeo_run.log"
PERCEP = sys.argv[2] if len(sys.argv) > 2 else "percepcion.csv"
SALIDA = sys.argv[3] if len(sys.argv) > 3 else "turnos_reconstruidos.csv"

TS = re.compile(r"\[(\d+\.\d+)\]")


def ts(linea):
    m = TS.search(linea)
    return float(m.group(1)) if m else None


# --- 1. percepcion.csv: STT/SER/TER (medidos en la Pi) ---
# Se guarda por texto (match preferente) y tambien la LISTA en orden, porque el
# texto a veces difiere un poco entre el log y el CSV; si el texto no empareja,
# se usa el orden (ambos van en la misma secuencia temporal de turnos).
percep = {}
percep_orden = []
with open(PERCEP, encoding="utf-8", errors="replace") as f:
    for row in csv.DictReader(f):
        txt = (row.get("texto") or "").strip().lower()[:30]
        if txt:
            percep[txt] = row
            percep_orden.append(row)

# --- 2. recorrer el log y cortar en turnos ---
# Un turno empieza en "[mic] transcrito" y termina en el "Asistente:" (respuesta)
# que lo sigue. Dentro se marcan los instantes de cada etapa.
lineas = open(LOG, encoding="utf-8", errors="replace").read().splitlines()

turnos = []
cur = None
for ln in lineas:
    t = ts(ln)
    if t is None:
        continue
    if "[mic] transcrito:" in ln:
        # cerrar el anterior si quedo abierto sin respuesta
        m = re.search(r"transcrito: '(.*)'", ln)
        texto = (m.group(1) if m else "").strip()
        if not texto:
            cur = None          # activacion de mic sin habla: no es turno
            continue
        cur = {"t_mic": t, "texto": texto}
        turnos.append(cur)
    elif cur is None:
        continue
    elif "[afecto] el texto expresa" in ln:
        cur["t_ter"] = t
    elif "[afecto] la camara vio" in ln:
        cur["t_fer"] = t
        mm = re.search(r"(\d+) cuadros en ([\d.]+) s", ln)
        if mm:
            cur["fer_cuadros"] = int(mm.group(1))
            cur["fer_ventana_s"] = float(mm.group(2))
    elif "[afecto] la voz suena" in ln:
        cur["t_ser"] = t
    elif "[afecto] usuario (fusion" in ln:
        cur["t_pulse"] = t
    elif re.search(r"\[afecto\] v .*->.*", ln):
        cur["t_rose"] = t
    elif "[aeb]" in ln and "t_aeb" not in cur:
        cur["t_aeb"] = t
        mm = re.search(r"-> (\w+_L\d|\w+) \(cara (\w+), movimiento (\w+)", ln)
        if mm:
            cur["emocion"] = mm.group(1)
            cur["cara"] = mm.group(2)
            cur["movimiento"] = mm.group(3)
    elif "Asistente:" in ln and "Asistente [" not in ln:
        cur["t_resp"] = t
        cur = None          # turno cerrado

# --- 3. calcular tiempos por etapa (ms) y escribir ---
cols = ["turno", "texto",
        "stt_ms",                 # de percepcion.csv (Pi)
        "ser_ms", "ter_ms",       # de percepcion.csv (Pi)
        "capa_afectiva_ms",       # t_aeb - t_ter (PULSE+ROSE+AEB, por log)
        "fer_ventana_s",          # ventana de camara de ese turno
        "latencia_total_ms",      # t_resp - t_mic (mic -> IA responde)
        "emocion", "cara", "movimiento"]

filas = []
for i, tu in enumerate(turnos, 1):
    key = tu["texto"].strip().lower()[:30]
    p = percep.get(key)
    if p is None:                        # texto no empareja: usar el orden
        p = percep_orden[i - 1] if i - 1 < len(percep_orden) else {}
    def num(v):
        try: return float(v)
        except (TypeError, ValueError): return ""
    stt = num(p.get("t_stt_ms"))
    ser = num(p.get("t_ser_ms"))
    ter = num(p.get("t_ter_ms"))
    # capa afectiva: desde el primer analisis (TER) hasta el AEB
    capa = ""
    if "t_ter" in tu and "t_aeb" in tu:
        capa = round((tu["t_aeb"] - tu["t_ter"]) * 1000, 1)
    lat = ""
    if "t_resp" in tu:
        lat = round((tu["t_resp"] - tu["t_mic"]) * 1000, 1)
    filas.append([
        i, tu["texto"][:40],
        stt, ser, ter, capa,
        tu.get("fer_ventana_s", ""),
        lat,
        tu.get("emocion", ""), tu.get("cara", ""), tu.get("movimiento", ""),
    ])

with open(SALIDA, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(cols)
    w.writerows(filas)

# --- resumen ---
def prom(idx):
    vals = [r[idx] for r in filas if isinstance(r[idx], (int, float)) and r[idx] != ""]
    return sum(vals) / len(vals) if vals else 0

print(f"Turnos reconstruidos (con texto real): {len(filas)}")
print(f"  STT          promedio: {prom(2):.0f} ms")
print(f"  SER          promedio: {prom(3):.0f} ms")
print(f"  TER          promedio: {prom(4):.0f} ms")
print(f"  Capa afectiva promedio: {prom(5):.0f} ms")
print(f"  Ventana FER  promedio: {prom(6):.1f} s")
print(f"  Latencia total promedio: {prom(7):.0f} ms")
print(f"Guardado en: {SALIDA}")
