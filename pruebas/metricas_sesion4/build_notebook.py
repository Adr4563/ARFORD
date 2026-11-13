# -*- coding: utf-8 -*-
"""Notebook de metricas de la TRIVIA de Romeo - SESION OPTIMISTA (sesion 4).

Novedad frente a sesiones anteriores: STT, SER y TER se miden POR SEPARADO
en la propia Raspberry (percepcion_*.csv), ya no en el PC.
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
cells = []
def md(t): cells.append(new_markdown_cell(t))
def code(t): cells.append(new_code_cell(t))

md("""# Métricas de la trivia — Robot Romeo (perfil OPTIMISTA)
**Sesión 11 nov 2026 · perfil optimista (empatía 0.4, base (0.40, 0.30, 0.20)) · captura con `ROMEO_MEDIR=1`**

Solo métricas de la **trivia** (primeras 33 preguntas). Este notebook lee los CSV crudos y genera las gráficas.

**Novedad de esta sesión:** los tiempos de **STT, SER y TER** se miden **por separado en la propia Raspberry**
(archivo `percepcion.csv`), ya no en el PC. Cada turno guarda `t_stt_ms`, `t_ser_ms`, `t_ter_ms` e `ida_vuelta_ms`.

**Nodos del sistema:** `communication_node` (STT+TTS) · `expression_faces_node` (FER+caras) · `moving_control_node` (carrito) · `orchestrator_node` (afectiva PULSE/ROSE/AEB + LLM + trivia).""")

code("""import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "figure.dpi": 110, "font.size": 11, "axes.spines.top": False,
    "axes.spines.right": False, "font.family": "DejaVu Sans",
})
GRIS="#2b3a52"; VERDE="#1ba378"; NARANJA="#e07a1a"; AZUL="#2a78d6"; ROJO="#c0392b"; MORADO="#6a5aa0"

BASE = Path(r".")   # misma carpeta del notebook
def ruta(n): return BASE / n
N_PREGUNTAS = 33   # la sesion de trivia: primeras 33 preguntas""")

md("""## 0. Reconstrucción del flujo completo desde el log (robot OPTIMISTA)

Se **reconstruye cada pregunta de la trivia leyendo el `romeo_run.log`**,
siguiendo el flujo real desde que el micro captura hasta que la IA responde, y
midiendo cada etapa con los **timestamps del propio log**. Solo se miden las
**33 preguntas de la trivia** (se descarta la charla previa). Perfil
**OPTIMISTA** (empatía 0.4, base (0.40, 0.30, 0.20)).

```
[mic] transcrito → STT → TER + FER + SER → fusión (PULSE) → ROSE → AEB → Asistente responde
```

El tiempo del **STT** no está en el log (solo "[mic] transcrito"), pero sí en
`percepcion.csv` (medido en la Pi); se empareja por el texto de la frase, y por
orden si el texto difiere. Así cada métrica queda alineada a su turno real.""")

code('''import re, csv as _csv
def reconstruir(log_txt, percep_path):
    """Devuelve una lista de turnos con sus tiempos por etapa, desde el log."""
    TS = re.compile(r"\\[(\\d+\\.\\d+)\\]")
    def ts(l):
        m = TS.search(l); return float(m.group(1)) if m else None
    # percepcion.csv: STT/SER/TER por texto + en orden
    percep, percep_orden = {}, []
    with open(percep_path, encoding="utf-8", errors="replace") as f:
        for row in _csv.DictReader(f):
            txt = (row.get("texto") or "").strip().lower()[:30]
            if txt:
                percep[txt] = row; percep_orden.append(row)
    # Inicio de la TRIVIA: solo interesan las preguntas de la trivia, no la
    # charla previa (saludo, "quiero jugar"...). Se mide desde la 1a pregunta.
    mi = re.search(r"\\[(\\d+\\.\\d+)\\].*Vamos con una trivia", log_txt)
    if not mi:
        mi = re.search(r"\\[(\\d+\\.\\d+)\\].*Asistente \\[[^\\]]+\\]: [^\\n]*\\?", log_txt)
    inicio_trivia = float(mi.group(1)) if mi else 0.0
    turnos = []; cur = None
    for ln in log_txt.splitlines():
        t = ts(ln)
        if t is None: continue
        if "[mic] transcrito:" in ln:
            m = re.search(r"transcrito: \\'(.*)\\'", ln)
            texto = (m.group(1) if m else "").strip()
            if not texto: cur = None; continue
            if t < inicio_trivia: cur = None; continue   # descarta charla previa
            cur = {"t_mic": t, "texto": texto}; turnos.append(cur)
        elif cur is None: continue
        elif "[afecto] el texto expresa" in ln: cur["t_ter"] = t
        elif "[afecto] la camara vio" in ln:
            cur["t_fer"] = t
            mm = re.search(r"(\\d+) cuadros en ([\\d.]+) s", ln)
            if mm: cur["fer_ventana_s"] = float(mm.group(2))
        elif "[afecto] la voz suena" in ln: cur["t_ser"] = t
        elif "[afecto] usuario (fusion" in ln: cur["t_pulse"] = t
        elif re.search(r"\\[afecto\\] v .*->.*", ln): cur["t_rose"] = t
        elif "[aeb]" in ln and "t_aeb" not in cur:
            cur["t_aeb"] = t
            mm = re.search(r"-> (\\w+_L\\d|\\w+) \\(cara (\\w+), movimiento (\\w+)", ln)
            if mm:
                cur["emocion"], cur["cara"], cur["movimiento"] = mm.group(1), mm.group(2), mm.group(3)
        elif "Asistente:" in ln and "Asistente [" not in ln:
            cur["t_resp"] = t; cur = None
    # tiempos por etapa
    def num(v):
        try: return float(v)
        except (TypeError, ValueError): return np.nan
    out = []
    for i, tu in enumerate(turnos, 1):
        p = percep.get(tu["texto"].strip().lower()[:30])
        if p is None: p = percep_orden[i-1] if i-1 < len(percep_orden) else {}
        capa = (tu["t_aeb"]-tu["t_ter"])*1000 if "t_ter" in tu and "t_aeb" in tu else np.nan
        lat  = (tu["t_resp"]-tu["t_mic"])*1000 if "t_resp" in tu else np.nan
        out.append({"turno": i, "texto": tu["texto"][:40],
                    "stt_ms": num(p.get("t_stt_ms")), "ser_ms": num(p.get("t_ser_ms")),
                    "ter_ms": num(p.get("t_ter_ms")),
                    "capa_afectiva_ms": round(capa,1) if capa==capa else np.nan,
                    "fer_ventana_s": tu.get("fer_ventana_s", np.nan),
                    "latencia_ms": round(lat,1) if lat==lat else np.nan,
                    "emocion": tu.get("emocion",""), "cara": tu.get("cara",""),
                    "movimiento": tu.get("movimiento","")})
    return pd.DataFrame(out)

log = open(ruta("romeo_run.log"), encoding="utf-8", errors="replace").read()
rec = reconstruir(log, ruta("percepcion.csv"))
rec = rec.head(N_PREGUNTAS).reset_index(drop=True)   # solo las 33 de la trivia
rec["turno"] = range(1, len(rec)+1)
rec.to_csv(ruta("turnos_reconstruidos.csv"), index=False)
print(f"Preguntas de trivia reconstruidas desde el log: {len(rec)}")
print(rec[["turno","stt_ms","ser_ms","ter_ms","capa_afectiva_ms","fer_ventana_s","latencia_ms","emocion"]].head(10).to_string(index=False))''')

code('''# Resumen del flujo completo reconstruido (promedios por etapa).
resumen = pd.DataFrame([
    ("STT (voz→texto)",        rec["stt_ms"].mean(),           "ms"),
    ("SER (emoción voz)",      rec["ser_ms"].mean(),           "ms"),
    ("TER (emoción texto)",    rec["ter_ms"].mean(),           "ms"),
    ("Capa afectiva (PULSE+ROSE+AEB)", rec["capa_afectiva_ms"].mean(), "ms"),
    ("Ventana FER (cámara)",   rec["fer_ventana_s"].mean(),    "s"),
    ("Latencia total (mic→IA)",rec["latencia_ms"].mean(),      "ms"),
], columns=["Etapa del flujo", "Promedio", "Unidad"])
resumen["Promedio"] = resumen["Promedio"].round(1)
print("Flujo de una pregunta de trivia (robot OPTIMISTA), medido desde el log:")
resumen''')

md("## 1. Cargar los datos")

code('''# La trivia empieza en la PRIMERA pregunta que hace Romeo (Asistente [emo]: ...?).
import re
INICIO_TRIVIA = 0.0
log = open(ruta("romeo_run.log"), encoding="utf-8", errors="replace").read()
m = re.search(r"\\[(\\d+\\.\\d+)\\].*Vamos con una trivia", log)
if not m:
    m = re.search(r"\\[(\\d+\\.\\d+)\\].*Asistente \\[[^\\]]+\\]: [^\\n]*\\?", log)
if m: INICIO_TRIVIA = float(m.group(1))
print("La trivia empieza en t =", INICIO_TRIVIA)''')

code('''# turnos (Pi): afectiva + LLM + tokens + emocion + movimiento, SOLO desde la trivia.
# Algunas filas vienen con una columna extra, por eso se parsea por el campo t_fin (>1e9).
filas = []
with open(ruta("turnos.csv"), encoding="utf-8", errors="replace") as f:
    next(f)
    for ln in f:
        partes = ln.rstrip("\\n").split(",")
        nums = []
        for x in partes:
            try: nums.append(float(x))
            except ValueError: pass
        ei = next((i for i,v in enumerate(nums) if v > 1e9), None)  # t_fin
        if ei is None or ei < 3: continue
        t_fin = nums[ei]
        if t_fin < INICIO_TRIVIA: continue
        afect = nums[0]*1000          # t_pulse_rose (s -> ms)
        t_llm = nums[ei-1]            # t_llm (ms)
        tok   = nums[ei-3]            # llm_tokens_salida
        # emocion y movimiento: las dos columnas de texto antes del ultimo texto
        emocion = partes[-3].strip() if len(partes) >= 3 else ""
        movimiento = partes[-2].strip() if len(partes) >= 2 else ""
        if 50 <= t_llm <= 6000 and 1 <= tok <= 300:
            # La capa afectiva SIEMPRE se calcula (se quito la regla de saltarla
            # con todo neutral). Se muestran los 33 turnos sin recorte; el tiempo
            # incluye la espera de la ventana de camara antes de fusionar, por eso
            # algunos turnos son altos.
            filas.append({"afectiva_ms": afect,
                          "t_llm_ms": t_llm, "tokens_salida": tok,
                          "emocion": emocion, "movimiento": movimiento})
turnos = pd.DataFrame(filas[:N_PREGUNTAS])

# percepcion (Pi): STT / SER / TER por separado + ida_vuelta.
# Solo cuentan los turnos donde el usuario REALMENTE habla: se descartan las
# filas sin texto (micro activado por ruido/silencio, sin SER/TER). Asi cada
# punto es una respuesta real del usuario y no quedan huecos.
per = pd.read_csv(ruta("percepcion.csv"))
per = per[per["texto"].notna() & (per["texto"].astype(str).str.strip() != "")]
per_triv = per[per["t"] >= INICIO_TRIVIA].head(N_PREGUNTAS).reset_index(drop=True)

tts = pd.read_csv(ruta("tts.csv"))
sis = pd.read_csv(ruta("sistema.csv"))
sis["min"] = (sis["t"] - sis["t"].iloc[0]) / 60
print("turnos:", len(turnos), "| percepcion trivia:", len(per_triv),
      "| tts:", len(tts), "| sistema:", len(sis))''')

md("""## 2. Comparación Rocha vs IA  —  ¿coincide la respuesta?

Para cada pregunta de la trivia se comparó la **respuesta del usuario** con la
**respuesta esperada** (banco de Rocha); el veredicto lo dio el corrector LLM
de Romeo (registrado en el log). Aquí se cuenta por categoría temática.
**Recalculado desde el log de esta sesión: 19/33 = 58 %.**""")

code("""# Aciertos/total por categoria, recalculados desde los veredictos del log (sesion optimista).
cats = [
    ("Geografía", 7, 11),
    ("Arte, música y cultura", 4, 6),
    ("Ciencia y naturaleza", 3, 5),
    ("Razonamiento matemático", 3, 4),
    ("Cultura general", 2, 4),
    ("Historia", 0, 3),
]
dfc = pd.DataFrame(cats, columns=["categoria","coincide","total"])
dfc["no_coincide"] = dfc["total"] - dfc["coincide"]
dfc = dfc.iloc[::-1].reset_index(drop=True)  # mayor arriba

fig, ax = plt.subplots(figsize=(8,5.5))
y = np.arange(len(dfc))
ax.barh(y, dfc["coincide"], color=VERDE, zorder=3, label="Coincide")
ax.barh(y, dfc["no_coincide"], left=dfc["coincide"], color=NARANJA, zorder=3, label="No coincide")
for i,(c,t) in enumerate(zip(dfc["coincide"], dfc["total"])):
    ax.text(t+0.1, i, f"{c}/{t}", va="center", fontsize=10, color="#333")
ax.set_yticks(y); ax.set_yticklabels(dfc["categoria"], fontsize=10)
co, to = dfc["coincide"].sum(), dfc["total"].sum()
ax.set_xlabel(f"Preguntas (N = {to})")
ax.legend(loc="upper center", bbox_to_anchor=(0.5,1.10), ncol=2, frameon=False, fontsize=11)
ax.set_xlim(0, dfc["total"].max()+1.2)
plt.tight_layout(); plt.show()
print(f"Coincidencia Rocha vs IA: {co}/{to} = {100*co/to:.0f}%")""")

md("""## 3. Datos por turno

Se arma la tabla con el tiempo de cada etapa en cada pregunta. **STT, SER y TER
se toman directamente de `percepcion.csv`** (medidos en la Pi, por turno, por
separado). El resto (afectiva, LLM, tokens, emoción, movimiento) viene de
`turnos.csv`; TTS de `tts.csv`; la ventana FER se lee del log.""")

code('''import re
# Ventana FER por turno, EMPAREJADA POR TIEMPO. El log tiene mas lineas "FER:"
# que preguntas (la camara cierra varias ventanas por turno), asi que NO sirve
# el orden ciego. El patron real de cada turno es:
#   [mic] transcrito  ->  Asistente: (respuesta)  ->  FER: X cuadros en Y s
# Se toma la ventana FER que aparece justo DESPUES de cada transcripcion del
# usuario (la que corresponde a mientras el usuario hablaba ese turno).
eventos = []
for ln in log.splitlines():
    m = re.search(r"\\[(\\d+\\.\\d+)\\]", ln)
    if not m: continue
    t = float(m.group(1))
    if "[mic] transcrito:" in ln:
        mm = re.search(r"transcrito: '(.*)'", ln)
        txt = (mm.group(1) if mm else "").strip()
        if txt:                       # solo turnos reales (con texto)
            eventos.append((t, "MIC"))
    elif "FER:" in ln:
        mm = re.search(r"FER: \\d+ cuadros en ([\\d.]+) s", ln)
        if mm: eventos.append((t, "FER", float(mm.group(1))*1000))
# para cada MIC, la primera ventana FER que viene despues
fer_vent = []
for i, ev in enumerate(eventos):
    if ev[1] != "MIC": continue
    sig = next((e for e in eventos[i+1:] if e[1] == "FER"), None)
    if sig: fer_vent.append(sig[2])
# TTS por turno (en orden; tts.csv no trae texto)
tts_vals = pd.to_numeric(tts["t_tts_ms"], errors="coerce").dropna().tolist()

n = min(N_PREGUNTAS, len(turnos), len(per_triv))
dft = pd.DataFrame({
    "pregunta": range(1, n+1),
    "STT": per_triv["t_stt_ms"].iloc[:n].values,
    "SER": per_triv["t_ser_ms"].iloc[:n].values,
    "TER": per_triv["t_ter_ms"].iloc[:n].values,
    "ida_vuelta": per_triv["ida_vuelta_ms"].iloc[:n].values,
    "afectiva": turnos["afectiva_ms"].iloc[:n].values,
    "LLM": turnos["t_llm_ms"].iloc[:n].values,
    "emocion": turnos["emocion"].iloc[:n].values,
    "movimiento": turnos["movimiento"].iloc[:n].values,
})
dft["TTS"] = [tts_vals[i] if i < len(tts_vals) else np.nan for i in range(n)]
dft["FER"] = [fer_vent[i] if i < len(fer_vent) else np.nan for i in range(n)]
# por nodo
dft["communication_node"] = dft[["STT","TTS"]].sum(axis=1, min_count=1)
dft["orchestrator_node"]  = dft[["afectiva","LLM"]].sum(axis=1, min_count=1)
dft["expression_faces_node"] = dft["FER"]
print(f"{len(dft)} turnos."); dft.head(10)''')

md("""## 4. STT, SER y TER por turno — medidos en la Raspberry

Los tres tiempos de percepción del lenguaje/voz, capturados **por separado en la
propia Pi** (lo que pediste): **STT** (voz→texto), **SER** (emoción de la voz) y
**TER** (emoción del texto).

Como los tres tienen **escalas muy distintas** (el TER es ~15-25 ms, el SER
~200 ms y el STT ~80 ms), se grafican en **paneles separados, cada uno con su
propia escala**, para poder ver el procesamiento real de cada uno por pregunta.
Abajo va además la vista combinada con el viaje de ida y vuelta.

Solo se cuentan los turnos donde **el usuario realmente habla** (con texto
transcrito): las activaciones del micro sin habla se descartan, por eso no hay
huecos.""")

code("""# Tres paneles, cada uno con su escala, para ver el detalle de cada etapa.
fig, axs = plt.subplots(3, 1, figsize=(12,8), sharex=True)
x = dft["pregunta"]
for ax, col, color, mk, nombre in [
    (axs[0], "STT", AZUL,    "o", "STT (voz→texto)"),
    (axs[1], "SER", NARANJA, "s", "SER (emoción voz)"),
    (axs[2], "TER", VERDE,   "^", "TER (emoción texto)"),
]:
    serie = dft[col]
    ax.plot(x, serie, color=color, lw=1.8, marker=mk, ms=4, label=nombre)
    prom = serie.mean()
    ax.axhline(prom, color=GRIS, ls="--", lw=1, alpha=0.7)
    ax.text(len(dft), prom, f" prom {prom:.0f} ms", va="center", fontsize=9, color=GRIS)
    ax.fill_between(x, 0, serie, color=color, alpha=0.08)
    ax.set_ylabel("Tiempo (ms)")
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.set_xlim(1, len(dft))
axs[2].set_xlabel("Pregunta (turno)")
axs[0].set_title("STT, SER y TER por turno — cada uno con su escala (Raspberry)")
plt.tight_layout(); plt.show()
print(f"STT: {dft['STT'].mean():.0f} ms (min {dft['STT'].min():.0f}, max {dft['STT'].max():.0f}) | "
      f"SER: {dft['SER'].mean():.0f} ms (min {dft['SER'].min():.0f}, max {dft['SER'].max():.0f}) | "
      f"TER: {dft['TER'].mean():.0f} ms (min {dft['TER'].min():.0f}, max {dft['TER'].max():.0f})")""")

code("""# Vista combinada (misma escala) + viaje de ida y vuelta.
fig, ax = plt.subplots(figsize=(12,4.6))
x = dft["pregunta"]
ax.plot(x, dft["STT"], color=AZUL,    lw=1.8, marker="o", ms=3, label="STT (voz→texto)")
ax.plot(x, dft["SER"], color=NARANJA, lw=1.8, marker="s", ms=3, label="SER (emoción voz)")
ax.plot(x, dft["TER"], color=VERDE,   lw=1.8, marker="^", ms=3, label="TER (emoción texto)")
ax.plot(x, dft["ida_vuelta"], color=GRIS, lw=1.2, ls="--", label="Ida y vuelta (total)")
ax.set_xlabel("Pregunta (turno)"); ax.set_ylabel("Tiempo (ms)")
ax.set_xlim(1, len(dft))
ax.set_title("STT, SER y TER por turno (vista combinada)")
ax.legend(frameon=False, fontsize=9, ncol=2)
plt.tight_layout(); plt.show()
print(f"STT: {dft['STT'].mean():.0f} ms | SER: {dft['SER'].mean():.0f} ms | "
      f"TER: {dft['TER'].mean():.0f} ms | Ida-vuelta: {dft['ida_vuelta'].mean():.0f} ms (promedios)")""")

md("""## 5. Línea de tiempo por nodo

El tiempo de `expression_faces_node` es la **ventana de la cámara**: crece cuando
el usuario tarda en responder. Unos pocos turnos con pausas largas disparan ese
tiempo a decenas de segundos y aplastarían el resto, así que el eje se **recorta
a 60 s** y esos picos se **anotan con su valor real** (son pausas del usuario, no
lentitud del sistema).""")

code("""fig, ax = plt.subplots(figsize=(12,4.5))
x = dft["pregunta"]
ax.plot(x, dft["communication_node"]/1000, color=AZUL, lw=1.8, label="communication_node (STT+TTS)")
ax.plot(x, dft["orchestrator_node"]/1000, color=VERDE, lw=1.8, label="orchestrator_node (afectiva+LLM)")
ax.plot(x, dft["expression_faces_node"]/1000, color=NARANJA, lw=1.8, label="expression_faces_node (FER)")
TOPE = 60  # segundos
ax.set_ylim(0, TOPE)
# anotar los picos que se salen del tope (pausas largas del usuario)
fer_s = dft["expression_faces_node"]/1000
for xi, v in zip(x, fer_s):
    if v > TOPE:
        ax.annotate(f"{v:.0f} s", xy=(xi, TOPE), xytext=(xi, TOPE*0.9),
                    ha="center", fontsize=8, color=NARANJA,
                    arrowprops=dict(arrowstyle="->", color=NARANJA, lw=1))
ax.set_xlabel("Pregunta (turno)"); ax.set_ylabel("Tiempo del nodo (s)")
ax.set_xlim(1, len(dft))
ax.set_title("Tiempo usado por cada nodo a lo largo de la trivia (eje recortado a 60 s)")
ax.legend(frameon=False, fontsize=9)
plt.tight_layout(); plt.show()
print("moving_control_node (carrito): no interviene en la trivia.")
picos = [(int(xi), float(v)) for xi, v in zip(x, fer_s) if v > TOPE]
print("Picos de ventana FER (pausas largas del usuario):",
      ", ".join(f"preg {p}: {s:.0f}s" for p, s in picos))""")

md("""## 6. Latencia de respuesta por interacción (medida en la Raspberry)

Tiempo desde que el usuario termina de hablar hasta que Romeo responde, por cada
pregunta. Se mide en la Pi con los instantes del log (fin de transcripción →
inicio de la respuesta del asistente).""")

code('''latencias = []
t_mic = None
for ln in log.splitlines():
    m = re.search(r"\\[(\\d+\\.\\d+)\\]", ln)
    if not m: continue
    ts = float(m.group(1))
    if "[mic] transcrito:" in ln:
        t_mic = ts
    elif "Asistente:" in ln and "Asistente [" not in ln and t_mic is not None and ts >= INICIO_TRIVIA:
        lat = (ts - t_mic) * 1000
        if 0 < lat < 30000:
            latencias.append(lat)
        t_mic = None
latencias = latencias[:N_PREGUNTAS]

fig, ax = plt.subplots(figsize=(12,4.5))
xx = range(1, len(latencias) + 1)
ax.plot(xx, latencias, color=AZUL, lw=2, marker="o", ms=4, label="Latencia de respuesta (ms)")
ax.axhline(np.mean(latencias), color=NARANJA, ls="--", lw=1.4,
           label=f"promedio {np.mean(latencias):.0f} ms")
ax.fill_between(xx, 0, latencias, color=AZUL, alpha=0.10)
ax.set_xlabel("Pregunta (turno)"); ax.set_ylabel("Latencia (ms)")
ax.set_xlim(1, len(latencias))
ax.set_title("Latencia de respuesta por interacción (inicio a fin, en la Raspberry)")
ax.legend(frameon=False, fontsize=9)
plt.tight_layout(); plt.show()
print(f"Latencia: promedio {np.mean(latencias):.0f} ms, min {min(latencias):.0f}, max {max(latencias):.0f} ms")''')

md("""## 7. Los tres modelos de percepción emocional: FER, SER y TER

Las tres fuentes que alimentan la capa afectiva, por turno:
- **FER** (emoción por la cara) — es la **ventana de la cámara**, en segundos;
  crece cuando el usuario tarda en responder.
- **SER** (emoción por la voz) y **TER** (emoción por el texto) — en milisegundos.

Como FER está en segundos y SER/TER en milisegundos, van en **paneles separados,
cada uno con su escala**. En el FER el eje se recorta a 60 s y las pausas largas
del usuario se anotan con su valor real.""")

code("""fig, axs = plt.subplots(3, 1, figsize=(12,8), sharex=True)
x = dft["pregunta"]
fer_s = dft["FER"] / 1000.0

# (1) FER en segundos, recortado a 60s
TOPE = 60
axs[0].plot(x, fer_s, color=AZUL, lw=2, marker="o", ms=4, label="FER (emoción cara · ventana)")
axs[0].fill_between(x, 0, fer_s, color=AZUL, alpha=0.08)
axs[0].set_ylim(0, TOPE); axs[0].set_ylabel("FER (s)")
axs[0].axhline(fer_s[fer_s<=TOPE].mean(), color=GRIS, ls="--", lw=1, alpha=0.7)
for xi, v in zip(x, fer_s):
    if v > TOPE:
        axs[0].annotate(f"{v:.0f} s", xy=(xi, TOPE), xytext=(xi, TOPE*0.88),
                        ha="center", fontsize=8, color=AZUL,
                        arrowprops=dict(arrowstyle="->", color=AZUL, lw=1))
axs[0].legend(loc="upper right", frameon=False, fontsize=10)

# (2) SER en ms
axs[1].plot(x, dft["SER"], color=NARANJA, lw=2, marker="s", ms=4, label="SER (emoción voz)")
axs[1].fill_between(x, 0, dft["SER"], color=NARANJA, alpha=0.08)
axs[1].axhline(dft["SER"].mean(), color=GRIS, ls="--", lw=1, alpha=0.7)
axs[1].text(len(dft), dft["SER"].mean(), f" prom {dft['SER'].mean():.0f} ms", va="center", fontsize=9, color=GRIS)
axs[1].set_ylabel("SER (ms)")
axs[1].legend(loc="upper right", frameon=False, fontsize=10)

# (3) TER en ms
axs[2].plot(x, dft["TER"], color=VERDE, lw=2, marker="^", ms=4, label="TER (emoción texto)")
axs[2].fill_between(x, 0, dft["TER"], color=VERDE, alpha=0.08)
axs[2].axhline(dft["TER"].mean(), color=GRIS, ls="--", lw=1, alpha=0.7)
axs[2].text(len(dft), dft["TER"].mean(), f" prom {dft['TER'].mean():.0f} ms", va="center", fontsize=9, color=GRIS)
axs[2].set_ylabel("TER (ms)"); axs[2].set_xlabel("Pregunta (turno)")
axs[2].legend(loc="upper right", frameon=False, fontsize=10)
axs[2].set_xlim(1, len(dft))

axs[0].set_title("FER, SER y TER por turno — cada uno con su escala (Raspberry)")
plt.tight_layout(); plt.show()
print(f"FER: promedio {fer_s.mean():.1f} s (max {fer_s.max():.0f} s) | "
      f"SER: {dft['SER'].mean():.0f} ms | TER: {dft['TER'].mean():.0f} ms")
picos = [(int(xi), float(v)) for xi, v in zip(x, fer_s) if v > TOPE]
print("Picos FER (pausas largas del usuario):",
      ", ".join(f"preg {p}: {s:.0f}s" for p, s in picos))""")

md("""## 8. Línea de tiempo — LLM y TTS durante la trivia

Tiempo del modelo de lenguaje (LLM gemma3:4b) y de la síntesis de voz (TTS) por turno.""")

code("""fig, ax = plt.subplots(figsize=(12,4.5))
x = dft["pregunta"]
ax.plot(x, dft["LLM"], color=NARANJA, lw=1.8, marker="o", ms=3, label="LLM (gemma3:4b)")
ax.plot(x, dft["TTS"], color=AZUL, lw=1.8, marker="o", ms=3, label="TTS (voz)")
ax.set_xlabel("Pregunta (turno)"); ax.set_ylabel("Tiempo (ms)")
ax.set_xlim(1, len(dft))
ax.set_title("LLM y TTS a lo largo de la sesión de trivia")
ax.legend(frameon=False, fontsize=9)
plt.tight_layout(); plt.show()
print(f"LLM: promedio {dft['LLM'].mean():.0f} ms | TTS: promedio {dft['TTS'].mean():.0f} ms")""")

md("""## 9. Estado emocional de Romeo durante la trivia (AEB)

Emoción y nivel que Romeo mostró en cada turno (su estado afectivo real, no una
cara por defecto de acertó/falló). Con el perfil **optimista** su base lo mantiene
en felicidad aunque el usuario esté neutral.""")

code("""conteo = dft["emocion"].value_counts()
fig, ax = plt.subplots(figsize=(9,4.2))
colores = {"felicidad_L1":VERDE, "felicidad_L2":"#15c08a", "felicidad_L3":"#0e8f66",
           "neutral_L0":GRIS, "tristeza_L1":AZUL, "enojo_L1":ROJO}
cs = [colores.get(e, MORADO) for e in conteo.index]
ax.bar(range(len(conteo)), conteo.values, color=cs, zorder=3)
ax.set_xticks(range(len(conteo))); ax.set_xticklabels(conteo.index, rotation=20, ha="right", fontsize=10)
for i,v in enumerate(conteo.values):
    ax.text(i, v+0.2, str(v), ha="center", fontsize=10, color="#333")
ax.set_ylabel("Turnos"); ax.set_title("Estado emocional de Romeo (AEB) por la trivia")
plt.tight_layout(); plt.show()
print("Emociones mostradas:"); print(conteo.to_string())
mov = dft["movimiento"].value_counts()
print("\\nMovimientos:"); print(mov.to_string())""")

md("""## 10. Recursos de la Raspberry Pi (de inicio a fin)

CPU total, proceso de cámara (FER), **temperatura real** y RAM durante la sesión.
En esta corrida la temperatura **sí se capturó** (lectura de `thermal_zone0`).""")

code("""ram_gb = sis["ram_usada_mb"] / 1024

fig, axs = plt.subplots(3, 1, figsize=(10,8), sharex=True)
# (1) CPU total + camara
axs[0].plot(sis["min"], sis["cpu_total_pct"], color=GRIS, lw=1, label="CPU total (%)")
axs[0].plot(sis["min"], sis["cpu_camara_pct"], color=NARANJA, lw=1.6, label="Proceso de cámara (% de un núcleo)")
axs[0].set_ylabel("CPU (%)")
axs[0].legend(loc="upper center", bbox_to_anchor=(0.5,1.32), ncol=2, frameon=False, fontsize=10)
# (2) Temperatura real
axs[1].plot(sis["min"], sis["temp_c"], color=NARANJA, lw=1.3)
axs[1].axhline(80, color=GRIS, ls="--", lw=1)
axs[1].text(sis["min"].max(), 80.4, "80 °C", ha="right", fontsize=9, color=GRIS)
axs[1].set_ylabel("Temp. (°C)")
# (3) RAM
axs[2].plot(sis["min"], ram_gb, color=AZUL, lw=1.4)
axs[2].set_ylabel("RAM (GB)")
axs[2].set_xlabel("Minutos desde el inicio del muestreo")
plt.tight_layout(); plt.show()

print("Recursos:")
print(f"  CPU total:  mediana {sis['cpu_total_pct'].median():.1f}%  max {sis['cpu_total_pct'].max():.1f}%")
print(f"  CPU camara: mediana {sis['cpu_camara_pct'].median():.1f}%  max {sis['cpu_camara_pct'].max():.1f}%")
print(f"  Temp:       mediana {sis['temp_c'].median():.1f}°C  max {sis['temp_c'].max():.1f}°C")
print(f"  RAM usada:  mediana {sis['ram_usada_mb'].median():.0f} MB  max {sis['ram_usada_mb'].max():.0f} MB")""")

md("## 11. Tabla resumen (trivia)")

code("""def prom1(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return round(s.mean(),1) if len(s) else np.nan

tabla = []
for nombre, serie, unidad, nodo in [
    ("STT (voz→texto)", dft["STT"], "ms", "communication / Pi"),
    ("SER (emoción voz)", dft["SER"], "ms", "communication / Pi"),
    ("TER (emoción texto)", dft["TER"], "ms", "communication / Pi"),
    ("Ida-vuelta percepción", dft["ida_vuelta"], "ms", "Pi ↔ servidor"),
    ("Afectiva PULSE+ROSE+AEB", dft["afectiva"], "ms", "orchestrator / Pi"),
    ("Respuesta LLM", dft["t_llm_ms"] if "t_llm_ms" in dft else dft["LLM"], "ms", "orchestrator / Pi→PC"),
    ("Síntesis TTS", dft["TTS"], "ms", "communication / PC"),
    ("Ventana FER", dft["FER"], "ms", "expression / Pi"),
    ("Tokens de salida", turnos["tokens_salida"], "tokens", "orchestrator"),
    ("CPU total Pi", sis["cpu_total_pct"], "%", "Raspberry"),
    ("CPU cámara/FER", sis["cpu_camara_pct"], "%", "expression"),
    ("Temperatura Pi", sis["temp_c"], "°C", "Raspberry"),
    ("RAM usada Pi", sis["ram_usada_mb"], "MB", "Raspberry"),
]:
    tabla.append([nombre, prom1(serie), unidad, nodo])
df = pd.DataFrame(tabla, columns=["Indicador","Promedio","Unidad","Nodo / máquina"])
print("Coincidencia Rocha vs IA: 19/33 = 58%\\n")
df""")

nb["cells"] = cells
import os as _os
out = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                    "metricas_trivia_romeo.ipynb")
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("notebook creado:", out, "| celdas:", len(cells))
