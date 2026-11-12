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
            filas.append({"afectiva_ms": afect if afect < 8000 else np.nan,
                          "t_llm_ms": t_llm, "tokens_salida": tok,
                          "emocion": emocion, "movimiento": movimiento})
turnos = pd.DataFrame(filas[:N_PREGUNTAS])

# percepcion (Pi): STT / SER / TER por separado + ida_vuelta
per = pd.read_csv(ruta("percepcion.csv"))
# nos quedamos con las de la trivia (desde INICIO_TRIVIA) y las primeras 33
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
# ventana FER por turno (del log, en orden de aparicion)
fer_vent = [float(m.group(1))*1000 for m in re.finditer(r"FER: \\d+ cuadros en ([\\d.]+) s", log)]
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
**TER** (emoción del texto), más el **viaje ida-vuelta** total al servidor.""")

code("""fig, ax = plt.subplots(figsize=(12,4.6))
x = dft["pregunta"]
ax.plot(x, dft["STT"], color=AZUL,    lw=1.8, marker="o", ms=3, label="STT (voz→texto)")
ax.plot(x, dft["SER"], color=NARANJA, lw=1.8, marker="s", ms=3, label="SER (emoción voz)")
ax.plot(x, dft["TER"], color=VERDE,   lw=1.8, marker="^", ms=3, label="TER (emoción texto)")
ax.plot(x, dft["ida_vuelta"], color=GRIS, lw=1.2, ls="--", label="Ida y vuelta (total)")
ax.set_xlabel("Pregunta (turno)"); ax.set_ylabel("Tiempo (ms)")
ax.set_xlim(1, len(dft))
ax.set_title("STT, SER y TER por turno (medidos en la Raspberry)")
ax.legend(frameon=False, fontsize=9, ncol=2)
plt.tight_layout(); plt.show()
print(f"STT: {dft['STT'].mean():.0f} ms | SER: {dft['SER'].mean():.0f} ms | "
      f"TER: {dft['TER'].mean():.0f} ms | Ida-vuelta: {dft['ida_vuelta'].mean():.0f} ms (promedios)")""")

md("## 5. Línea de tiempo por nodo")

code("""fig, ax = plt.subplots(figsize=(12,4.5))
x = dft["pregunta"]
ax.plot(x, dft["communication_node"], color=AZUL, lw=1.8, label="communication_node (STT+TTS)")
ax.plot(x, dft["orchestrator_node"], color=VERDE, lw=1.8, label="orchestrator_node (afectiva+LLM)")
ax.plot(x, dft["expression_faces_node"], color=NARANJA, lw=1.8, label="expression_faces_node (FER)")
ax.set_xlabel("Pregunta (turno)"); ax.set_ylabel("Tiempo del nodo (ms)")
ax.set_title("Tiempo usado por cada nodo a lo largo de la trivia")
ax.legend(frameon=False, fontsize=9)
plt.tight_layout(); plt.show()
print("moving_control_node (carrito): no interviene en la trivia.")""")

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

md("""## 7. FER y capa afectiva por turno (medidos en la Raspberry)

La **ventana del FER** (duración de la percepción facial mientras el usuario
habla) y el bloque **percepción + capa afectiva** (PULSE+ROSE+AEB, que incluye
la espera de la ventana de cámara antes de fusionar).""")

code("""fig, ax = plt.subplots(figsize=(12,4.6))
x = dft["pregunta"]
fer_s = dft["FER"] / 1000.0
afe_s = dft["afectiva"] / 1000.0
ax.plot(x, fer_s, color=AZUL, lw=2, marker="o", ms=3, label="FER (ventana de percepción)")
ax.plot(x, afe_s, color=MORADO, lw=2, marker="s", ms=3, label="Percepción + capa afectiva")
ax.fill_between(x, 0, fer_s, color=AZUL, alpha=0.08)
ax.set_xlabel("Pregunta (turno)"); ax.set_ylabel("Tiempo (s)")
ax.set_xlim(1, len(dft))
ax.set_title("FER y capa afectiva por turno (medidos en la Raspberry)")
ax.legend(frameon=False, fontsize=9)
plt.tight_layout(); plt.show()
print(f"FER ventana: promedio {fer_s.mean():.1f} s | Percepción+afectiva: promedio {afe_s.mean():.1f} s")""")

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
