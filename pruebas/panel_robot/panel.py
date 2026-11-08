r"""panel.py -- resultados del Romeo REAL, en vivo, leyendo su log.

La pagina de pruebas (pruebas/web) simula a Romeo en la PC. Esta, en
cambio, mira al robot: corre en la Pi, lee el log que deja
~/lanzar_romeo.sh (o ~/lanzar_tunel.sh) y muestra frase a frase lo que
entendio, lo que respondio, lo que leyo de la voz, el texto y la cara, tu
emocion fusionada (S_u) y el animo de Romeo, en dos cubos 3D.

No toca ROS ni el robot: solo lee el archivo. Por eso no cuesta nada y se
puede abrir y cerrar cuando se quiera.

Uso, en la Pi:
    python3 panel.py [ruta-del-log]       (por defecto ~/romeo_run.log)
y abrir http://<ip-de-la-pi>:8090/
"""
import json
import os
import re
import sys
import urllib.request

from flask import Flask, jsonify

LOG = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/romeo_run.log")
PUERTO = int(os.environ.get("ROMEO_PANEL_PUERTO", "8090"))

_MSG = re.compile(r"\]: (.*)$")
_NUM = r"([+-]?\d+\.\d+)"
_TRANSCRITO = re.compile(r"\[mic\] transcrito: '(.*)'$")
_ASISTENTE = re.compile(r"^Asistente: (.*)$")
_BASE = re.compile(r"\[afecto\] perfil (\w+): empatia ([\d.]+), base \(" + _NUM + r", " + _NUM + r", " + _NUM + r"\)")
_VOZ = re.compile(r"\[afecto\] la voz suena a (\w+) \(confianza ([\d.]+)\)")
_TEXTO = re.compile(r"\[afecto\] el texto expresa (\w+) \(confianza ([\d.]+)")
_CARA = re.compile(r"\[afecto\] la camara vio (\w+) \(confianza ([\d.]+)")
_SIN_CARA = re.compile(r"\[afecto\] la camara no vio cara")
_FUSION = re.compile(r"\[afecto\] usuario \(fusion de ([^)]*)\): v=" + _NUM + r" a=" + _NUM + r" d=" + _NUM)
_ROMEO = re.compile(r"\[afecto\] v " + _NUM + r"->" + _NUM + r"\s+a " + _NUM + r"->" + _NUM + r"\s+d " + _NUM + r"->" + _NUM)
_AEB = re.compile(r"\[aeb\] v=" + _NUM + r" a=" + _NUM + r" d=" + _NUM + r" -> (\S+) \(cara (\w+)")


def _leer():
    """Recorre el log y arma un turno por cada frase transcrita."""
    estado = {"perfil": None, "empatia": None, "base": None, "romeo": None,
              "estado_romeo": None, "turnos": [], "log": LOG}
    try:
        with open(LOG, encoding="utf-8", errors="replace") as f:
            lineas = f.readlines()
    except OSError:
        estado["error"] = f"No encuentro {LOG}. ¿Está corriendo Romeo?"
        return estado
    turno = None
    for linea in lineas:
        m = _MSG.search(linea)
        if not m:
            continue
        msg = m.group(1).strip()
        if (b := _BASE.search(msg)):
            estado["perfil"], estado["empatia"] = b.group(1), float(b.group(2))
            estado["base"] = dict(zip("vad", map(float, b.groups()[2:])))
        elif (t := _TRANSCRITO.search(msg)):
            turno = {"texto": t.group(1), "lecturas": [], "fusion": None,
                     "romeo": estado["romeo"], "respuesta": None, "sin_cara": False}
            estado["turnos"].append(turno)
        elif (a := _ASISTENTE.search(msg)):
            if turno is not None and turno["respuesta"] is None:
                turno["respuesta"] = a.group(1)
            estado["ultima_respuesta"] = a.group(1)
        elif (r := _ROMEO.search(msg)):
            v = list(map(float, r.groups()))
            estado["romeo"] = {"v": v[1], "a": v[3], "d": v[5]}
            if turno is not None:
                turno["romeo"] = estado["romeo"]
        elif (e := _AEB.search(msg)):
            estado["romeo"] = dict(zip("vad", map(float, e.groups()[:3])))
            estado["estado_romeo"], estado["cara"] = e.group(4), e.group(5)
            if turno is not None:
                turno["romeo"] = estado["romeo"]
        elif turno is None:
            continue
        elif (x := _VOZ.search(msg)):
            turno["lecturas"].append({"fuente": "Voz (SER)", "emocion": x.group(1), "confianza": float(x.group(2))})
        elif (x := _TEXTO.search(msg)):
            turno["lecturas"].append({"fuente": "Texto (TER)", "emocion": x.group(1), "confianza": float(x.group(2))})
        elif (x := _CARA.search(msg)):
            turno["lecturas"].append({"fuente": "Cara (FER)", "emocion": x.group(1), "confianza": float(x.group(2))})
        elif _SIN_CARA.search(msg):
            turno["sin_cara"] = True
        elif (fu := _FUSION.search(msg)):
            turno["fusion"] = dict(zip("vad", map(float, fu.groups()[1:])))
    # Un turno sin animo de Romeo todavia (la primera frase) toma la base.
    for t in estado["turnos"]:
        if t["romeo"] is None:
            t["romeo"] = estado["base"]
    return estado


app = Flask(__name__)


@app.route("/")
def pagina():
    return PAGINA


@app.route("/estado")
def estado():
    return jsonify(_leer())


# Traza del calculo afectivo que escribe el orquestador, una linea JSON por
# turno (romeo_brain/traza_afectiva.py).
TRAZA = os.path.expanduser(os.environ.get("ROMEO_TRAZA_AFECTO", "~/romeo_afecto.jsonl"))


@app.route("/traza")
def traza():
    turnos = []
    try:
        with open(TRAZA, encoding="utf-8") as f:
            for linea in f.readlines()[-30:]:
                try:
                    turnos.append(json.loads(linea))
                except ValueError:
                    pass
    except OSError:
        return jsonify(turnos=[], error=f"No encuentro {TRAZA}. ¿Está corriendo Romeo con la traza?")
    return jsonify(turnos=turnos)


@app.route("/calculo")
def calculo():
    return CALCULO


# Botones del carrito (pagina /carrito): se pasan tal cual a la pagina de
# control que sirve moving_control_node (romeo_drivers/_control_web.py),
# asi no se duplica la logica del carrito.
CARRITO_WEB = os.environ.get("ROMEO_CARRITO_WEB", "http://127.0.0.1:8080")
MOVIMIENTOS = {"F", "B", "SL", "SR", "RL", "RR", "FL", "FR", "BL", "BR", "S"}


@app.route("/carrito")
def carrito():
    return PAGINA_CARRITO


@app.route("/cmd/<comando>", methods=["POST"])
def cmd(comando):
    if comando not in MOVIMIENTOS:
        return jsonify(ok=False, error="comando desconocido"), 400
    try:
        req = urllib.request.Request(f"{CARRITO_WEB}/cmd/{comando}", method="POST")
        with urllib.request.urlopen(req, timeout=1.5) as r:
            return r.read(), r.status, {"Content-Type": "application/json"}
    except Exception as e:  # noqa: BLE001 - nodo del carrito apagado
        return jsonify(ok=False, error=f"el carrito no responde ({e})"), 503


PAGINA = r"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Romeo en vivo</title>
<style>
  :root { --fondo:#f9f9f7; --tarjeta:#fcfcfb; --texto:#0b0b0b; --suave:#52514e; --tenue:#898781;
    --borde:#e1e0d9; --eje:#c3c2b7; --fusion:#eda100; --romeo:#e87ba4;
    --val:#2a78d6; --aro:#eb6834; --dom:#1baf7a; --ok:#1baf7a;
    --enojo:#d1352b; --triste:#2a78d6; --feliz:#d4a400; --normal:#0b0b0b; --antes:#8b5cf6; --ahora:#16a34a; }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
    --fondo:#0d0d0d; --tarjeta:#1a1a19; --texto:#f0efec; --suave:#c3c2b7; --tenue:#898781;
    --borde:#2c2c2a; --eje:#383835; --fusion:#c98500; --romeo:#d55181;
    --val:#3987e5; --aro:#d95926; --dom:#199e70; --ok:#199e70;
    --enojo:#ef5a4f; --triste:#4d94ec; --feliz:#f2c94c; --normal:#f0efec; --antes:#a78bfa; --ahora:#22c55e; color-scheme: dark; } }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--fondo); color:var(--texto); font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
  main { max-width:960px; margin:0 auto; padding:28px 16px 48px; }
  h1 { font-size:1.5rem; margin:0 0 4px; }
  .sub { color:var(--suave); margin:0 0 16px; overflow-wrap:anywhere; }
  .vivo { display:inline-flex; align-items:center; gap:6px; font-size:.85rem; color:var(--suave); }
  .vivo i { width:8px; height:8px; border-radius:50%; background:var(--ok); }
  .tarjeta { background:var(--tarjeta); border:1px solid var(--borde); border-radius:12px; padding:20px; margin-bottom:16px; min-width:0; }
  .tarjeta h2 { font-size:.8rem; text-transform:uppercase; letter-spacing:.06em; color:var(--suave); margin:0 0 12px; font-weight:600; }
  .dos { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:20px; }
  @media (max-width:640px) { .dos { grid-template-columns:minmax(0,1fr); } }
  .frase { font-size:1.15rem; margin:0; }
  .vacio { color:var(--suave); font-style:italic; }
  .punto { width:10px; height:10px; border-radius:50%; display:inline-block; }
  h3 { font-size:.95rem; margin:0 0 4px; display:flex; align-items:center; gap:8px; }
  .cubo canvas { width:100%; display:block; touch-action:none; cursor:grab; }
  .nota { font-size:.8rem; color:var(--tenue); margin:8px 0 0; }
  .chips { display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
  .chip { border:1px solid var(--borde); border-radius:999px; padding:3px 10px; font-size:.85rem; }
  .tabla { overflow-x:auto; }
  table { width:100%; border-collapse:collapse; font-size:.85rem; font-variant-numeric:tabular-nums; }
  th { text-align:left; color:var(--suave); font-size:.72rem; text-transform:uppercase; letter-spacing:.04em; padding:6px; border-bottom:1px solid var(--borde); }
  td { padding:7px 6px; border-bottom:1px solid var(--borde); vertical-align:top; }
  td.n { text-align:right; white-space:nowrap; }
  .nav { display:flex; flex-wrap:wrap; gap:6px 14px; font-size:.9rem; margin:0 0 14px; }
  .nav a { color:var(--suave); text-decoration:none; border-bottom:2px solid transparent; padding-bottom:2px; }
  .nav a.aqui { color:var(--texto); border-color:var(--romeo); font-weight:600; }
  .emo { font-size:1.15rem; font-weight:700; margin:2px 0 4px; }
</style></head><body><main>
  <nav class="nav"><a href="./" class="aqui">Romeo en vivo</a><a href="calculo">Cálculo de valencia y arousal</a><a href="carrito">Carrito</a></nav>
  <h1>Romeo en vivo</h1>
  <p class="sub"><span class="vivo"><i></i><span id="estado">Leyendo el log…</span></span></p>

  <div class="dos">
    <div class="tarjeta"><h2>Lo que entendió</h2><p class="frase vacio" id="texto">—</p><div class="chips" id="lecturas"></div></div>
    <div class="tarjeta"><h2>Lo que respondió Romeo</h2><p class="frase vacio" id="respuesta">—</p><p class="nota" id="animo"></p></div>
  </div>

  <div class="tarjeta">
    <h2>Cómo cambia el ánimo, frase a frase</h2>
    <div class="dos">
      <div><h3><span class="punto" style="background:var(--romeo)"></span>Ánimo de Romeo</h3><p class="emo" id="emo-romeo">—</p><div class="cubo" id="cubo-romeo"><canvas></canvas></div></div>
      <div><h3><span class="punto" style="background:var(--fusion)"></span>Tu emoción (S<sub>u</sub>)</h3><p class="emo" id="emo-usuario">—</p><div class="cubo" id="cubo-usuario"><canvas></canvas></div></div>
    </div>
    <div class="chips" style="justify-content:center">
      <span class="chip"><span class="punto" style="background:var(--feliz)"></span> Feliz</span>
      <span class="chip"><span class="punto" style="background:var(--triste)"></span> Triste</span>
      <span class="chip"><span class="punto" style="background:var(--enojo)"></span> Enojado</span>
      <span class="chip"><span class="punto" style="background:var(--normal)"></span> Normal</span>
      <span class="chip"><span class="punto" style="background:var(--antes)"></span> Frase anterior</span>
      <span class="chip"><span class="punto" style="background:var(--ahora)"></span> Frase actual</span>
    </div>
    <p class="nota">Ejes: valencia, arousal y dominancia, de −1 a 1. Solo se ven dos puntos: el chico es la frase anterior y el grande la actual, con una flecha de uno a otro. Los rombos de color marcan dónde está cada emoción, y «Normal» es el centro: para ti el origen, para Romeo su ánimo base (anillo negro). La emoción sale de hacia dónde se aleja el punto del centro y cuánto, con la misma regla que usa Romeo (AEB). En «Tu emoción», un punto hueco es una frase en la que todo salió neutral: ahí Romeo no cambia su ánimo. Arrastra para girar; los dos cubos giran juntos.</p>
  </div>

  <div class="tarjeta">
    <h2>Todas las frases</h2>
    <div class="tabla"><table>
      <thead><tr><th>#</th><th>Dijiste</th><th>Lecturas</th><th class="n">Tú (v, a, d)</th><th class="n">Romeo (v, a, d)</th></tr></thead>
      <tbody id="tabla"><tr><td colspan="5" class="vacio">Háblale a Romeo.</td></tr></tbody>
    </table></div>
  </div>
</main>
<script>
const $ = (id) => document.getElementById(id);
const f2 = (x) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(2);
const vad = (p) => p ? `${f2(p.v)}, ${f2(p.a)}, ${f2(p.d)}` : "—";
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const EMO = { felicidad: "felicidad", tristeza: "tristeza", enojo: "enojo", miedo: "miedo", sorpresa: "sorpresa", asco: "asco", neutral: "neutral" };
let datos = { turnos: [], base: null }, firma = "";
const giro = { yaw: -0.6, pitch: 0.35 };

// Emocion de un punto, con la regla del AEB (aeb.mapear): direccion por
// coseno respecto al centro, zona neutra = 20 % del camino al anclaje y
// nivel 1-4 en tramos iguales. Para Romeo el centro es su animo base (da lo
// mismo que su log); para el usuario, el origen.
const ANCLAS = { felicidad: [0.76, 0.48, 0.35], tristeza: [-0.63, -0.27, -0.33], enojo: [-0.51, 0.59, 0.25] };
const NOMBRE = { felicidad: "Feliz", tristeza: "Triste", enojo: "Enojado", neutral: "Normal" };
const COLOR_EMO = { felicidad: "--feliz", tristeza: "--triste", enojo: "--enojo", neutral: "--normal" };
const INTENS = { 1: "un poco ", 2: "", 3: "muy ", 4: "muy " };
function emocionDe(p, centro) {
  if (!p) return { clave: "neutral", nivel: 0 };
  const dv = [p.v - centro[0], p.a - centro[1], p.d - centro[2]], mag = Math.hypot(...dv);
  if (mag < 1e-6) return { clave: "neutral", nivel: 0 };
  let mejor = null, cos = -2;
  for (const [k, a] of Object.entries(ANCLAS)) {
    const c = (dv[0] * a[0] + dv[1] * a[1] + dv[2] * a[2]) / (mag * Math.hypot(...a));
    if (c > cos) { cos = c; mejor = k; }
  }
  const a = ANCLAS[mejor], dist = Math.hypot(a[0] - centro[0], a[1] - centro[1], a[2] - centro[2]), radio = 0.2 * dist;
  if (mag < radio) return { clave: "neutral", nivel: 0 };
  return { clave: mejor, nivel: Math.max(1, Math.min(4, 1 + Math.floor(4 * (mag - radio) / (dist - radio)))) };
}
const textoEmo = (e) => e.clave === "neutral" ? "Normal" : (INTENS[e.nivel] + NOMBRE[e.clave].toLowerCase()).replace(/^./, (c) => c.toUpperCase());
const centroDe = (quien) => quien === "romeo" ? (datos.base ? [datos.base.v, datos.base.a, datos.base.d] : [0.4, 0.3, 0.2]) : [0, 0, 0];

function cubo(cont, quien) {
  const cv = cont.querySelector("canvas"), w = cont.clientWidth, h = Math.min(w, 340), dpr = window.devicePixelRatio || 1;
  cv.width = w * dpr; cv.height = h * dpr; cv.style.height = h + "px";
  const ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
  const css = getComputedStyle(document.documentElement), col = (n) => css.getPropertyValue(n).trim();
  const R = Math.min(w, h) * 0.27, cx = w / 2, cy = h / 2;
  const c1 = Math.cos(giro.yaw), s1 = Math.sin(giro.yaw), c2 = Math.cos(giro.pitch), s2 = Math.sin(giro.pitch);
  const P = (v, a, d) => { const x1 = v * c1 + d * s1, z1 = -v * s1 + d * c1, y2 = a * c2 - z1 * s2, z2 = a * s2 + z1 * c2, k = 4 / (4 + z2); return [cx + x1 * R * k, cy - y2 * R * k, z2]; };
  const L = (p, q, c, wd = 1) => { const A = P(...p), B = P(...q); ctx.beginPath(); ctx.strokeStyle = c; ctx.lineWidth = wd; ctx.moveTo(A[0], A[1]); ctx.lineTo(B[0], B[1]); ctx.stroke(); };
  for (const a of [-1, 1]) for (const b of [-1, 1]) { L([-1, a, b], [1, a, b], col("--eje")); L([a, -1, b], [a, 1, b], col("--eje")); L([a, b, -1], [a, b, 1], col("--eje")); }
  L([-1, 0, 0], [1, 0, 0], col("--tenue"), 1.5); L([0, -1, 0], [0, 1, 0], col("--tenue"), 1.5); L([0, 0, -1], [0, 0, 1], col("--tenue"), 1.5);
  ctx.font = "12px system-ui, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  for (const [p, t, c] of [[[1.32, 0, 0], "Valencia +", "--val"], [[0, 1.2, 0], "Arousal +", "--aro"], [[0, 0, 1.34], "Dominancia +", "--dom"], [[-1.2, 0, 0], "−", "--tenue"], [[0, -1.18, 0], "−", "--tenue"], [[0, 0, -1.2], "−", "--tenue"]]) { const [x, y] = P(...p); ctx.fillStyle = col(c); ctx.fillText(t, x, y); }
  const cAntes = col("--antes"), cAhora = col("--ahora"), color = cAhora;
  // Solo la frase anterior y la actual, no todo el registro.
  const turnos = datos.turnos.slice(-2);
  const pts = turnos.map((t) => quien === "romeo" ? (t.romeo || datos.base) : (t.fusion || { v: 0, a: 0, d: 0 }));
  const n = pts.length;
  ctx.globalAlpha = 0.3; pts.forEach((p, i) => { if (!p) return; const [x, y] = P(p.v, -1, p.d); ctx.beginPath(); ctx.fillStyle = i === n - 1 ? cAhora : cAntes; ctx.arc(x, y, 3, 0, 7); ctx.fill(); }); ctx.globalAlpha = 1;
  if (n === 2 && pts[0] && pts[1]) {
    const A = P(pts[0].v, pts[0].a, pts[0].d), B = P(pts[1].v, pts[1].a, pts[1].d);
    ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.setLineDash([5, 4]); ctx.moveTo(A[0], A[1]); ctx.lineTo(B[0], B[1]); ctx.stroke(); ctx.setLineDash([]);
    const ang = Math.atan2(B[1] - A[1], B[0] - A[0]), lejos = Math.hypot(B[0] - A[0], B[1] - A[1]);
    if (lejos > 16) {   // punta de flecha justo antes del punto actual
      const px = B[0] - 9 * Math.cos(ang), py = B[1] - 9 * Math.sin(ang);
      ctx.beginPath(); ctx.fillStyle = color; ctx.moveTo(px, py);
      ctx.lineTo(px - 9 * Math.cos(ang - 0.45), py - 9 * Math.sin(ang - 0.45));
      ctx.lineTo(px - 9 * Math.cos(ang + 0.45), py - 9 * Math.sin(ang + 0.45)); ctx.fill();
    }
  }
  // Referencias: donde estan Feliz, Triste y Enojado, y el centro "Normal".
  const centro = centroDe(quien);
  ctx.font = "11px system-ui, sans-serif";
  for (const [k, a] of Object.entries(ANCLAS)) {
    const [x, y] = P(...a);
    const ce = col(COLOR_EMO[k]);
    ctx.beginPath(); ctx.fillStyle = ce; ctx.moveTo(x, y - 7); ctx.lineTo(x + 7, y); ctx.lineTo(x, y + 7); ctx.lineTo(x - 7, y); ctx.fill();
    ctx.font = "bold 11px system-ui, sans-serif"; ctx.fillStyle = ce; ctx.fillText(NOMBRE[k], x, y + 16);
  }
  if (quien === "usuario") {
    const [x, y] = P(0, 0, 0), cn = col("--normal");
    ctx.beginPath(); ctx.fillStyle = cn; ctx.moveTo(x, y - 7); ctx.lineTo(x + 7, y); ctx.lineTo(x, y + 7); ctx.lineTo(x - 7, y); ctx.fill();
    ctx.font = "bold 11px system-ui, sans-serif"; ctx.fillText("Normal", x + 26, y + 12);
  }
  const marcas = pts.map((p, i) => { if (!p) return null; const [x, y, z] = P(p.v, p.a, p.d); return { x, y, z, ultimo: i === n - 1, hueco: quien === "usuario" && !turnos[i].fusion, emo: textoEmo(emocionDe(p, centro)) }; }).filter(Boolean);
  if (quien === "romeo" && datos.base) { const [x, y, z] = P(datos.base.v, datos.base.a, datos.base.d); marcas.push({ x, y, z, base: true }); }
  marcas.sort((a, b) => b.z - a.z);
  for (const m of marcas) {
    ctx.beginPath();
    if (m.base) { ctx.arc(m.x, m.y, 7, 0, 7); ctx.strokeStyle = col("--normal"); ctx.lineWidth = 2.5; ctx.stroke(); ctx.fillStyle = col("--normal"); ctx.font = "bold 11px system-ui, sans-serif"; ctx.fillText("Normal", m.x + 26, m.y + 12); continue; }
    const cp = m.ultimo ? cAhora : cAntes;
    ctx.arc(m.x, m.y, m.ultimo ? 8 : 6, 0, 7); ctx.fillStyle = m.hueco ? col("--tarjeta") : cp; ctx.fill();
    ctx.strokeStyle = m.hueco ? cp : col("--tarjeta"); ctx.lineWidth = 2; ctx.stroke();
    ctx.fillStyle = cp; ctx.font = "bold 11px system-ui, sans-serif";
    ctx.fillText(m.ultimo ? "ahora" : "antes", m.x, m.y - 15);
  }
  if (!n) { ctx.fillStyle = col("--suave"); ctx.font = "italic 13px system-ui, sans-serif"; ctx.fillText("Háblale a Romeo para ver el recorrido.", cx, h - 14); }
}
const cubos = () => { cubo($("cubo-romeo"), "romeo"); cubo($("cubo-usuario"), "usuario"); };

for (const id of ["cubo-romeo", "cubo-usuario"]) {
  const cv = $(id).querySelector("canvas"); let prev = null;
  cv.addEventListener("pointerdown", (e) => { prev = [e.clientX, e.clientY]; cv.setPointerCapture(e.pointerId); });
  cv.addEventListener("pointermove", (e) => { if (!prev) return; giro.yaw += (e.clientX - prev[0]) * 0.01; giro.pitch = Math.max(-1.4, Math.min(1.4, giro.pitch + (e.clientY - prev[1]) * 0.01)); prev = [e.clientX, e.clientY]; cubos(); });
  cv.addEventListener("pointerup", () => prev = null); cv.addEventListener("pointercancel", () => prev = null);
}
window.addEventListener("resize", cubos);

const lect = (t) => t.lecturas.map((l) => `${l.fuente.split(" ")[0]}: ${EMO[l.emocion] || l.emocion} ${Math.round(l.confianza * 100)}%`).concat(t.sin_cara ? ["Cara: no vio"] : []);

function mostrar(d) {
  datos = d;
  $("estado").textContent = d.error || `En vivo · perfil ${d.perfil || "—"} (empatía ${d.empatia ?? "—"}) · ${d.turnos.length} frases`;
  const ult = d.turnos[d.turnos.length - 1];
  if (ult) {
    $("texto").className = "frase"; $("texto").textContent = ult.texto;
    $("lecturas").innerHTML = lect(ult).map((s) => `<span class="chip">${esc(s)}</span>`).join("") + `<span class="chip">Tú: ${ult.fusion ? vad(ult.fusion) : "neutral"}</span>`;
  }
  if (d.ultima_respuesta) { $("respuesta").className = "frase"; $("respuesta").textContent = d.ultima_respuesta; }
  for (const quien of ["romeo", "usuario"]) {
    const t = d.turnos[d.turnos.length - 1];
    const p = !t ? (quien === "romeo" ? d.base : null) : (quien === "romeo" ? (t.romeo || d.base) : t.fusion);
    const e = emocionDe(p, centroDe(quien));
    const el = $("emo-" + quien);
    el.textContent = (quien === "romeo" ? "Romeo está: " : "Tú estás: ") + textoEmo(e);
    el.style.color = `var(${COLOR_EMO[e.clave]})`;
    const prev = d.turnos[d.turnos.length - 2];
    if (prev) {
      const pp = quien === "romeo" ? (prev.romeo || d.base) : prev.fusion;
      const s = document.createElement("span");
      s.style.cssText = "display:block;font-size:.8rem;font-weight:400;color:var(--suave)";
      s.textContent = "antes: " + textoEmo(emocionDe(pp, centroDe(quien)));
      el.appendChild(s);
    }
  }
  if (d.romeo) $("animo").textContent = `Ánimo de Romeo: ${vad(d.romeo)}${d.estado_romeo ? " · " + d.estado_romeo.replace("_L", " nivel ") : ""}`;
  $("tabla").innerHTML = d.turnos.length ? d.turnos.map((t, i) => `<tr><td>${i + 1}</td><td>${esc(t.texto)}</td><td>${esc(lect(t).join(" · ") || "—")}</td><td class="n">${t.fusion ? vad(t.fusion) : "neutral"}</td><td class="n">${vad(t.romeo)}</td></tr>`).reverse().join("")
    : '<tr><td colspan="5" class="vacio">Háblale a Romeo.</td></tr>';
  cubos();
}

async function actualizar() {
  try {
    const d = await (await fetch("estado")).json();
    const nueva = JSON.stringify([d.turnos.length, d.ultima_respuesta, d.romeo]);
    if (nueva !== firma) { firma = nueva; mostrar(d); }
  } catch (e) { $("estado").textContent = "Sin conexión con la Pi, reintentando…"; }
}
actualizar(); setInterval(actualizar, 2000);

</script></body></html>"""


PAGINA_CARRITO = r"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Carrito de Romeo</title>
<style>
  :root { --fondo:#f9f9f7; --tarjeta:#fcfcfb; --texto:#0b0b0b; --suave:#52514e; --tenue:#898781;
    --borde:#e1e0d9; --eje:#c3c2b7; --fusion:#eda100; --romeo:#e87ba4;
    --val:#2a78d6; --aro:#eb6834; --dom:#1baf7a; --ok:#1baf7a; }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
    --fondo:#0d0d0d; --tarjeta:#1a1a19; --texto:#f0efec; --suave:#c3c2b7; --tenue:#898781;
    --borde:#2c2c2a; --eje:#383835; --fusion:#c98500; --romeo:#d55181;
    --val:#3987e5; --aro:#d95926; --dom:#199e70; --ok:#199e70; color-scheme: dark; } }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--fondo); color:var(--texto); font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
  main { max-width:960px; margin:0 auto; padding:28px 16px 48px; }
  h1 { font-size:1.5rem; margin:0 0 4px; }
  .sub { color:var(--suave); margin:0 0 16px; overflow-wrap:anywhere; }
  .vivo { display:inline-flex; align-items:center; gap:6px; font-size:.85rem; color:var(--suave); }
  .vivo i { width:8px; height:8px; border-radius:50%; background:var(--ok); }
  .tarjeta { background:var(--tarjeta); border:1px solid var(--borde); border-radius:12px; padding:20px; margin-bottom:16px; min-width:0; }
  .tarjeta h2 { font-size:.8rem; text-transform:uppercase; letter-spacing:.06em; color:var(--suave); margin:0 0 12px; font-weight:600; }
  .dos { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:20px; }
  @media (max-width:640px) { .dos { grid-template-columns:minmax(0,1fr); } }
  .frase { font-size:1.15rem; margin:0; }
  .vacio { color:var(--suave); font-style:italic; }
  .punto { width:10px; height:10px; border-radius:50%; display:inline-block; }
  h3 { font-size:.95rem; margin:0 0 4px; display:flex; align-items:center; gap:8px; }
  .cubo canvas { width:100%; display:block; touch-action:none; cursor:grab; }
  .nota { font-size:.8rem; color:var(--tenue); margin:8px 0 0; }
  .chips { display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
  .chip { border:1px solid var(--borde); border-radius:999px; padding:3px 10px; font-size:.85rem; }
  .tabla { overflow-x:auto; }
  table { width:100%; border-collapse:collapse; font-size:.85rem; font-variant-numeric:tabular-nums; }
  th { text-align:left; color:var(--suave); font-size:.72rem; text-transform:uppercase; letter-spacing:.04em; padding:6px; border-bottom:1px solid var(--borde); }
  td { padding:7px 6px; border-bottom:1px solid var(--borde); vertical-align:top; }
  td.n { text-align:right; white-space:nowrap; }
  .nav { display:flex; flex-wrap:wrap; gap:6px 14px; font-size:.9rem; margin:0 0 14px; }
  .nav a { color:var(--suave); text-decoration:none; border-bottom:2px solid transparent; padding-bottom:2px; }
  .nav a.aqui { color:var(--texto); border-color:var(--romeo); font-weight:600; }
  .emo { font-size:1.15rem; font-weight:700; margin:2px 0 4px; }
  .carrito { display:flex; flex-direction:column; gap:20px; align-items:center; text-align:center; }
  .pad { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; width:min(100%,420px); }
  .pad button, .giro button { font:inherit; font-size:2.2rem; aspect-ratio:1; border:1px solid var(--borde); border-radius:14px;
    background:var(--fondo); color:var(--texto); cursor:pointer; touch-action:none; user-select:none; }
  .pad button.activo, .giro button.activo { background:var(--romeo); color:#fff; border-color:var(--romeo); }
  .pad button.stop { background:#d1352b; color:#fff; border-color:#d1352b; font-size:1.4rem; font-weight:700; }
  .giro { display:grid; grid-template-columns:1fr 1fr; gap:12px; width:min(100%,420px); margin-top:12px; }
  .giro button { aspect-ratio:auto; padding:22px 0; font-size:1.3rem; }
  .ayuda-carrito { max-width:420px; min-width:0; }
  .carrito > div:first-child { width:min(100%,420px); }
  .pad, .giro { margin-left:auto; margin-right:auto; }
</style></head><body><main>
  <nav class="nav"><a href="./">Romeo en vivo</a><a href="calculo">Cálculo de valencia y arousal</a><a href="carrito" class="aqui">Carrito</a></nav>
  <h1>Carrito</h1>
  <p class="sub">Mueve el carrito de Romeo. Funciona con Romeo encendido: lo sirve su nodo del carrito.</p>

  <div class="tarjeta">
    <h2>Controles</h2>
    <div class="carrito">
      <div>
        <div class="pad">
          <button data-c="FL" aria-label="Diagonal adelante izquierda">↖</button>
          <button data-c="F" aria-label="Adelante">↑</button>
          <button data-c="FR" aria-label="Diagonal adelante derecha">↗</button>
          <button data-c="SL" aria-label="Lateral izquierda">←</button>
          <button class="stop" data-stop aria-label="Parar">STOP</button>
          <button data-c="SR" aria-label="Lateral derecha">→</button>
          <button data-c="BL" aria-label="Diagonal atrás izquierda">↙</button>
          <button data-c="B" aria-label="Atrás">↓</button>
          <button data-c="BR" aria-label="Diagonal atrás derecha">↘</button>
        </div>
        <div class="giro">
          <button data-c="RL" aria-label="Girar a la izquierda">⟲ Girar</button>
          <button data-c="RR" aria-label="Girar a la derecha">Girar ⟳</button>
        </div>
      </div>
      <div class="ayuda-carrito">
        <p class="frase" id="carrito-estado" style="font-size:1rem">Mantén apretado un botón para mover el carrito; al soltarlo se para.</p>
        <p class="nota">En la PC también sirven las flechas del teclado y Q / E para girar. Si se corta el WiFi, el carrito se para solo en medio segundo.</p>
      </div>
    </div>
  </div>

</main>
<script>
const $ = (id) => document.getElementById(id);
// ── Carrito: mientras el boton esta apretado manda el comando cada 150 ms
// (el firmware se para solo si deja de recibir); al soltar manda "S".
let reloj = null, botonActivo = null;
async function mandar(c) {
  try {
    const r = await fetch("cmd/" + c, { method: "POST" });
    if (!r.ok) { const d = await r.json().catch(() => ({})); $("carrito-estado").textContent = d.error || "El carrito no responde."; }
    else if (c !== "S") $("carrito-estado").textContent = "Moviendo: " + (botonActivo ? botonActivo.getAttribute("aria-label") : c);
    else $("carrito-estado").textContent = "Parado.";
  } catch (e) { $("carrito-estado").textContent = "Sin conexión con la Pi."; }
}
function empezar(c, b) { parar(false); botonActivo = b; if (b) b.classList.add("activo"); mandar(c); reloj = setInterval(() => mandar(c), 150); }
function parar(enviar = true) {
  if (reloj) { clearInterval(reloj); reloj = null; }
  if (botonActivo) { botonActivo.classList.remove("activo"); botonActivo = null; }
  if (enviar) mandar("S");
}
document.querySelectorAll("button[data-c]").forEach((b) => {
  b.addEventListener("pointerdown", (e) => { e.preventDefault(); b.setPointerCapture(e.pointerId); empezar(b.dataset.c, b); });
  ["pointerup", "pointercancel", "lostpointercapture"].forEach((ev) => b.addEventListener(ev, () => { if (botonActivo === b) parar(); }));
});
document.querySelector("[data-stop]").addEventListener("pointerdown", (e) => { e.preventDefault(); parar(); });
const teclas = { w: "F", arrowup: "F", s: "B", arrowdown: "B", a: "SL", arrowleft: "SL", d: "SR", arrowright: "SR", q: "RL", e: "RR" };
document.addEventListener("keydown", (e) => { const c = teclas[e.key.toLowerCase()]; if (c && !e.repeat) { e.preventDefault(); empezar(c, document.querySelector(`[data-c="${c}"]`)); } });
document.addEventListener("keyup", (e) => { if (teclas[e.key.toLowerCase()]) parar(); });
window.addEventListener("blur", () => { if (reloj) parar(); });
document.addEventListener("visibilitychange", () => { if (document.hidden && reloj) parar(); });
</script></body></html>"""


CALCULO = r"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cálculo afectivo de Romeo</title>
<style>
  :root { --fondo:#f9f9f7; --tarjeta:#fcfcfb; --texto:#0b0b0b; --suave:#52514e; --tenue:#898781;
    --borde:#e1e0d9; --barra:#eceae4; --fusion:#eda100; --romeo:#e87ba4; --ser:#eb6834; --ter:#1baf7a; --fer:#2a78d6;
    --val:#2a78d6; --aro:#eb6834; --dom:#1baf7a; }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
    --fondo:#0d0d0d; --tarjeta:#1a1a19; --texto:#f0efec; --suave:#c3c2b7; --tenue:#898781;
    --borde:#2c2c2a; --barra:#2c2c2a; --fusion:#c98500; --romeo:#d55181; --ser:#d95926; --ter:#199e70; --fer:#3987e5;
    --val:#3987e5; --aro:#d95926; --dom:#199e70; color-scheme: dark; } }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--fondo); color:var(--texto); font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
  main { max-width:960px; margin:0 auto; padding:28px 16px 48px; }
  h1 { font-size:1.5rem; margin:0 0 4px; }
  .sub { color:var(--suave); margin:0 0 16px; overflow-wrap:anywhere; }
  a { color:inherit; }
  .barra-sup { display:flex; flex-wrap:wrap; gap:10px 16px; align-items:center; margin-bottom:16px; }
  select, label { font:inherit; font-size:.9rem; }
  select { padding:6px 8px; border-radius:8px; border:1px solid var(--borde); background:var(--tarjeta); color:var(--texto); max-width:100%; }
  .paso { background:var(--tarjeta); border:1px solid var(--borde); border-radius:12px; padding:20px; margin-bottom:16px; min-width:0; }
  .paso h2 { font-size:1.05rem; margin:0 0 4px; display:flex; gap:10px; align-items:baseline; }
  .paso h2 .n { font-size:.75rem; color:var(--suave); border:1px solid var(--borde); border-radius:999px; padding:1px 8px; font-weight:600; }
  .que { color:var(--suave); font-size:.9rem; margin:0 0 12px; }
  .formula { font-family:ui-monospace,Consolas,monospace; font-size:.9rem; background:var(--fondo); border:1px solid var(--borde);
    border-radius:8px; padding:8px 12px; margin:8px 0; overflow-x:auto; white-space:pre-wrap; overflow-wrap:anywhere; }
  .tabla { overflow-x:auto; }
  table { width:100%; border-collapse:collapse; font-size:.85rem; font-variant-numeric:tabular-nums; }
  th { text-align:left; color:var(--suave); font-size:.72rem; text-transform:uppercase; letter-spacing:.04em; padding:6px; border-bottom:1px solid var(--borde); }
  td { padding:6px; border-bottom:1px solid var(--borde); }
  td.n, th.n { text-align:right; white-space:nowrap; }
  tr.total td { font-weight:600; border-top:2px solid var(--borde); }
  .barrita { height:8px; background:var(--barra); border-radius:4px; overflow:hidden; min-width:60px; }
  .barrita i { display:block; height:100%; border-radius:4px; }
  .modalidades { display:grid; grid-template-columns:minmax(0,1fr); gap:16px; }
  .modalidad { border:1px solid var(--borde); border-radius:10px; padding:14px; min-width:0; }
  .modalidad h3 { margin:0 0 6px; font-size:.95rem; display:flex; gap:8px; align-items:center; }
  .punto { width:10px; height:10px; border-radius:50%; display:inline-block; }
  .resultado { font-weight:600; margin:8px 0 0; }
  .vacio { color:var(--suave); font-style:italic; }
  .chip { display:inline-block; border:1px solid var(--borde); border-radius:999px; padding:2px 10px; font-size:.85rem; font-weight:600; }
  .nav { display:flex; flex-wrap:wrap; gap:6px 14px; font-size:.9rem; margin:0 0 14px; }
  .nav a { color:var(--suave); text-decoration:none; border-bottom:2px solid transparent; padding-bottom:2px; }
  .nav a.aqui { color:var(--texto); border-color:var(--romeo); font-weight:600; }
</style></head><body><main>
  <nav class="nav"><a href="./">Romeo en vivo</a><a href="calculo" class="aqui">Cálculo de valencia y arousal</a><a href="carrito">Carrito</a></nav>
  <h1>Cálculo de valencia y arousal</h1>
  <p class="sub">Cada paso con los números reales del turno, tal como los calcula el robot (PULSE → ROSE → AEB, metodología de Rocha).</p>
  <div class="barra-sup">
    <select id="turno" aria-label="Frase"></select>
    <label><input type="checkbox" id="seguir" checked> Seguir la última frase</label>
    <span class="que" id="info" style="margin:0"></span>
  </div>
  <div id="pasos"><p class="vacio">Esperando la primera frase. Háblale a Romeo.</p></div>
</main>
<script>
const $ = (id) => document.getElementById(id);
const n2 = (x) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(2);
const n3 = (x) => (x >= 0 ? "" : "−") + Math.abs(x).toFixed(3);
const v3 = (v) => v ? `(${n2(v[0])}, ${n2(v[1])}, ${n2(v[2])})` : "—";
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const FUENTE = { SER: ["Voz (SER)", "--ser"], TER: ["Texto (TER)", "--ter"], FER: ["Cara (FER)", "--fer"] };
let turnos = [], firma = "";

function modalidad(l) {
  const [nombre, color] = FUENTE[l.fuente] || [l.fuente, "--suave"];
  const extra = l.cuadros ? ` · ${l.cuadros} cuadros` : l.ventana ? ` · mediana de las últimas ${l.ventana} frases` : "";
  const filas = l.filas.map((f) => `<tr><td>${esc(f.emocion)}</td>
      <td><div class="barrita"><i style="width:${(f.p * 100).toFixed(0)}%;background:var(${color})"></i></div></td>
      <td class="n">${f.p.toFixed(3)}</td><td class="n">${v3(f.ancla)}</td><td class="n">${v3(f.aporte)}</td></tr>`).join("");
  const terminos = l.filas.filter((f) => f.p > 0).map((f) => `${f.p.toFixed(3)}·ln ${f.p.toFixed(3)}`).join(" + ");
  return `<div class="modalidad">
    <h3><span class="punto" style="background:var(${color})"></span>${nombre}<span class="que" style="margin:0;font-weight:400">${extra}</span></h3>
    <div class="tabla"><table>
      <thead><tr><th>Emoción</th><th></th><th class="n">p</th><th class="n">Ancla (v, a, d)</th><th class="n">p × ancla</th></tr></thead>
      <tbody>${filas}<tr class="total"><td colspan="4">S<sub>${l.fuente}</sub> = Σ p × ancla</td><td class="n">${v3(l.vad)}</td></tr></tbody>
    </table></div>
    <div class="formula">H = −(${terminos}) / ln ${l.k} = ${l.entropia.toFixed(3)}
c = 1 − H = ${l.confianza.toFixed(3)}</div>
    <p class="resultado">Dominante: ${esc(l.dominante)} · confianza ${l.confianza.toFixed(2)}</p>
  </div>`;
}

// La camara se consulto pero no vio ninguna cara: la tarjeta queda igual,
// diciendo por que no hay lectura, en vez de desaparecer.
function sinCara(t) {
  if (t.fer_sin_cara == null || t.lecturas.some((l) => l.fuente === "FER")) return "";
  return `<div class="modalidad"><h3><span class="punto" style="background:var(--fer)"></span>Cara (FER)</h3>
    <p class="vacio" style="margin:0">Sin lectura: la cámara no vio ninguna cara en los ${t.fer_sin_cara} s que duró la frase (0 cuadros), así que la cara no entra en la fusión.</p>
    <p class="que" style="margin:8px 0 0">Si pasa en todas las frases, revisa que la cámara mande imagen: mira a la cámara al hablar y comprueba su cable.</p></div>`;
}

function render(t) {
  const p = t.perfil, b = p.base;
  let h = "";
  // 0. Decaimiento
  const d0 = t.decaer.antes, d1 = t.decaer.despues, movio = d0.some((x, i) => Math.abs(x - d1[i]) > 1e-4);
  h += `<section class="paso"><h2><span class="n">Paso 0</span>ROSE · Decaimiento hacia el ánimo base</h2>
    <p class="que">Antes de leer la frase, el ánimo de Romeo vuelve un poco a su base según el tiempo que pasó (latencia ${p.latencia} s, un paso cada ${p.paso} s).</p>
    <div class="formula">s_r ← b + (1 − d)ⁿ · (s_r − b)      d = ${p.decaimiento}   b = ${v3(b)}
${v3(d0)}  →  ${v3(d1)}${movio ? "" : "   (todavía dentro de la latencia: no decae)"}</div></section>`;
  // 1. Modalidades
  h += `<section class="paso"><h2><span class="n">Paso 1</span>PULSE · De probabilidades a VAD, por modalidad</h2>
    <p class="que">Cada clasificador da una probabilidad por emoción (agrupadas en las cuatro del robot). El punto VAD es el promedio de los anclajes pesado por esas probabilidades, y la confianza sale de la entropía: si duda entre varias emociones, pesa menos.</p>
    ${t.lecturas.length || t.fer_sin_cara != null ? `<div class="modalidades">${t.lecturas.map(modalidad).join("")}${sinCara(t)}</div>` : '<p class="vacio">Ninguna modalidad dio lectura en esta frase.</p>'}</section>`;
  // 2. Fusion
  const fu = t.fusion;
  let cuerpo;
  if (!fu) cuerpo = '<p class="vacio">Sin lecturas: no hay nada que fusionar.</p>';
  else if (fu.todos_neutral) cuerpo = `<p class="resultado">Todas las modalidades vieron neutral → el ánimo de Romeo no se mueve en esta frase.</p>
      <p class="que">Neutral es el origen (0, 0, 0). Como la base de Romeo es positiva, acercarlo al origen se vería como tristeza, así que se omite.</p>`;
  else {
    const filas = fu.entradas.map((e) => `<tr><td>${esc((FUENTE[e.fuente] || [e.fuente])[0])}</td><td class="n">${e.confianza.toFixed(3)}</td>
        <td class="n">${v3(e.vad)}</td><td class="n">${v3(e.vad.map((x) => x * e.confianza))}</td></tr>`).join("");
    cuerpo = `<div class="formula">S_u = Σ c_m · S_m / Σ c_m</div>
      <div class="tabla"><table><thead><tr><th>Fuente</th><th class="n">c</th><th class="n">S_m (v, a, d)</th><th class="n">c × S_m</th></tr></thead>
      <tbody>${filas}<tr class="total"><td>Σ</td><td class="n">${(fu.suma_confianzas ?? 0).toFixed(3)}</td><td></td>
      <td class="n">S_u = ${v3(fu.s_u)}</td></tr></tbody></table></div>
      ${fu.encoge ? `<div class="formula">Extensión ROMEO_NEUTRAL_ENCOGE: S_u × ${fu.encoge.peso.toFixed(2)} = ${v3(fu.encoge.s_u)}</div>` : ""}
      <p class="resultado">Tu emoción en esta frase: S_u = ${v3(fu.encoge ? fu.encoge.s_u : fu.s_u)}</p>`;
  }
  h += `<section class="paso"><h2><span class="n">Paso 2</span>PULSE · Fusión de las modalidades</h2>
    <p class="que">Se juntan voz, texto y cara en un solo punto, pesando cada una por su confianza.</p>${cuerpo}</section>`;
  // 3. Empatia
  const em = t.empatia;
  h += `<section class="paso"><h2><span class="n">Paso 3</span>ROSE · Empatía: Romeo se mueve hacia tu emoción</h2>
    <p class="que">Romeo toma una parte (e) de tu emoción y conserva el resto de la suya: acompaña sin hundirse.</p>
    ${em ? `<div class="formula">s_r' = e · S_u + (1 − e) · s_r        e = ${em.e}
${["v", "a", "d"].map((k, i) => `${k}: ${em.e} × ${n3(em.s_u[i])} + ${(1 - em.e).toFixed(1)} × ${n3(em.antes[i])} = ${n3(em.despues[i])}`).join("\n")}</div>
      <p class="resultado">Ánimo de Romeo: ${v3(em.antes)} → ${v3(em.despues)}</p>`
    : '<p class="vacio">No se aplicó: no hubo emoción que contagiar en esta frase.</p>'}</section>`;
  // 4. AEB
  const a = t.aeb;
  const cos = Object.entries(a.cosenos || {}).sort((x, y) => y[1] - x[1])
    .map(([e, c], i) => `<tr><td>${esc(e)}${i === 0 ? " ←" : ""}</td><td class="n">${c.toFixed(3)}</td></tr>`).join("");
  h += `<section class="paso"><h2><span class="n">Paso 4</span>AEB · Del ánimo a una emoción con nivel</h2>
    <p class="que">Se mide cuánto y hacia dónde se apartó Romeo de su base. La dirección se elige por el coseno con cada anclaje, y el nivel por qué tan lejos llegó, en cuatro tramos.</p>
    <div class="formula">Δ = s_r − b = ${v3(a.vector)} − ${v3(a.base)} = ${v3(a.desviacion)}
|Δ| = ${a.magnitud.toFixed(3)}</div>
    ${cos ? `<div class="tabla"><table><thead><tr><th>Dirección</th><th class="n">cos(Δ, ancla)</th></tr></thead><tbody>${cos}</tbody></table></div>` : ""}
    ${a.emocion ? `<div class="formula">radio neutro = ${a.fraccion_neutral} × |ancla(${a.emocion}) − b| = ${a.radio.toFixed(3)}
${a.magnitud < a.radio ? `|Δ| = ${a.magnitud.toFixed(3)} < radio → zona neutra, nivel 0`
  : `nivel = 1 + ⌊4 · (|Δ| − radio) / recorrido⌋ = 1 + ⌊4 · (${a.magnitud.toFixed(3)} − ${a.radio.toFixed(3)}) / ${a.recorrido.toFixed(3)}⌋ = ${a.nivel}`}</div>` : ""}
    <p class="resultado">Resultado: <span class="chip">${esc(a.estado.replace("_L", " · nivel "))}</span> · cara ${esc(a.cara)}${a.emocion === "tristeza" && a.cara === "content" ? " (con tristeza Romeo mantiene la cara normal)" : ""}</p></section>`;
  $("pasos").innerHTML = h;
  $("info").textContent = `«${t.texto}» · ${new Date(t.t * 1000).toLocaleTimeString()}`;
}

function llenarSelector() {
  const sel = $("turno"), previo = sel.value;
  sel.innerHTML = turnos.map((t, i) => `<option value="${i}">${i + 1}. ${esc(t.texto).slice(0, 50)}</option>`).reverse().join("");
  sel.value = $("seguir").checked || previo === "" ? String(turnos.length - 1) : previo;
}

async function actualizar() {
  try {
    const d = await (await fetch("traza")).json();
    if (d.error) { $("info").textContent = d.error; return; }
    const nueva = JSON.stringify(d.turnos.map((t) => t.t));
    if (nueva === firma) return;
    firma = nueva; turnos = d.turnos;
    if (!turnos.length) return;
    llenarSelector();
    render(turnos[+$("turno").value]);
  } catch (e) { $("info").textContent = "Sin conexión con la Pi, reintentando…"; }
}
$("turno").addEventListener("change", () => { $("seguir").checked = +$("turno").value === turnos.length - 1; render(turnos[+$("turno").value]); });
$("seguir").addEventListener("change", () => { if ($("seguir").checked && turnos.length) { $("turno").value = String(turnos.length - 1); render(turnos[turnos.length - 1]); } });
actualizar(); setInterval(actualizar, 2000);
</script></body></html>"""


if __name__ == "__main__":
    print(f"Panel de Romeo en http://0.0.0.0:{PUERTO}/ leyendo {LOG}")
    app.run(host="0.0.0.0", port=PUERTO, threaded=True)
