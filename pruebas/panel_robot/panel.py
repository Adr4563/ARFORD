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
# turno (romeo_brain/traza_afectiva.py). El default coincide con donde la
# escribe lanzar_romeo.sh (ROMEO_TRAZA_AFECTO=~/romeo/romeo_afecto.jsonl):
# si el panel arranca sin esa variable en el entorno, cae a la misma ruta
# en vez de buscar en ~/ y no encontrarla.
TRAZA = os.path.expanduser(
    os.environ.get("ROMEO_TRAZA_AFECTO", "~/romeo/romeo_afecto.jsonl"))


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
<link rel="stylesheet" media="print" onload="this.media='all'" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Manrope:wght@400;600;700;800&display=swap">
<style>/*TEMA*/</style></head><body>
<header class="top"><div class="top-in"><p class="marca"><span class="rombo" style="background:var(--romeo)"></span>ROMEO</p><nav class="nav"><a href="./" class="aqui">En vivo</a><a href="calculo">Cálculo VAD</a><a href="carrito">Carrito</a></nav><span class="vivo"><i></i><span id="estado">Leyendo el log…</span></span></div></header>
<main>
  <div class="dos">
    <section class="tarjeta"><h2>Lo que entendió</h2><p class="frase vacio" id="texto">—</p><div class="medidores" id="lecturas"></div></section>
    <section class="tarjeta"><h2>Lo que respondió Romeo</h2><p class="frase vacio" id="respuesta">—</p><p class="nota" id="animo"></p></section>
  </div>

  <div class="dos">
      <section class="modulo romeo">
        <header>Ánimo de Romeo</header>
        <div class="modulo-in">
          <p class="emo" id="emo-romeo">—</p>
          <div class="vad" id="vad-romeo"></div>
          <div class="cubo" id="cubo-romeo"><canvas></canvas></div>
        </div>
      </section>
      <section class="modulo usuario">
        <header>Tu emoción · S<sub>u</sub></header>
        <div class="modulo-in">
          <p class="emo" id="emo-usuario">—</p>
          <div class="vad" id="vad-usuario"></div>
          <div class="cubo" id="cubo-usuario"><canvas></canvas></div>
        </div>
      </section>
  </div>

  <section class="tarjeta">
    <div class="leyenda"><span><i class="rombo" style="background:var(--feliz)"></i>Feliz</span><span><i class="rombo" style="background:var(--triste)"></i>Triste</span><span><i class="rombo" style="background:var(--enojo)"></i>Enojado</span><span><i class="rombo" style="background:var(--normal)"></i>Normal</span><span><i class="rombo" style="background:var(--punto-romeo)"></i>Romeo</span><span><i class="rombo" style="background:var(--punto-usuario)"></i>Tú</span></div>
  </section>

  <section class="tarjeta">
    <h2>Todas las frases</h2>
    <div class="tabla"><table>
      <thead><tr><th>#</th><th>Dijiste</th><th>Lecturas</th><th class="n">Tú (v, a, d)</th><th class="n">Romeo (v, a, d)</th></tr></thead>
      <tbody id="tabla"><tr><td colspan="5" class="vacio">Háblale a Romeo.</td></tr></tbody>
    </table></div>
  </section>
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
  const cv = cont.querySelector("canvas"), w = cont.clientWidth, h = Math.min(w, 420), dpr = window.devicePixelRatio || 1;
  cv.width = w * dpr; cv.height = h * dpr; cv.style.height = h + "px";
  const ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
  const css = getComputedStyle(document.documentElement), col = (n) => css.getPropertyValue(n).trim();
  const R = Math.min(w, h) * 0.26, cx = w / 2, cy = h / 2 + 10;
  const c1 = Math.cos(giro.yaw), s1 = Math.sin(giro.yaw), c2 = Math.cos(giro.pitch), s2 = Math.sin(giro.pitch);
  const P = (v, a, d) => { const x1 = v * c1 + d * s1, z1 = -v * s1 + d * c1, y2 = a * c2 - z1 * s2, z2 = a * s2 + z1 * c2, k = 4 / (4 + z2); return [cx + x1 * R * k, cy - y2 * R * k, z2]; };
  // Todos los puntos del cubo son rombos.
  const rombo = (x, y, r) => { ctx.moveTo(x, y - r); ctx.lineTo(x + r, y); ctx.lineTo(x, y + r); ctx.lineTo(x - r, y); ctx.closePath(); };
  const L = (p, q, c, wd = 1) => { const A = P(...p), B = P(...q); ctx.beginPath(); ctx.strokeStyle = c; ctx.lineWidth = wd; ctx.moveTo(A[0], A[1]); ctx.lineTo(B[0], B[1]); ctx.stroke(); };
  for (const a of [-1, 1]) for (const b of [-1, 1]) { L([-1, a, b], [1, a, b], col("--eje")); L([a, -1, b], [a, 1, b], col("--eje")); L([a, b, -1], [a, b, 1], col("--eje")); }
  L([-1, 0, 0], [1, 0, 0], col("--tenue"), 1.5); L([0, -1, 0], [0, 1, 0], col("--tenue"), 1.5); L([0, 0, -1], [0, 0, 1], col("--tenue"), 1.5);
  ctx.font = "12px system-ui, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  for (const [p, t, c] of [[[1.32, 0, 0], "Valencia +", "--val"], [[0, 1.2, 0], "Arousal +", "--aro"], [[0, 0, 1.34], "Dominancia +", "--dom"], [[-1.2, 0, 0], "−", "--tenue"], [[0, -1.18, 0], "−", "--tenue"], [[0, 0, -1.2], "−", "--tenue"]]) { const [x, y] = P(...p); ctx.fillStyle = col(c); ctx.fillText(t, x, y); }
  // Punto actual: relleno con el color de su emocion. Anterior: circulo negro vacio.
  const cNegro = col("--normal"), color = cNegro;
  // Solo la frase anterior y la actual, no todo el registro.
  const turnos = datos.turnos.slice(-2);
  const pts = turnos.map((t) => quien === "romeo" ? (t.romeo || datos.base) : (t.fusion || { v: 0, a: 0, d: 0 }));
  const n = pts.length;
  ctx.globalAlpha = 0.3; pts.forEach((p, i) => { if (!p) return; const [x, y] = P(p.v, -1, p.d); ctx.beginPath(); ctx.fillStyle = col("--tenue"); rombo(x, y, 3); ctx.fill(); }); ctx.globalAlpha = 1;
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
  }
  if (quien === "usuario") {
    const [x, y] = P(0, 0, 0), cn = col("--normal");
    ctx.beginPath(); ctx.fillStyle = cn; ctx.moveTo(x, y - 7); ctx.lineTo(x + 7, y); ctx.lineTo(x, y + 7); ctx.lineTo(x - 7, y); ctx.fill();
  }
  const marcas = pts.map((p, i) => { if (!p) return null; const [x, y, z] = P(p.v, p.a, p.d); return { x, y, z, ultimo: i === n - 1, clave: emocionDe(p, centro).clave }; }).filter(Boolean);
  if (quien === "romeo" && datos.base) { const [x, y, z] = P(datos.base.v, datos.base.a, datos.base.d); marcas.push({ x, y, z, base: true }); }
  marcas.sort((a, b) => b.z - a.z);
  for (const m of marcas) {
    ctx.beginPath();
    if (m.base) {   // Normal de Romeo = su animo base: rombo negro, como las demas emociones
      ctx.fillStyle = col("--normal"); ctx.moveTo(m.x, m.y - 7); ctx.lineTo(m.x + 7, m.y); ctx.lineTo(m.x, m.y + 7); ctx.lineTo(m.x - 7, m.y); ctx.fill();
      continue;
    }
    if (m.ultimo) {
      // Romeo naranja, la persona morada; del tamaño de los rombos.
      const ce = col(quien === "romeo" ? "--punto-romeo" : "--punto-usuario");
      rombo(m.x, m.y, 7); ctx.fillStyle = ce; ctx.fill();
      ctx.fillStyle = ce;
    } else {
      rombo(m.x, m.y, 7); ctx.fillStyle = col("--tarjeta"); ctx.fill();
      ctx.strokeStyle = cNegro; ctx.lineWidth = 2.5; ctx.stroke();
      ctx.fillStyle = cNegro;
    }
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
    $("lecturas").innerHTML = ult.lecturas.map((l) => `<div class="medidor"><b>${esc(l.fuente.split(" ")[0])}</b><div class="pista"><i style="width:${Math.round(l.confianza * 100)}%"></i></div><span>${esc(EMO[l.emocion] || l.emocion)}</span><span class="n">${Math.round(l.confianza * 100)}%</span></div>`).join("")
      + (ult.sin_cara ? '<div class="medidor"><b>Cara</b><div class="pista"></div><span class="vacio">no vio cara</span><span class="n">—</span></div>' : "");
  }
  if (d.ultima_respuesta) { $("respuesta").className = "frase"; $("respuesta").textContent = d.ultima_respuesta; }
  for (const quien of ["romeo", "usuario"]) {
    const t = d.turnos[d.turnos.length - 1];
    const p = !t ? (quien === "romeo" ? d.base : null) : (quien === "romeo" ? (t.romeo || d.base) : t.fusion);
    const e = emocionDe(p, centroDe(quien));
    const el = $("emo-" + quien);
    el.textContent = (quien === "romeo" ? "Romeo está: " : "Tú estás: ") + textoEmo(e);
    el.style.color = `var(${COLOR_EMO[e.clave]})`;
    $("vad-" + quien).innerHTML = [["Valencia", "v"], ["Arousal", "a"], ["Dominancia", "d"]].map(([n, k]) => `<div><small>${n}</small><strong>${p ? f2(p[k]) : "0.00"}</strong></div>`).join("");
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
<link rel="stylesheet" media="print" onload="this.media='all'" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Manrope:wght@400;600;700;800&display=swap">
<style>/*TEMA*/</style></head><body>
<header class="top"><div class="top-in"><p class="marca"><span class="rombo" style="background:var(--romeo)"></span>ROMEO</p><nav class="nav"><a href="./">En vivo</a><a href="calculo">Cálculo VAD</a><a href="carrito" class="aqui">Carrito</a></nav></div></header>
<main>
  <div>
  <h1>Carrito</h1>
  <p class="sub">Mueve el carrito de Romeo. Funciona con Romeo encendido: lo sirve su nodo del carrito.</p>
  </div>

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
<link rel="stylesheet" media="print" onload="this.media='all'" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Manrope:wght@400;600;700;800&display=swap">
<style>/*TEMA*/</style></head><body>
<header class="top"><div class="top-in"><p class="marca"><span class="rombo" style="background:var(--romeo)"></span>ROMEO</p><nav class="nav"><a href="./">En vivo</a><a href="calculo" class="aqui">Cálculo VAD</a><a href="carrito">Carrito</a></nav></div></header>
<main>
  <div>
  <h1>Cálculo de valencia y arousal</h1>
  <p class="sub">Cada paso con los números reales del turno, tal como los calcula el robot (PULSE → ROSE → AEB, metodología de Rocha).</p>
  </div>
  <div class="barra-sup">
    <select id="turno" aria-label="Frase"></select>
    <label><input type="checkbox" id="seguir" checked> Seguir la última frase</label>
    <span class="que" id="info" style="margin:0"></span>
  </div>
  <div id="pasos" style="display:grid;gap:18px"><p class="vacio">Esperando la primera frase. Háblale a Romeo.</p></div>
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

// Los estados posibles de Romeo, con los rangos reales de |Δ| para su animo
// base. Misma cuenta que aeb.nivel(): radio neutro y cuatro tramos iguales.
// Cara, movimiento y tonos resumen aeb.py (_MOVIMIENTOS, _TONOS, _TONOS_TRIVIA).
const ESTADOS = [
  { clave: "felicidad", nombre: "Feliz", color: "--feliz", ancla: [0.76, 0.48, 0.35], cara: "Feliz",
    mov: ["Avanza y vuelve", "Avanza y vuelve", "Avanza y vuelve", "Avanza y vuelve"],
    chat: ["Se alegra contigo y celebra lo que cuentas", "Comparte tu alegría con entusiasmo y te pide que cuentes más"],
    trivia: ["Te felicita con alegría; si fallas, buen humor", "Te premia con mucho entusiasmo; si fallas, te motiva"] },
  { clave: "tristeza", nombre: "Triste", color: "--triste", ancla: [-0.63, -0.27, -0.33], cara: "Normal (no pone cara triste)",
    mov: ["—", "—", "—", "—"],
    chat: ["Entiende cómo te sientes y te da ánimo", "Te acompaña y te ayuda a sentirte un poco mejor"],
    trivia: ["Si fallas, «la próxima lo logras»; si aciertas, destaca el logro", "Igual, más suave y breve"] },
  { clave: "enojo", nombre: "Enojado", color: "--enojo", ancla: [-0.51, 0.59, 0.25], cara: "Enojado",
    mov: ["Retrocede y vuelve", "Retrocede y vuelve", "Retrocede y vuelve", "Retrocede y vuelve"],
    chat: ["Se pone de tu lado, sin regañarte", "Te da la razón en lo que sientes y te escucha, sin pedirte calma"],
    trivia: ["Con calma: «la próxima te sale»", "Muy calmado y con pocas palabras"] },
];

function tablaEstados(a) {
  const b = a.base, f = a.fraccion_neutral;
  const datos = ESTADOS.map((e) => { const d = Math.hypot(e.ancla[0] - b[0], e.ancla[1] - b[1], e.ancla[2] - b[2]); return { ...e, radio: f * d, tramo: (d - f * d) / 4 }; });
  const radios = datos.map((e) => e.radio);
  const fila = (clase, celdas) => `<tr${clase ? ' class="actual"' : ""}>${celdas.map((c, i) => `<td${i === 2 ? ' class="n"' : ""}>${c}</td>`).join("")}</tr>`;
  const marca = (si) => si ? ' <span class="chip">ahora</span>' : "";
  let filas = fila(a.nivel === 0, [`<i class="rombo" style="background:var(--normal)"></i> Normal${marca(a.nivel === 0)}`, "L0",
    `&lt; ${Math.min(...radios).toFixed(3)} a ${Math.max(...radios).toFixed(3)}`, "Normal", "—", "Sin tono: responde como siempre", "Comenta como siempre"]);
  for (const e of datos) for (let n = 1; n <= 4; n++) {
    const desde = e.radio + (n - 1) * e.tramo, hasta = e.radio + n * e.tramo, si = a.estado === `${e.clave}_L${n}`;
    filas += fila(si, [`<i class="rombo" style="background:var(${e.color})"></i> ${e.nombre}${marca(si)}`, `L${n}`,
      `${desde.toFixed(3)} – ${n === 4 ? "más" : hasta.toFixed(3)}`, e.cara, e.mov[n - 1], e.chat[n <= 2 ? 0 : 1], e.trivia[n <= 2 ? 0 : 1]]);
  }
  return `<section class="paso"><h2><span class="n">Estados</span>Niveles L1 a L4 y qué hace Romeo</h2>
    <p class="que">El nivel sale de cuánto se apartó Romeo de su ánimo base (|Δ|). Cada emoción tiene sus propios rangos, porque cada una está a distinta distancia de la base. La fila marcada es el estado de esta frase.</p>
    <div class="tabla"><table><thead><tr><th>Estado</th><th>Nivel</th><th class="n">|Δ|</th><th>Cara</th><th>Movimiento</th><th>En la conversación</th><th>En la trivia</th></tr></thead>
    <tbody>${filas}</tbody></table></div></section>`;
}

const flecha = '<div class="flecha"></div>';
function vecbar(v) {
  if (!v) return "";
  const c = ["cv", "ca", "cd"], k = ["V", "A", "D"];
  return `<div class="vecbar">${v.map((x, i) => `<div class="comp ${c[i]}"><small>${k[i]}</small><b>${n2(x)}</b></div>`).join("")}</div>`;
}
// Una columna de fuente del diagrama: la caja del clasificador + su PULSE.
function fuenteFlow(l) {
  const [nombre, color] = FUENTE[l.fuente] || [l.fuente, "--suave"];
  const chips = l.filas.filter((f) => f.p > 0.001)
    .map((f) => `<span class="chip-emo">${esc(f.emocion)} ${f.p.toFixed(2)}</span>`).join("");
  return `<div class="fuente-col">
    <div class="caja c-pulse"><p class="ct"><span class="punto" style="background:var(${color})"></span>${esc(nombre)}</p>
      <div class="chips-emo">${chips}</div></div>
    ${flecha}
    <div class="caja c-pulse"><p class="cd"><span class="ecref">c = 1 − H(p)</span> = <b>${l.confianza.toFixed(3)}</b></p>
      <p class="cd"><span class="ecref">s = Σ pᵢ·vᵢ</span></p>${vecbar(l.vad)}
      <p class="cd">Dominante: <b>${esc(l.dominante)}</b></p></div>
  </div>`;
}
function fuenteSinCara(t) {
  if (t.fer_sin_cara == null || t.lecturas.some((l) => l.fuente === "FER")) return "";
  return `<div class="fuente-col"><div class="caja c-pulse"><p class="ct"><span class="punto" style="background:var(--fer)"></span>Cara (FER)</p>
    <p class="cd">Sin cara en los ${t.fer_sin_cara} s de la frase (0 cuadros). No entra en la fusión.</p></div></div>`;
}

function render(t) {
  const p = t.perfil, b = p.base;
  const d0 = t.decaer.antes, d1 = t.decaer.despues, movio = d0.some((x, i) => Math.abs(x - d1[i]) > 1e-4);
  const fu = t.fusion, em = t.empatia, a = t.aeb;
  let h = "";

  // Entrada
  h += `<div class="caja c-entrada"><p class="ct">El usuario habla → «${esc(t.texto)}»</p>
    <p class="cd">Silero VAD corta la frase · ventana [t_inicio, t_fin]</p></div>${flecha}`;

  // ROSE · decaimiento (paso previo)
  h += `<div class="grupo g-rose"><span class="tag">Paso previo · poner al día el ánimo (ROSE)</span>
    <div class="caja c-rose"><p class="ct">Decaimiento por el tiempo en silencio <span class="ecref">(ec. 4.6)</span></p>
      <p class="cd mono">s_r ← (1 − d)·s_r + d·b &nbsp; d = ${p.decaimiento}</p>
      <p class="cd">${v3(d0)} → ${v3(d1)}${movio ? "" : " · dentro de la latencia, no decae"}</p></div></div>${flecha}`;

  // PULSE por fuente (en paralelo)
  const cols = t.lecturas.map(fuenteFlow).join("") + fuenteSinCara(t);
  h += `<div class="grupo g-pulse"><span class="tag">PULSE · por fuente</span>
    ${cols ? `<div class="fuentes-flow">${cols}</div>` : '<p class="vacio">Ninguna fuente dio lectura en esta frase.</p>'}</div>${flecha}`;

  // PULSE · fusión (con rombo de neutral)
  let fusionBody;
  if (!fu) fusionBody = '<p class="vacio">Sin lecturas: no hay nada que fusionar.</p>';
  else if (fu.todos_neutral) fusionBody = `<div class="rombo-dec"><div class="rombo-caja">¿todas las fuentes dicen neutral?</div>
      <div class="rombo-resp">sí → <b>el ánimo no cambia</b></div></div>`;
  else {
    const filas = fu.entradas.map((e) => `<tr><td>${esc((FUENTE[e.fuente] || [e.fuente])[0])}</td><td class="n">${e.confianza.toFixed(3)}</td>
        <td class="n">${v3(e.vad)}</td></tr>`).join("");
    fusionBody = `<div class="rombo-dec"><div class="rombo-caja">¿todas las fuentes dicen neutral?</div><div class="rombo-resp">no ↓</div></div>${flecha}
      <div class="caja c-pulse"><p class="ct">Fusión ponderada <span class="ecref">(ec. 4.4)</span></p>
      <p class="cd mono">S_u = Σ c·S / Σ c</p>
      <div class="tabla flow-tabla"><table><thead><tr><th>Fuente</th><th class="n">c</th><th class="n">S (v,a,d)</th></tr></thead><tbody>${filas}</tbody></table></div>
      ${vecbar(fu.s_u)}<p class="cd">→ estado del usuario S_u</p></div>`;
  }
  h += `<div class="grupo g-pulse"><span class="tag">PULSE · fusión</span>${fusionBody}</div>${flecha}`;

  // ROSE · empatía
  h += `<div class="grupo g-rose"><span class="tag">ROSE · reacción (empatía)</span>
    <div class="caja c-rose"><p class="ct">Transformación empática <span class="ecref">(ec. 4.5)</span></p>
    ${em ? `<p class="cd mono">s_r ← e·S_u + (1 − e)·s_r &nbsp; e = ${em.e}</p>${vecbar(em.despues)}
      <p class="cd">${v3(em.antes)} → <b>${v3(em.despues)}</b> · ánimo de Romeo</p>`
    : '<p class="vacio">No se aplicó: no hubo emoción que contagiar.</p>'}</div></div>${flecha}`;

  // AEB
  const cosOrden = Object.entries(a.cosenos || {}).sort((x, y) => y[1] - x[1]);
  const cosChips = cosOrden.map(([e, c], i) => `<span class="chip-emo">${esc(e)} ${c.toFixed(2)}${i === 0 ? " ←" : ""}</span>`).join("");
  h += `<div class="grupo g-aeb"><span class="tag">AEB</span>
    <div class="caja c-aeb"><p class="ct">Del ánimo al comportamiento</p>
      <p class="cd mono">Δ = s_r − b = ${v3(a.desviacion)} · |Δ| = ${a.magnitud.toFixed(3)}</p>
      ${cosChips ? `<p class="cd">dirección (coseno): <span class="chips-emo">${cosChips}</span></p>` : ""}
      ${a.emocion ? `<p class="cd mono">${a.magnitud < a.radio
        ? `|Δ| ${a.magnitud.toFixed(3)} < radio ${a.radio.toFixed(3)} → L0`
        : `nivel = 1 + ⌊4·(|Δ|−r)/(‖a−b‖−r)⌋ = ${a.nivel}`}</p>` : ""}
      <p class="resultado">→ <span class="chip">${esc(a.estado.replace("_L", " · nivel "))}</span> · cara ${esc(a.cara)}${a.emocion === "tristeza" && a.cara === "content" ? " (con tristeza, cara normal)" : ""}</p>
    </div>${flecha}
    <div class="salidas-flow">
      <div class="caja c-pulse"><p class="ct">Cara</p><p class="cd">${esc(a.cara)}</p></div>
      <div class="caja c-pulse"><p class="ct">Tono del LLM</p><p class="cd">cómo responde</p></div>
      <div class="caja c-pulse"><p class="ct">Movimiento</p><p class="cd">carrito según nivel</p></div>
    </div></div>`;

  // Detalle completo (tablas de PULSE por modalidad + niveles) en un desplegable
  h += `<details class="ver-detalle"><summary>Ver el cálculo completo (probabilidades, entropía, tabla de niveles)</summary>
    <div style="margin-top:12px">
    <section class="paso"><h2><span class="n">PULSE</span>Por modalidad, con probabilidades</h2>
      ${t.lecturas.length || t.fer_sin_cara != null ? `<div class="modalidades">${t.lecturas.map(modalidad).join("")}${sinCara(t)}</div>` : '<p class="vacio">—</p>'}</section>
    ${tablaEstados(a)}</div></details>`;

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


TEMA = r"""
/* Consola del robot: barra de tinta arriba, modulos blancos, numeros en mono. */
:root { --fondo:#eceff4; --tarjeta:#ffffff; --texto:#12162b; --suave:#555c74; --tenue:#8a90a3;
  --borde:#d9dde7; --eje:#c4c9d6; --barra:#e4e7ef; --tinta:#171b34; --tinta-texto:#f2f3f9; --tinta-suave:#a9afc9;
  --val:#2a78d6; --aro:#eb6834; --dom:#1baf7a; --ok:#35d39a; --ser:#eb6834; --ter:#1baf7a; --fer:#2a78d6;
  --romeo:#e8501c; --fusion:#5b21b6;
  --enojo:#d1352b; --triste:#2a78d6; --feliz:#d4a400; --normal:#0b0b0b; --punto-romeo:#e8501c; --punto-usuario:#5b21b6;
  --ui:"Manrope",system-ui,-apple-system,"Segoe UI",sans-serif; --num:"IBM Plex Mono",ui-monospace,Consolas,monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --fondo:#0f1120; --tarjeta:#1a1d31; --texto:#eef0f8; --suave:#b4b9cf; --tenue:#868ca6;
  --borde:#2d3149; --eje:#3a3f5a; --barra:#2d3149; --tinta:#0a0c18; --tinta-texto:#f2f3f9; --tinta-suave:#a9afc9;
  --val:#4d94ec; --aro:#f07a45; --dom:#2cc08c; --ser:#f07a45; --ter:#2cc08c; --fer:#4d94ec;
  --romeo:#f2652e; --fusion:#8b5cf6;
  --enojo:#ef5a4f; --triste:#4d94ec; --feliz:#f2c94c; --normal:#f0efec; --punto-romeo:#f2652e; --punto-usuario:#8b5cf6; color-scheme:dark; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--fondo); color:var(--texto); font:16px/1.5 var(--ui); }
a { color:inherit; }
.top { background:var(--tinta); color:var(--tinta-texto); }
.top-in { max-width:1120px; margin:0 auto; padding:0 20px; display:flex; flex-wrap:wrap; align-items:center; gap:4px 28px; min-height:60px; }
.marca { font-weight:800; letter-spacing:.2em; font-size:1rem; display:inline-flex; align-items:center; gap:10px; margin:0; }
.nav { display:flex; flex-wrap:wrap; gap:4px; flex:1 1 auto; }
.nav a { color:var(--tinta-suave); text-decoration:none; font-size:.9rem; font-weight:600; padding:18px 12px 16px; border-bottom:3px solid transparent; }
.nav a:hover { color:var(--tinta-texto); }
.nav a.aqui { color:var(--tinta-texto); border-color:var(--romeo); }
.nav a:focus-visible, button:focus-visible, select:focus-visible { outline:2px solid var(--romeo); outline-offset:2px; }
.vivo { display:inline-flex; align-items:center; gap:8px; font-size:.8rem; color:var(--tinta-suave); overflow-wrap:anywhere; }
.vivo i { width:8px; height:8px; border-radius:50%; background:var(--ok); flex:none; }
main { max-width:1120px; margin:0 auto; padding:24px 20px 56px; display:grid; gap:18px; }
main > * { min-width:0; }
h1 { font-size:1.6rem; font-weight:800; margin:0; letter-spacing:-.01em; text-wrap:balance; }
.sub { color:var(--suave); margin:4px 0 0; max-width:70ch; }
.tarjeta, .paso { background:var(--tarjeta); border:1px solid var(--borde); border-radius:6px; padding:20px; min-width:0; }
.etiqueta, .tarjeta > h2 { font-size:.72rem; text-transform:uppercase; letter-spacing:.1em; color:var(--suave); margin:0 0 10px; font-weight:700; }
.dos { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:18px; }
@media (max-width:720px) { .dos { grid-template-columns:minmax(0,1fr); } }
.frase { font-size:1.3rem; font-weight:600; line-height:1.35; margin:0; }
.vacio { color:var(--suave); font-style:italic; font-weight:400; }
.nota { font-size:.82rem; color:var(--suave); margin:10px 0 0; }
.punto { width:10px; height:10px; border-radius:50%; display:inline-block; }
.rombo { width:10px; height:10px; display:inline-block; transform:rotate(45deg); margin:0 3px; flex:none; }
/* Medidores de voz, texto y cara */
.medidores { display:grid; gap:8px; margin-top:14px; }
.medidor { display:grid; grid-template-columns:52px minmax(0,1fr) 92px 44px; gap:10px; align-items:center; font-size:.85rem; }
.medidor b { font-weight:700; }
.pista { height:8px; background:var(--barra); border-radius:2px; overflow:hidden; }
.pista i { display:block; height:100%; background:var(--texto); }
.medidor .n { font-family:var(--num); text-align:right; font-size:.8rem; }
/* Modulos de Romeo y de la persona */
.modulo { background:var(--tarjeta); border:1px solid var(--borde); border-radius:6px; overflow:hidden; min-width:0; }
.modulo header { color:#fff; padding:9px 16px; font-size:.78rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
.modulo header sub { line-height:0; }
.modulo.romeo header { background:var(--punto-romeo); }
.modulo.usuario header { background:var(--punto-usuario); }
.modulo-in { padding:16px; }
.emo { font-size:1.75rem; font-weight:800; line-height:1.15; margin:0; }
.emo span { font-family:var(--ui); }
.vad { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); border:1px solid var(--borde); border-radius:4px; margin:14px 0 6px; }
.vad div { padding:8px 10px; border-left:1px solid var(--borde); min-width:0; }
.vad div:first-child { border-left:0; }
.vad small { display:block; font-size:.68rem; color:var(--suave); text-transform:uppercase; letter-spacing:.06em; font-weight:700; }
.vad strong { font-family:var(--num); font-size:1.05rem; font-weight:600; }
.cubo canvas { width:100%; display:block; touch-action:none; cursor:grab; }
.leyenda { display:flex; flex-wrap:wrap; justify-content:center; gap:6px 18px; font-size:.85rem; font-weight:600; }
.leyenda span { display:inline-flex; align-items:center; gap:7px; }
details { font-size:.85rem; color:var(--suave); margin-top:10px; }
summary { cursor:pointer; font-weight:700; text-align:center; }
.chip { display:inline-block; border:1px solid var(--borde); border-radius:3px; padding:2px 10px; font-size:.85rem; font-weight:700; }
.tabla { overflow-x:auto; }
table { width:100%; border-collapse:collapse; font-size:.86rem; }
th { text-align:left; color:var(--suave); font-size:.68rem; text-transform:uppercase; letter-spacing:.07em; padding:7px 8px; border-bottom:2px solid var(--borde); }
td { padding:8px; border-bottom:1px solid var(--borde); vertical-align:top; }
td.n, th.n { text-align:right; white-space:nowrap; font-family:var(--num); font-size:.8rem; }
tr.total td { font-weight:700; border-top:2px solid var(--borde); }
tr.actual td { background:var(--barra); font-weight:700; }
td .rombo { width:9px; height:9px; margin-right:6px; }
/* Carrito */
.carrito { display:flex; flex-direction:column; gap:20px; align-items:center; text-align:center; }
.carrito > div:first-child { width:min(100%,420px); }
.pad { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; width:min(100%,420px); margin:0 auto; }
.pad button, .giro button { font:inherit; font-size:2.2rem; aspect-ratio:1; border:0; border-radius:6px; background:var(--tinta); color:var(--tinta-texto); cursor:pointer; touch-action:none; user-select:none; }
.pad button.activo, .giro button.activo { background:var(--romeo); color:#fff; }
.pad button.stop { background:var(--enojo); color:#fff; font-size:1.3rem; font-weight:800; letter-spacing:.06em; }
.giro { display:grid; grid-template-columns:1fr 1fr; gap:10px; width:min(100%,420px); margin:10px auto 0; }
.giro button { aspect-ratio:auto; padding:20px 0; font-size:1.2rem; font-weight:700; }
.ayuda-carrito { max-width:420px; min-width:0; }
/* Calculo */
.barra-sup { display:flex; flex-wrap:wrap; gap:10px 16px; align-items:center; }
select, label { font:inherit; font-size:.9rem; }
select { padding:7px 9px; border-radius:4px; border:1px solid var(--borde); background:var(--tarjeta); color:var(--texto); max-width:100%; }
.paso h2 { font-size:1.08rem; font-weight:800; margin:0 0 4px; display:flex; flex-wrap:wrap; gap:10px; align-items:baseline; }
.paso h2 .n { font-family:var(--num); font-size:.72rem; background:var(--tinta); color:var(--tinta-texto); border-radius:3px; padding:2px 8px; font-weight:600; }
.que { color:var(--suave); font-size:.9rem; margin:0 0 12px; }
.formula { font-family:var(--num); font-size:.86rem; background:var(--tinta); color:var(--tinta-texto); border-radius:4px; padding:10px 14px; margin:10px 0; overflow-x:auto; white-space:pre-wrap; overflow-wrap:anywhere; }
.barrita { height:8px; background:var(--barra); border-radius:2px; overflow:hidden; min-width:60px; }
.barrita i { display:block; height:100%; }
.modalidades { display:grid; grid-template-columns:minmax(0,1fr); gap:14px; }
.modalidad { border:1px solid var(--borde); border-radius:4px; padding:14px; min-width:0; }
.modalidad h3 { margin:0 0 6px; font-size:.95rem; display:flex; gap:8px; align-items:center; }
.resultado { font-weight:700; margin:10px 0 0; }
@media (max-width:560px) { .medidor { grid-template-columns:44px minmax(0,1fr) 72px 40px; } .emo { font-size:1.45rem; } }
/* Diagrama de flujo del calculo VAD: cajas por etapa, flechas y fuentes en paralelo. */
#pasos { display:flex; flex-direction:column; align-items:center; gap:0; }
.grupo { width:100%; border:2px dashed var(--eje); border-radius:14px; padding:30px 16px 18px; margin:2px 0; position:relative; display:flex; flex-direction:column; align-items:center; gap:12px; }
.grupo > .tag { position:absolute; top:9px; left:16px; font-weight:800; font-size:.82rem; letter-spacing:.02em; }
.grupo.g-rose { border-color:var(--punto-romeo); } .grupo.g-rose > .tag { color:var(--punto-romeo); }
.grupo.g-pulse { border-color:var(--fusion); } .grupo.g-pulse > .tag { color:var(--fusion); }
.grupo.g-aeb { border-color:var(--aro); } .grupo.g-aeb > .tag { color:var(--aro); }
.caja { width:100%; max-width:600px; border:2px solid var(--borde); border-radius:12px; background:var(--tarjeta); padding:14px 16px; }
.caja .ct { font-weight:800; font-size:1rem; margin:0 0 6px; display:flex; flex-wrap:wrap; gap:8px; align-items:baseline; }
.caja .cd { color:var(--suave); font-size:.86rem; margin:4px 0; }
.caja .cd.mono, .caja .mono { font-family:var(--num); }
.caja .cd b { color:var(--texto); }
.caja.c-rose { border-color:var(--punto-romeo); }
.caja.c-pulse { border-color:var(--fusion); }
.caja.c-aeb { border-color:var(--aro); }
.caja.c-entrada { border-style:dashed; }
.flecha { width:2px; height:24px; background:var(--tenue); position:relative; flex:none; }
.flecha::after { content:""; position:absolute; bottom:-1px; left:50%; transform:translateX(-50%); border-left:5px solid transparent; border-right:5px solid transparent; border-top:7px solid var(--tenue); }
.fuentes-flow { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; width:100%; }
@media (max-width:620px) { .fuentes-flow { grid-template-columns:minmax(0,1fr); } }
.fuente-col { display:flex; flex-direction:column; gap:8px; min-width:0; }
.fuente-col .caja { max-width:none; height:100%; }
.chips-emo { display:flex; flex-wrap:wrap; gap:6px; margin-top:6px; }
.chip-emo { font-size:.78rem; border:1px solid var(--borde); border-radius:999px; padding:2px 9px; font-family:var(--num); }
.rombo-dec { display:flex; flex-wrap:wrap; gap:12px; align-items:center; justify-content:center; width:100%; }
.rombo-caja { background:var(--tarjeta); border:2px solid var(--fusion); width:280px; max-width:85vw; aspect-ratio:2.4/1; display:flex; align-items:center; justify-content:center; text-align:center; clip-path:polygon(50% 0,100% 50%,50% 100%,0 50%); padding:0 2rem; font-weight:700; font-size:.86rem; color:var(--texto); }
.rombo-resp { font-size:.82rem; color:var(--suave); }
.rombo-resp b { color:var(--romeo); }
.vecbar { display:flex; gap:8px; flex-wrap:wrap; margin:8px 0 2px; }
.vecbar .comp { border:1px solid var(--borde); border-radius:8px; padding:5px 10px; text-align:center; min-width:66px; background:var(--fondo); }
.vecbar .comp small { display:block; font-size:.62rem; text-transform:uppercase; letter-spacing:.06em; color:var(--suave); }
.vecbar .comp b { font-family:var(--num); font-size:1.05rem; }
.vecbar .comp.cv b { color:var(--val); } .vecbar .comp.ca b { color:var(--aro); } .vecbar .comp.cd b { color:var(--dom); }
.ecref { font-size:.7rem; color:var(--suave); font-weight:600; font-family:var(--num); }
.salidas-flow { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; width:100%; }
@media (max-width:620px) { .salidas-flow { grid-template-columns:minmax(0,1fr); } }
.salidas-flow .caja { max-width:none; text-align:center; }
.flow-tabla { width:100%; }
details.ver-detalle { width:100%; max-width:600px; margin-top:4px; }
details.ver-detalle summary { text-align:left; font-size:.82rem; color:var(--suave); }
"""
for _n in ("PAGINA", "PAGINA_CARRITO", "CALCULO"):
    globals()[_n] = globals()[_n].replace("/*TEMA*/", TEMA)


if __name__ == "__main__":
    print(f"Panel de Romeo en http://0.0.0.0:{PUERTO}/ leyendo {LOG}")
    app.run(host="0.0.0.0", port=PUERTO, threaded=True)
