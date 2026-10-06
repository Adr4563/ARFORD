"""Página web para mover el carrito a mano desde el celular o la PC.

La sirve moving_control_node (Flask, en un hilo aparte) y cada botón manda
el comando por el MISMO CartSerial que usa Lora, así que no hace falta el
WiFi del ESP32 ni tocar su firmware.

Seguridad: el firmware frena solo si no recibe un comando en 500 ms. La
página, mientras se mantiene apretado un botón, repite el comando cada
150 ms; al soltar manda "S". Si se cierra la página o se corta el WiFi a
mitad de un movimiento, el carrito se detiene solo por ese watchdog.
"""
import logging
import threading

MOVIMIENTOS = {"F", "B", "SL", "SR", "RL", "RR", "FL", "FR", "BL", "BR", "S"}

_PAGINA = """<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<title>Carrito de Lora</title>
<style>
  :root { --fondo:#f4f5f7; --tarjeta:#fff; --texto:#1c1e21; --boton:#2f6fec; --stop:#d93025; --borde:#d0d4da; }
  @media (prefers-color-scheme: dark) { :root { --fondo:#15171a; --tarjeta:#202328; --texto:#e8eaed; --boton:#4c8bf5; --stop:#f0584c; --borde:#3a3f46; } }
  * { box-sizing:border-box; -webkit-user-select:none; user-select:none; -webkit-touch-callout:none; }
  body { margin:0; font-family:system-ui,sans-serif; background:var(--fondo); color:var(--texto);
         display:flex; flex-direction:column; align-items:center; padding:16px; }
  h1 { font-size:1.3rem; margin:8px 0 4px; }
  #estado { font-size:.95rem; min-height:1.4em; margin-bottom:12px; opacity:.8; }
  .pad { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; width:min(92vw,360px); }
  button { aspect-ratio:1; border:1px solid var(--borde); border-radius:16px; background:var(--tarjeta);
           color:var(--texto); font-size:1.6rem; touch-action:none; cursor:pointer; }
  button.activo { background:var(--boton); color:#fff; border-color:var(--boton); }
  button.stop { background:var(--stop); color:#fff; border-color:var(--stop); font-size:1.1rem; font-weight:700; }
  .giro { display:grid; grid-template-columns:1fr 1fr; gap:10px; width:min(92vw,360px); margin-top:10px; }
  .giro button { aspect-ratio:auto; padding:18px 0; font-size:1.1rem; }
  p.ayuda { font-size:.85rem; opacity:.7; max-width:360px; text-align:center; }
</style></head><body>
<h1>Carrito de Lora</h1>
<div id="estado">Mantén pulsado un botón para mover</div>
<div class="pad">
  <button data-c="FL" aria-label="Diagonal adelante izquierda">↖</button>
  <button data-c="F"  aria-label="Adelante">↑</button>
  <button data-c="FR" aria-label="Diagonal adelante derecha">↗</button>
  <button data-c="SL" aria-label="Lateral izquierda">←</button>
  <button class="stop" data-stop aria-label="Parar">STOP</button>
  <button data-c="SR" aria-label="Lateral derecha">→</button>
  <button data-c="BL" aria-label="Diagonal atrás izquierda">↙</button>
  <button data-c="B"  aria-label="Atrás">↓</button>
  <button data-c="BR" aria-label="Diagonal atrás derecha">↘</button>
</div>
<div class="giro">
  <button data-c="RL" aria-label="Girar a la izquierda">⟲ Girar</button>
  <button data-c="RR" aria-label="Girar a la derecha">Girar ⟳</button>
</div>
<p class="ayuda">Teclado: W/A/S/D para moverte, Q/E para girar, flechas también. Al soltar, el carrito se detiene.</p>
<script>
const estado = document.getElementById("estado");
let actual = null, timer = null, boton = null;
async function mandar(c) {
  try {
    const r = await fetch("/cmd/" + c, { method: "POST" });
    const j = await r.json();
    estado.textContent = j.ok ? (c === "S" ? "Detenido" : "Moviendo: " + c) : "El carrito no responde (¿cable USB?)";
  } catch (e) { estado.textContent = "Sin conexión con la Raspberry Pi"; }
}
function empezar(c, b) {
  if (actual === c) return;
  parar(false); actual = c; boton = b; if (b) b.classList.add("activo");
  mandar(c); timer = setInterval(() => mandar(c), 150);
}
function parar(enviar = true) {
  if (timer) clearInterval(timer); timer = null;
  if (boton) boton.classList.remove("activo"); boton = null;
  if (actual !== null || enviar) { actual = null; mandar("S"); }
}
document.querySelectorAll("button[data-c]").forEach(b => {
  b.addEventListener("pointerdown", e => { e.preventDefault(); b.setPointerCapture(e.pointerId); empezar(b.dataset.c, b); });
  ["pointerup", "pointercancel", "lostpointercapture"].forEach(ev => b.addEventListener(ev, () => parar()));
});
document.querySelector("[data-stop]").addEventListener("pointerdown", e => { e.preventDefault(); parar(); });
const teclas = { w:"F", arrowup:"F", s:"B", arrowdown:"B", a:"SL", arrowleft:"SL", d:"SR", arrowright:"SR", q:"RL", e:"RR" };
document.addEventListener("keydown", e => { const c = teclas[e.key.toLowerCase()]; if (c && !e.repeat) empezar(c, document.querySelector(`[data-c="${c}"]`)); });
document.addEventListener("keyup", e => { if (teclas[e.key.toLowerCase()]) parar(); });
window.addEventListener("blur", () => parar());
document.addEventListener("visibilitychange", () => { if (document.hidden) parar(); });
</script></body></html>"""


class ControlWeb:
    """Servidor Flask de la página de control. `ejecutar(comando)` es la
    función que manda el comando al carrito y devuelve True/False."""

    def __init__(self, ejecutar, puerto=8080, logger=None):
        self._ejecutar = ejecutar
        self.puerto = puerto
        self.logger = logger
        self._hilo = None

    def iniciar(self):
        if self._hilo is not None:
            return
        from flask import Flask, jsonify

        app = Flask(__name__)
        # Flask loguea cada petición: con el botón repitiendo cada 150 ms
        # llenaría el log de ROS.
        logging.getLogger("werkzeug").setLevel(logging.WARNING)

        @app.get("/")
        def pagina():
            return _PAGINA

        @app.post("/cmd/<comando>")
        def cmd(comando):
            comando = comando.upper()
            if comando not in MOVIMIENTOS:
                return jsonify(ok=False, error="comando desconocido"), 400
            return jsonify(ok=bool(self._ejecutar(comando)))

        self._hilo = threading.Thread(
            target=lambda: app.run(host="0.0.0.0", port=self.puerto, threaded=True),
            daemon=True)
        self._hilo.start()
        if self.logger is not None:
            self.logger.info(f"[control] página de control en http://0.0.0.0:{self.puerto}/")
