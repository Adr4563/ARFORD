"""Servidor de entrada/salida web -- portado de voz_server.py del proyecto
original, con un solo cambio estructural: en vez de empujar el texto
recibido a una `queue.Queue()` local que leía Orchestrator_Management.py en
el mismo proceso, acá `WebBridge` recibe un callback (`on_texto`) que
communication_node.py usa para publicar un `lora_interfaces/UserInput` en
/lora/user_input -- el resto (la página HTML, el flujo de SSE para la voz
de salida hacia el teléfono, por qué SSE y no WebSockets, por qué la
entrada de voz es Web Speech API del navegador y no algo local en la Pi)
es EXACTAMENTE igual al original, mismo HTML/JS, mismos endpoints.

Sigue corriendo Flask con `threaded=True` en un hilo de fondo -- ahora es
un hilo DENTRO de communication_node (arrancado en on_activate()), no del
proceso de Orchestrator_Management.py, pero el patrón (no bloquear al
resto del nodo) es el mismo.
"""

import json
import queue
import threading

from flask import Flask, Response, jsonify, request

# ── HTML/JS de la página -- copiado tal cual de voz_server.py original ──
_PAGINA = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lora — texto</title>
<style>
  body { font-family: system-ui, sans-serif; text-align: center; margin-top: 3rem; padding: 0 1rem; }
  #estado { color: #666; min-height: 1.4em; }
  #transcripcion { font-style: italic; min-height: 1.4em; }
  form { margin-top: 1.5rem; display: flex; gap: 0.5rem; justify-content: center; }
  input[type=text] { font-size: 1rem; padding: 0.6rem; flex: 1; max-width: 20rem; }
  button[type=submit] { font-size: 1rem; padding: 0.6rem 1.2rem; }
  .fila-botones-circulares { display: flex; flex-direction: column; align-items: center; gap: 0.4rem; margin-top: 1.5rem; }
  .btn-circular {
    width: 4rem; height: 4rem; border-radius: 50%;
    border: none; background: #2563eb; color: #fff;
    display: flex; align-items: center; justify-content: center;
    cursor: pointer; box-shadow: 0 2px 6px rgba(0,0,0,0.25);
    transition: background 0.15s, transform 0.1s;
  }
  .btn-circular:active { transform: scale(0.94); }
  .btn-circular:disabled { background: #aaa; cursor: not-allowed; }
  .btn-circular.escuchando { background: #dc2626; animation: pulso 1s infinite; }
  .btn-circular svg { width: 1.8rem; height: 1.8rem; fill: currentColor; }
  .etiqueta-boton { font-size: 0.85rem; color: #666; }
  @keyframes pulso {
    0%, 100% { box-shadow: 0 0 0 0 rgba(220,38,38,0.5); }
    50% { box-shadow: 0 0 0 10px rgba(220,38,38,0); }
  }
</style>
</head>
<body>
  <h1>Habla con Lora</h1>
  <p id="estado">Escribí tu mensaje.</p>
  <p id="transcripcion"></p>

  <form id="form-texto">
    <input type="text" id="texto" placeholder="Escribí acá..." autocomplete="off" autofocus>
    <button type="submit">Enviar</button>
  </form>

  <div class="fila-botones-circulares">
    <button id="btn-mic" class="btn-circular" type="button" aria-label="Hablar" title="Hablar">
      <svg viewBox="0 0 24 24"><path d="M12 14a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v5a3 3 0 0 0 3 3zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.92V21h2v-3.08A7 7 0 0 0 19 11h-2z"/></svg>
    </button>
    <span class="etiqueta-boton" id="etiqueta-mic">Hablar</span>
  </div>

  <div class="fila-botones-circulares" style="margin-top: 1.5rem;">
    <button id="btn-voz" class="btn-circular" type="button" aria-label="Activar voz de Lora" title="Activar voz de Lora">
      <svg viewBox="0 0 24 24"><path d="M4 9v6h4l5 5V4L8 9H4zm11.5 3a4.5 4.5 0 0 0-2.5-4.03v8.06A4.5 4.5 0 0 0 15.5 12z"/></svg>
    </button>
    <span class="etiqueta-boton">Activar voz de Lora</span>
  </div>
  <p id="estado-voz" style="color: #666;"></p>

<script>
  async function enviarTexto(texto) {
    texto = texto.trim();
    if (!texto) return;
    document.getElementById('estado').textContent = 'Enviando...';
    await fetch('/texto', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texto })
    });
    document.getElementById('transcripcion').textContent = '"' + texto + '"';
    document.getElementById('estado').textContent = 'Mandado a Lora.';
  }

  document.getElementById('form-texto').addEventListener('submit', async (e) => {
    e.preventDefault();
    const campo = document.getElementById('texto');
    await enviarTexto(campo.value);
    campo.value = '';
    campo.focus();
  });

  const ReconocedorVoz = window.SpeechRecognition || window.webkitSpeechRecognition;
  const btnMic = document.getElementById('btn-mic');
  const etiquetaMic = document.getElementById('etiqueta-mic');

  if (!ReconocedorVoz) {
    btnMic.disabled = true;
    etiquetaMic.textContent = 'No disponible en este navegador';
  } else {
    const reconocedor = new ReconocedorVoz();
    reconocedor.lang = 'es-AR';
    reconocedor.continuous = false;
    reconocedor.interimResults = false;

    let escuchando = false;

    reconocedor.onresult = (e) => {
      const texto = e.results[0][0].transcript;
      enviarTexto(texto);
    };
    reconocedor.onerror = (e) => {
      document.getElementById('estado').textContent =
        e.error === 'not-allowed'
          ? 'Sin permiso de micrófono (ver chrome://flags en el teléfono).'
          : 'No se pudo escuchar (' + e.error + ').';
    };
    reconocedor.onend = () => {
      escuchando = false;
      btnMic.classList.remove('escuchando');
      etiquetaMic.textContent = 'Hablar';
    };

    btnMic.addEventListener('click', () => {
      if (escuchando) return;
      escuchando = true;
      btnMic.classList.add('escuchando');
      etiquetaMic.textContent = 'Escuchando...';
      document.getElementById('estado').textContent = 'Hablá ahora...';
      reconocedor.start();
    });
  }

  const estadoVoz = document.getElementById('estado-voz');

  function elegirVoz() {
    const voces = speechSynthesis.getVoices();
    return voces.find(v => v.lang === 'es-AR')
        || voces.find(v => v.lang && v.lang.startsWith('es'))
        || null;
  }

  function hablar(texto) {
    const u = new SpeechSynthesisUtterance(texto);
    const voz = elegirVoz();
    if (voz) u.voice = voz;
    u.lang = 'es-AR';
    u.onstart = () => { estadoVoz.textContent = 'Lora está hablando...'; };
    u.onend = u.onerror = () => {
      estadoVoz.textContent = 'Escuchando a Lora...';
      fetch('/listo', { method: 'POST' }).catch(() => {});
    };
    speechSynthesis.speak(u);
  }

  document.getElementById('btn-voz').addEventListener('click', () => {
    speechSynthesis.speak(new SpeechSynthesisUtterance(''));
    estadoVoz.textContent = 'Escuchando a Lora...';
    document.getElementById('btn-voz').disabled = true;

    const fuente = new EventSource('/eventos');
    fuente.onmessage = (e) => {
      const datos = JSON.parse(e.data);
      hablar(datos.texto);
    };
    fuente.onerror = () => {
      estadoVoz.textContent = 'Se cortó la conexión con Lora, reintentando...';
    };
  });
</script>
</body>
</html>
"""


class WebBridge:
    """Encapsula el servidor Flask+SSE -- antes módulo con globals
    (`app`, `_entrada_queue`, `_clientes_voz`, etc.) en voz_server.py, acá
    una clase para que communication_node pueda tener su propia instancia
    con el callback de publicación ROS2 inyectado."""

    def __init__(self, port=8081, on_texto=None, logger=None):
        self.port = port
        # on_texto(texto): callback que communication_node pasa para
        # publicar UserInput -- reemplaza el `_entrada_queue.put(texto)`
        # directo del original.
        self.on_texto = on_texto
        self.logger = logger

        self.app = Flask(__name__)
        self._clientes_voz = set()
        self._clientes_voz_lock = threading.Lock()
        self._listo_evento = threading.Event()
        self._hilo = None
        self._registrar_rutas()

    def _log(self, msg):
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

    def hay_cliente_conectado(self):
        """Usado por communication_node para atender
        IsVoiceClientConnected.srv."""
        with self._clientes_voz_lock:
            return bool(self._clientes_voz)

    def hablar_telefono(self, texto, timeout=20):
        """Le manda `texto` a todos los teléfonos con la página abierta y
        con la voz activada, y bloquea hasta que alguno avise que terminó
        de hablarlo (o hasta `timeout` segundos). Devuelve False si no hay
        ningún teléfono conectado, o si nadie avisa a tiempo -- en los dos
        casos _voice_backends.Voz cae al motor de respaldo (edge-tts),
        igual que en el proyecto original."""
        with self._clientes_voz_lock:
            destinatarios = list(self._clientes_voz)
        if not destinatarios:
            return False
        self._listo_evento.clear()
        for cola in destinatarios:
            cola.put(texto)
        return self._listo_evento.wait(timeout)

    def _registrar_rutas(self):
        app = self.app

        @app.route("/")
        def index():
            return Response(_PAGINA, mimetype="text/html")

        @app.route("/texto", methods=["POST"])
        def texto_directo():
            body = request.get_json(silent=True) or {}
            texto = (body.get("texto") or "").strip()
            if texto and self.on_texto is not None:
                self.on_texto(texto)
            return jsonify({"ok": bool(texto)})

        @app.route("/eventos")
        def eventos():
            cola = queue.Queue()
            with self._clientes_voz_lock:
                self._clientes_voz.add(cola)

            def generar():
                try:
                    while True:
                        texto = cola.get()
                        yield f"data: {json.dumps({'texto': texto})}\n\n"
                finally:
                    with self._clientes_voz_lock:
                        self._clientes_voz.discard(cola)

            return Response(generar(), mimetype="text/event-stream")

        @app.route("/listo", methods=["POST"])
        def listo():
            self._listo_evento.set()
            return jsonify({"ok": True})

    def iniciar(self):
        """Levanta Flask en un hilo de fondo -- llamar desde on_activate()
        del LifecycleNode, no antes."""
        self._hilo = threading.Thread(
            target=lambda: self.app.run(host="0.0.0.0", port=self.port,
                                         debug=False, use_reloader=False,
                                         threaded=True),
            daemon=True,
        )
        self._hilo.start()
        self._log(f"[web] Página de texto/voz en http://0.0.0.0:{self.port}/")
