"""Voz de salida: Lora dice en voz alta cada respuesta. TRES motores,
portados tal cual de Clients/Voice_Output_Client.py del proyecto original
(mismo benchmark, mismos números, ver ese archivo para la comparativa
completa de latencia entre piper/edge-tts/teléfono) -- lo único que cambia
es la integración: acá no hay una variable global VOZ_MOTOR leída de env
var al importar, sino un parámetro ROS2 (`voz_motor`) que communication_node
declara y pasa a Voz.__init__(), para poder tener varios devices probando
distintos motores sin depender de variables de entorno del proceso.

    voz_motor=telefono  (default) la habla el navegador del teléfono (Web
                          Speech API) -- local a ESE dispositivo, sin nube
                          y sin gastar CPU/parlante de la Pi. Necesita que
                          alguien tenga la página abierta con la voz
                          activada (ver _web_bridge.py); si no hay nadie
                          conectado o no contesta a tiempo, cae a edge-tts.
    voz_motor=piper      Piper, 100% local, corre en la Pi.
    voz_motor=edge       edge-tts, NECESITA INTERNET.

hablar() BLOQUEA -- es lo que communication_node usa para atender el
servicio Speak de forma síncrona (ver la nota de "Garantía de orden" en
orchestrator_node.py: el turno completo depende de que este bloqueo sea
real, igual que voz_output.hablar() bloqueaba en el proyecto original).
"""
import asyncio
import os
import subprocess
import tempfile
import wave

# Piper (onnxruntime) agarra los núcleos de la Pi por default. Medido en el
# proyecto original: la síntesis tarda igual con 1, 2 o 4 hilos, así que
# limitarlo a 1 es gratis en velocidad y deja núcleos libres para Ollama
# (que corre en lora_brain, otro proceso -- la contención de CPU es real
# igual, son procesos en la misma Pi). Tiene que ir ANTES de importar
# piper/onnxruntime, que leen esto al cargar.
os.environ.setdefault("OMP_NUM_THREADS", "1")

_MOTOR_RESPALDO = "edge"


class Voz:
    """Encapsula el estado de voz (antes módulo con globals `MOTOR`/
    `_voz_piper` en Voice_Output_Client.py) para que communication_node
    pueda instanciarlo con los parámetros ROS2 del nodo en vez de leer
    variables de entorno directo."""

    def __init__(self, motor="telefono", voz_edge="es-AR-ElenaNeural",
                 modelo_piper=None, web_bridge=None, logger=None):
        self.motor = (motor or "telefono").strip().lower()
        self.voz_edge = voz_edge
        self.modelo_piper = modelo_piper or os.path.expanduser(
            "~/piper-voces/es_MX-claude-high.onnx")
        # web_bridge: instancia de _web_bridge.WebBridge -- hablar_telefono()
        # necesita mandarle el texto por SSE a los clientes conectados y
        # esperar el ack de "terminó de hablar". Se inyecta en vez de
        # importar el módulo directo (como hacía Voice_Output_Client.hablar()
        # con `import voz_server`) para no acoplar este archivo al framework
        # web -- communication_node es quien conoce a los dos.
        self.web_bridge = web_bridge
        self.logger = logger
        self._voz_piper = None

    def _log(self, msg):
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

    def cargar(self):
        """Precarga lo que el motor activo necesite -- llamar en
        on_activate() del LifecycleNode, no en el constructor, para no
        tocar hardware/red antes de que el nodo esté realmente activo."""
        if self.motor == "telefono":
            self._log("[voz] motor=telefono -- abrí la página web y tocá "
                       "'Activar voz de Lora' en el teléfono")
            return
        if self.motor != "piper":
            return
        try:
            from piper import PiperVoice
            self._voz_piper = PiperVoice.load(self.modelo_piper)
            self._log(f"[voz] piper listo ({os.path.basename(self.modelo_piper)})")
        except Exception as e:
            self.motor = _MOTOR_RESPALDO
            self._log(f"[voz] no se pudo cargar piper ({e}) -- se usa "
                       f"{self.motor} como respaldo")

    async def _sintetizar_edge(self, texto, ruta):
        import edge_tts
        comunicador = edge_tts.Communicate(texto, self.voz_edge)
        await comunicador.save(ruta)

    def _hablar_piper(self, texto):
        """Ver la nota histórica en Voice_Output_Client.py sobre por qué NO
        se usa streaming acá (chasquido audible al final de cada frase) --
        se porta tal cual, sintetiza a .wav completo y reproduce con mpv."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            ruta = tmp.name
        try:
            with wave.open(ruta, "wb") as w:
                self._voz_piper.synthesize_wav(texto, w)
            subprocess.run(["mpv", "--no-video", "--really-quiet", ruta],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        finally:
            try:
                os.unlink(ruta)
            except OSError:
                pass

    def _hablar_edge(self, texto):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            ruta = tmp.name
        try:
            asyncio.run(self._sintetizar_edge(texto, ruta))
            subprocess.run(["mpv", "--no-video", "--really-quiet", ruta],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError as e:
            self._log(f"[voz] falta mpv ({e}) -- no se puede hablar")
        except Exception as e:
            pista = " -- ¿sin internet?" if self.motor == "edge" else ""
            self._log(f"[voz] error al hablar con motor=edge ({e}){pista}")
        finally:
            try:
                os.unlink(ruta)
            except OSError:
                pass

    def hablar(self, texto):
        """Sintetiza y reproduce `texto`, bloqueando hasta terminar.
        Devuelve True/False -- a diferencia del hablar() original (que no
        devolvía nada, communication_node necesita el bool para llenar
        Speak.srv.Response.success). No hace nada si texto viene vacío.
        Ante cualquier fallo del motor loguea y sigue -- el chat nunca se
        cae por un problema de audio, mismo criterio que el proyecto
        original."""
        if not texto or not texto.strip():
            return False

        if self.motor == "telefono":
            exito = self.web_bridge.hablar_telefono(texto) if self.web_bridge else False
            if exito:
                return True
            self._log(f"[voz] ningún teléfono conectado/activo, se usa "
                       f"{_MOTOR_RESPALDO} como respaldo")
            self._hablar_edge(texto)
            return True

        if self.motor == "piper":
            if self._voz_piper is None:
                self._log("[voz] piper no está cargado (ver cargar()), no se habla")
                return False
            try:
                self._hablar_piper(texto)
                return True
            except FileNotFoundError as e:
                self._log(f"[voz] falta un binario ({e}) -- no se puede hablar")
                return False
            except Exception as e:
                self._log(f"[voz] error al hablar con piper ({e})")
                return False

        # Solo llega edge-tts acá (motor="edge", el default explícito).
        self._hablar_edge(texto)
        return True
