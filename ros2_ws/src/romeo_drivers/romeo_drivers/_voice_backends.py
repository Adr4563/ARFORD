"""Voz de salida: Romeo dice en voz alta cada respuesta con Piper, 100%
local en la Pi y sin internet -- portado de Clients/Voice_Output_Client.py
del proyecto original (ver ese archivo para el benchmark de latencia). Lo
único que cambia es la integración: acá no hay una variable global
VOZ_MOTOR leída de env var al importar, sino los parámetros ROS2 que
communication_node declara y pasa a Voz.__init__().

Necesita el modelo .onnx de Piper en disco (ver `modelo_piper`). Si no se
puede cargar, el robot se queda MUDO: hablar() loguea y devuelve False,
sin cortar el turno -- mismo criterio de "falla gracioso" del resto de los
drivers, el chat nunca se cae por un problema de audio. Ya no hay motor de
respaldo, así que si falta el modelo hay que bajarlo (ver el README del
workspace).

Hubo otros dos motores, los dos eliminados a propósito:

- `telefono`: la voz la hacía el navegador del celular (Web Speech API) a
  través de la página web de _web_bridge.py. Se fue junto con la página,
  para no depender de un teléfono conectado.
- `edge`: edge-tts de Microsoft, mejor calidad pero sintetizando EN LA
  NUBE, así que exigía internet en cada frase. Se fue porque ARFORD tiene
  que funcionar en una red de colegio/hospital sin dar por hecho que hay
  salida a internet, y porque mandar a un servidor externo lo que el robot
  le dice a un niño no hace falta para nada.

hablar() BLOQUEA -- es lo que communication_node usa para atender el
servicio Speak de forma síncrona (ver la nota de "Garantía de orden" en
orchestrator_node.py: el turno completo depende de que este bloqueo sea
real, igual que voz_output.hablar() bloqueaba en el proyecto original).
"""
import os
import subprocess
import tempfile
import wave

# Piper (onnxruntime) agarra los núcleos de la Pi por default. Medido en el
# proyecto original: la síntesis tarda igual con 1, 2 o 4 hilos, así que
# limitarlo a 1 es gratis en velocidad y deja núcleos libres para Ollama
# (que corre en romeo_brain, otro proceso -- la contención de CPU es real
# igual, son procesos en la misma Pi). Tiene que ir ANTES de importar
# piper/onnxruntime, que leen esto al cargar.
os.environ.setdefault("OMP_NUM_THREADS", "1")


class Voz:
    """Encapsula el estado de voz (antes módulo con globals `MOTOR`/
    `_voz_piper` en Voice_Output_Client.py) para que communication_node
    pueda instanciarlo con los parámetros ROS2 del nodo en vez de leer
    variables de entorno directo."""

    def __init__(self, modelo_piper=None, logger=None):
        self.modelo_piper = modelo_piper or os.path.expanduser(
            "~/piper-voces/es_MX-claude-high.onnx")
        self.logger = logger
        self._voz_piper = None

    def _log(self, msg):
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

    @property
    def listo(self):
        return self._voz_piper is not None

    def cargar(self):
        """Precarga el modelo de Piper -- llamar en on_activate() del
        LifecycleNode, no en el constructor, para no tocar hardware antes
        de que el nodo esté realmente activo."""
        try:
            from piper import PiperVoice
            self._voz_piper = PiperVoice.load(self.modelo_piper)
            self._log(f"[voz] piper listo ({os.path.basename(self.modelo_piper)})")
        except Exception as e:
            # Ya no hay respaldo al que caer: se avisa fuerte y el robot
            # sigue funcionando mudo (el texto igual queda en el log).
            self._log(f"[voz] NO SE PUDO CARGAR PIPER ({e}) -- ARFORD no va a "
                       f"hablar. Falta {self.modelo_piper}?")

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

    def hablar(self, texto):
        """Sintetiza y reproduce `texto`, bloqueando hasta terminar.
        Devuelve True/False -- a diferencia del hablar() original (que no
        devolvía nada, communication_node necesita el bool para llenar
        Speak.srv.Response.success). No hace nada si texto viene vacío.
        Ante cualquier fallo loguea y sigue -- el chat nunca se cae por un
        problema de audio, mismo criterio que el proyecto original."""
        if not texto or not texto.strip():
            return False

        if self._voz_piper is None:
            self._log("[voz] piper no está cargado (ver cargar()), no se habla")
            return False
        try:
            self._hablar_piper(texto)
            return True
        except FileNotFoundError as e:
            # mpv no está instalado: el .wav se sintetizó pero no hay con qué
            # reproducirlo.
            self._log(f"[voz] falta un binario ({e}) -- no se puede hablar")
            return False
        except Exception as e:
            self._log(f"[voz] error al hablar con piper ({e})")
            return False
