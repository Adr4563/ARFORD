"""Voz de salida: Romeo dice en voz alta cada respuesta. DOS motores, los
dos 100% locales en la Pi y sin internet:

    voz_motor=vits   (default) la voz de ROMEO: GLaDOS en español, cargada
                       con sherpa-onnx. Es la elegida de oído ("versión
                       40"): los ajustes de fábrica del modelo, un poco más
                       pausada y +1.5 semitonos de tono. Da control fino
                       sobre ruido, duración y tono -- ver los parámetros
                       voz_vits_* en romeo_params.yaml.
    voz_motor=piper    Piper directo, sin control de tono. Se mantiene como
                       alternativa para comparar de oído.

Si el motor activo no carga (falta el modelo o sherpa-onnx), el robot se
queda MUDO: hablar() loguea y devuelve False, sin cortar el turno -- mismo
criterio de "falla gracioso" del resto de los drivers, el chat nunca se cae
por un problema de audio. Ya NO hay motor de respaldo, así que si falta el
modelo hay que bajarlo (ver el README del workspace).

Hubo otros dos motores, los dos eliminados a propósito:

- `telefono`: la voz la hacía el navegador del celular (Web Speech API) a
  través de la página web de _web_bridge.py. Se fue junto con la página,
  para no depender de un teléfono conectado. Ahora Romeo habla por el
  parlante de la placa ESP32-S3 (ver `audio_salida`).
- `edge`: edge-tts de Microsoft, mejor calidad pero sintetizando EN LA
  NUBE, así que exigía internet en cada frase. Se fue porque Romeo tiene
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

from ._audio_salida import opciones_mpv

# Piper/sherpa-onnx (onnxruntime) agarran los núcleos de la Pi por default.
# Medido en el proyecto original: la síntesis tarda igual con 1, 2 o 4
# hilos, así que limitarlo a 1 es gratis en velocidad y deja núcleos libres
# para Ollama (que corre en romeo_brain, otro proceso -- la contención de
# CPU es real igual, son procesos en la misma Pi). Tiene que ir ANTES de
# importar piper/onnxruntime, que leen esto al cargar.
os.environ.setdefault("OMP_NUM_THREADS", "1")


class Voz:
    """Encapsula el estado de voz (antes módulo con globals `MOTOR`/
    `_voz_piper` en Voice_Output_Client.py) para que communication_node
    pueda instanciarlo con los parámetros ROS2 del nodo en vez de leer
    variables de entorno directo."""

    def __init__(self, motor="vits", modelo_piper=None, logger=None,
                 audio_salida=None, audio_muestreo=0, volumen=100, vits=None):
        self.motor = (motor or "vits").strip().lower()
        self.modelo_piper = modelo_piper or os.path.expanduser(
            "~/piper-voces/es_CO-dii.onnx")
        self.logger = logger
        # audio_salida / audio_muestreo: ver _audio_salida.opciones_mpv().
        self.audio_salida = audio_salida or None
        self.audio_muestreo = audio_muestreo
        # Volumen de la voz (0-100) que mpv aplica al reproducir. 100 = sin
        # atenuar; el .wav del TTS ya viene normalizado cerca de 0 dBFS.
        self.volumen = volumen
        # vits: dict con dir, ruido, ruido_w, duracion, tono (semitonos).
        self.vits = vits or {}
        self._voz_piper = None
        self._tts_vits = None

    def _log(self, msg):
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

    def _cmd_mpv(self, ruta):
        return (["mpv", "--no-video", "--really-quiet", f"--volume={self.volumen}"]
                + opciones_mpv(self.audio_salida, self.audio_muestreo) + [ruta])

    @property
    def listo(self):
        return self._tts_vits is not None or self._voz_piper is not None

    # ─── Carga del motor activo ────────────────────────────────────────

    def cargar(self):
        """Precarga el modelo del motor activo -- llamar en on_activate()
        del LifecycleNode, no en el constructor, para no tocar hardware
        antes de que el nodo esté realmente activo."""
        if self.motor not in ("vits", "piper"):
            self._log(f"[voz] motor desconocido {self.motor!r} -- se usa vits")
            self.motor = "vits"
        if self.motor == "vits":
            self._cargar_vits()
        else:
            self._cargar_piper()

    def _cargar_vits(self):
        try:
            import glob

            import sherpa_onnx
            d = os.path.expanduser(self.vits.get("dir", ""))
            modelo = sorted(glob.glob(os.path.join(d, "*.onnx")))[0]
            cfg = sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=modelo, tokens=os.path.join(d, "tokens.txt"),
                    data_dir=os.path.join(d, "espeak-ng-data"),
                    noise_scale=float(self.vits.get("ruido", 0.667)),
                    noise_scale_w=float(self.vits.get("ruido_w", 0.5)),
                    length_scale=float(self.vits.get("duracion", 1.15))),
                num_threads=2))
            self._tts_vits = sherpa_onnx.OfflineTts(cfg)
            self._log(f"[voz] vits listo ({os.path.basename(modelo)}, "
                       f"tono {self.vits.get('tono', 0)} semitonos)")
        except Exception as e:
            # Sin respaldo al que caer: se avisa fuerte y el robot sigue
            # funcionando mudo (el texto igual queda en el log).
            self._log(f"[voz] NO SE PUDO CARGAR VITS ({e}) -- Romeo no va a "
                       f"hablar. ¿Falta el modelo en {self.vits.get('dir')!r}?")

    def _cargar_piper(self):
        try:
            from piper import PiperVoice
            self._voz_piper = PiperVoice.load(self.modelo_piper)
            self._log(f"[voz] piper listo ({os.path.basename(self.modelo_piper)})")
        except Exception as e:
            self._log(f"[voz] NO SE PUDO CARGAR PIPER ({e}) -- Romeo no va a "
                       f"hablar. ¿Falta {self.modelo_piper}?")

    # ─── Síntesis ──────────────────────────────────────────────────────

    def _hablar_vits(self, texto):
        """Sintetiza con sherpa-onnx y reproduce con mpv. El cambio de tono
        lo hace mpv al reproducir (asetrate + atempo: sube el tono sin
        cambiar la duración), igual que en las pruebas de oído con
        ffmpeg."""
        import numpy as np
        audio = self._tts_vits.generate(texto, sid=0)
        muestras = np.asarray(audio.samples, dtype=np.float32)
        if muestras.size == 0:
            return
        muestras *= 0.85 / max(1e-6, float(np.abs(muestras).max()))
        sr = audio.sample_rate
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            ruta = tmp.name
        try:
            with wave.open(ruta, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes((muestras * 32767).astype(np.int16).tobytes())
            cmd = self._cmd_mpv(ruta)
            tono = float(self.vits.get("tono", 0))
            if tono:
                f = 2 ** (tono / 12)
                cmd.insert(1, f"--af=lavfi=[asetrate={sr * f:.0f},aresample={sr},atempo={1 / f:.5f}]")
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        finally:
            try:
                os.unlink(ruta)
            except OSError:
                pass

    def _hablar_piper(self, texto):
        """Ver la nota histórica en Voice_Output_Client.py sobre por qué NO
        se usa streaming acá (chasquido audible al final de cada frase) --
        se porta tal cual, sintetiza a .wav completo y reproduce con mpv."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            ruta = tmp.name
        try:
            with wave.open(ruta, "wb") as w:
                self._voz_piper.synthesize_wav(texto, w)
            subprocess.run(self._cmd_mpv(ruta),
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

        if not self.listo:
            self._log(f"[voz] motor {self.motor!r} no está cargado "
                       f"(ver cargar()), no se habla")
            return False
        try:
            if self.motor == "vits":
                self._hablar_vits(texto)
            else:
                self._hablar_piper(texto)
            return True
        except FileNotFoundError as e:
            # mpv no está instalado: el .wav se sintetizó pero no hay con
            # qué reproducirlo.
            self._log(f"[voz] falta un binario ({e}) -- no se puede hablar")
            return False
        except Exception as e:
            self._log(f"[voz] error al hablar con {self.motor} ({e})")
            return False
