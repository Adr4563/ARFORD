"""Voz de salida: Lora dice en voz alta cada respuesta. TRES motores,
portados tal cual de Clients/Voice_Output_Client.py del proyecto original
(mismo benchmark, mismos números, ver ese archivo para la comparativa
completa de latencia entre piper/edge-tts) -- lo único que cambia
es la integración: acá no hay una variable global VOZ_MOTOR leída de env
var al importar, sino un parámetro ROS2 (`voz_motor`) que communication_node
declara y pasa a Voz.__init__(), para poder tener varios devices probando
distintos motores sin depender de variables de entorno del proceso.

    voz_motor=piper      Piper, 100% local, corre en la Pi.
    voz_motor=vits       Voz VITS/Piper cargada con sherpa-onnx, 100% local, con
                         control fino (ruido, duración) y cambio de tono. Es la
                         que usa Lora: GLaDOS en español, "versión 40" elegida
                         de oído (ver lora_params.yaml).
    voz_motor=edge       edge-tts, NECESITA INTERNET.

El motor "telefono" (la voz la ponía el navegador del teléfono vía la
página web) se quitó junto con la página: ahora Lora habla por el parlante
de la placa ESP32-S3 (ver `audio_salida`).

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

from ._audio_salida import opciones_mpv

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

    def __init__(self, motor="piper", voz_edge="es-AR-ElenaNeural",
                 modelo_piper=None, logger=None, audio_salida=None,
                 audio_muestreo=0, volumen=100, vits=None):
        self.motor = (motor or "piper").strip().lower()
        self.voz_edge = voz_edge
        self.modelo_piper = modelo_piper or os.path.expanduser(
            "~/piper-voces/es_CO-dii.onnx")
        self.logger = logger
        # audio_salida / audio_muestreo: ver _audio_salida.opciones_mpv().
        self.audio_salida = audio_salida or None
        self.audio_muestreo = audio_muestreo
        # Volumen de la voz (0-100) que mpv aplica al reproducir. 100 = sin
        # atenuar; el .wav del TTS ya viene normalizado cerca de 0 dBFS.
        self.volumen = volumen
        self._voz_piper = None
        # vits: dict con dir, ruido, ruido_w, duracion, tono (semitonos).
        self.vits = vits or {}
        self._tts_vits = None

    def _cmd_mpv(self, ruta):
        return (["mpv", "--no-video", "--really-quiet", f"--volume={self.volumen}"]
                + opciones_mpv(self.audio_salida, self.audio_muestreo) + [ruta])

    def _log(self, msg):
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

    def cargar(self):
        """Precarga lo que el motor activo necesite -- llamar en
        on_activate() del LifecycleNode, no en el constructor, para no
        tocar hardware/red antes de que el nodo esté realmente activo."""
        if self.motor not in ("piper", "vits", "edge"):
            self._log(f"[voz] motor desconocido {self.motor!r} -- se usa "
                       f"{_MOTOR_RESPALDO}")
            self.motor = _MOTOR_RESPALDO
        if self.motor == "vits":
            self._cargar_vits()
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
                    noise_scale_w=float(self.vits.get("ruido_w", 0.8)),
                    length_scale=float(self.vits.get("duracion", 1.0))),
                num_threads=2))
            self._tts_vits = sherpa_onnx.OfflineTts(cfg)
            self._log(f"[voz] vits listo ({os.path.basename(modelo)}, "
                       f"tono {self.vits.get('tono', 0)} semitonos)")
        except Exception as e:
            self.motor = _MOTOR_RESPALDO
            self._log(f"[voz] no se pudo cargar vits ({e}) -- se usa "
                       f"{self.motor} como respaldo")

    def _hablar_vits(self, texto):
        """Sintetiza con sherpa-onnx y reproduce con mpv. El cambio de tono lo
        hace mpv al reproducir (asetrate + atempo: sube el tono sin cambiar
        la duración), igual que en las pruebas de oído con ffmpeg."""
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
            subprocess.run(self._cmd_mpv(ruta), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
            subprocess.run(self._cmd_mpv(ruta), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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

        if self.motor == "vits":
            try:
                self._hablar_vits(texto)
                return True
            except Exception as e:
                self._log(f"[voz] error al hablar con vits ({e})")
                return False

        # Solo llega edge-tts acá (motor="edge").
        self._hablar_edge(texto)
        return True
