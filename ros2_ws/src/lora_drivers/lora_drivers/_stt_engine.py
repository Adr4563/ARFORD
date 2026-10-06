"""Reconocimiento de voz (STT) en español para Lora, con sherpa-onnx.

Se eligió sherpa-onnx sobre faster-whisper por dos razones concretas de este
proyecto:

1. Corre sobre `onnxruntime`, que YA es dependencia de este paquete para el
   detector de emociones (`_emotion_detector.py`). No suma un runtime nuevo a
   una Pi que va justa de RAM con Ollama residente.
2. `faster-whisper` se colgaba en silencio al construir `WhisperModel()` fuera
   del hilo principal -- que es justo como corre la entrada de audio (hilo de
   fondo). Ese bug nunca se resolvió y es la razón de que hoy no haya voz local
   en la Pi. sherpa-onnx no arrastra ese problema.

Mismo criterio de "falla gracioso" que el resto de los drivers: si falta la
librería o el modelo, `transcribir()` devuelve "" y se loguea con print; nunca
sale una excepción hacia `communication_node`.

Descarga de los modelos (una sola vez, en la Pi)::

    mkdir -p ~/.lora/modelos && cd ~/.lora/modelos
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-base.tar.bz2
    tar xvf sherpa-onnx-whisper-base.tar.bz2 && rm sherpa-onnx-whisper-base.tar.bz2
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

Para probar con el modelo más liviano, cambiar `base` por `tiny` en la URL y
poner LORA_STT_TAM=tiny. Cuál de los dos conviene se decide MIDIENDO en la Pi
con Ollama al lado (ver scripts/bench_stt.py), no por preferencia -- mismo
criterio que se usó para sacar el router y el corrector del LLM.

Variables de entorno:

    LORA_STT_MODELOS_DIR  carpeta de modelos      (default ~/.lora/modelos)
    LORA_STT_TAM          tiny | base | small     (default base)
    LORA_STT_HILOS        hilos de onnxruntime    (default 2)
    LORA_STT_IDIOMA       idioma del audio        (default es)
"""

import os
import time

MODELOS_DIR = os.environ.get(
    "LORA_STT_MODELOS_DIR", os.path.expanduser("~/.lora/modelos")
)
# tiny + 2 hilos: NO es una preferencia, es lo que salió de medir en la Pi real
# (4 núcleos, audio de 3.8s, ver scripts/bench_stt.py). Mediana por frase:
#
#   modelo / hilos     sin carga    con Ollama generando
#   tiny   / 2           2.22s          2.45s     <-- elegido
#   tiny   / 1             --           3.29s
#   tiny   / 4           3.45s            --
#   base   / 2           4.41s            --
#   base   / 4           5.57s            --
#
# Dos cosas contraintuitivas que confirman las medidas:
#
# 1. MÁS HILOS ES MÁS LENTO. Con 4 hilos no solo sube la mediana, se dispara la
#    varianza (3.63s-6.40s contra 2.21s-2.42s con 2). Mismo fenómeno que ya
#    estaba documentado para Piper, donde sintetizar tardaba igual con 1 que
#    con 4 hilos. El mínimo NO está en 1 hilo: bajo carga, 1 hilo es peor (3.29s).
# 2. Ollama apenas estorba: con el LLM saturando ~3 núcleos, el STT solo se
#    degrada un 10% (2.22s -> 2.45s). Conviven mejor de lo esperado.
#
# El precio de `tiny` es precisión: transcribe "Hola Laura" donde `base` acierta
# "Hola Lora". El nombre del robot se pierde. Para Trivia da igual (el
# agent_corrector compara con difflib y lo absorbe), pero si algún día el nombre
# importa -- una palabra de activación, por ejemplo -- hay que volver a `base` y
# pagar los 2 segundos extra.
TAM = os.environ.get("LORA_STT_TAM", "tiny")
HILOS = int(os.environ.get("LORA_STT_HILOS", "2"))

# Normalizacion automatica del volumen antes de transcribir.
#
# NO es un adorno: los microfonos de la placa ESP32-S3 entregan la senal muy
# floja por USB -- medido en la Pi, un pico del 26-31% de la escala hablandole
# de cerca. Con ese nivel whisper ALUCINA en vez de transcribir: devolvia
# '[MUSICA]' o repetia "hola, hola, hola..." indefinidamente. Amplificando la
# misma grabacion x3.5 salio una frase completa y correcta.
#
# Se amplifica hasta dejar el pico en 0.9 (no en 1.0, para no recortar), con un
# tope de ganancia para no convertir el silencio de sala en ruido atronador, y
# solo si hay algo que amplificar.
NORMALIZAR = os.environ.get("LORA_STT_NORMALIZAR", "1") not in ("0", "false", "False")
PICO_OBJETIVO = 0.9
GANANCIA_MAX = 20.0
PICO_MINIMO = 0.002  # por debajo de esto se asume silencio y no se toca
IDIOMA = os.environ.get("LORA_STT_IDIOMA", "es")

SAMPLE_RATE = 16000


class ReconocedorVoz:
    """Convierte audio en texto. Se carga perezosamente en el primer uso."""

    def __init__(self, modelos_dir=None, tam=None, hilos=None, idioma=None):
        self.modelos_dir = modelos_dir or MODELOS_DIR
        self.tam = tam or TAM
        self.hilos = hilos if hilos is not None else HILOS
        self.idioma = idioma or IDIOMA

        self._recognizer = None
        self._vad = None
        self._intento_fallido = False
        self.segundos_carga = 0.0
        self.ultima_ganancia = 1.0   # cuanto amplifico la ultima transcripcion

    # ------------------------------------------------------------------ carga

    def _rutas(self):
        base = os.path.join(self.modelos_dir, f"sherpa-onnx-whisper-{self.tam}")
        return {
            "encoder": os.path.join(base, f"{self.tam}-encoder.int8.onnx"),
            "decoder": os.path.join(base, f"{self.tam}-decoder.int8.onnx"),
            "tokens": os.path.join(base, f"{self.tam}-tokens.txt"),
            "vad": os.path.join(self.modelos_dir, "silero_vad.onnx"),
        }

    def cargar(self):
        """Carga el modelo. Devuelve True si quedó listo. Idempotente."""
        if self._recognizer is not None:
            return True
        if self._intento_fallido:
            return False

        # Import perezoso: el nodo tiene que poder arrancar en una máquina sin
        # sherpa-onnx instalado, igual que hace _emotion_detector.py con
        # picamera2. Mismo patrón, mismo motivo.
        try:
            import sherpa_onnx
        except ImportError:
            print("[STT] sherpa-onnx no instalado; el STT queda desactivado")
            self._intento_fallido = True
            return False

        rutas = self._rutas()
        faltan = [r for k, r in rutas.items() if k != "vad" and not os.path.exists(r)]
        if faltan:
            print(f"[STT] faltan modelos: {faltan[0]} (ver docstring para bajarlos)")
            self._intento_fallido = True
            return False

        t0 = time.perf_counter()
        try:
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
                encoder=rutas["encoder"],
                decoder=rutas["decoder"],
                tokens=rutas["tokens"],
                num_threads=self.hilos,
                decoding_method="greedy_search",
                language=self.idioma,
                task="transcribe",
            )
        except Exception as e:  # noqa: BLE001 - falla gracioso, no tumbar el nodo
            print(f"[STT] no se pudo cargar el modelo: {e}")
            self._intento_fallido = True
            return False

        # El VAD es opcional: solo hace falta para escuchar del micrófono en
        # continuo. Si no está, transcribir() sigue funcionando igual.
        if os.path.exists(rutas["vad"]):
            try:
                cfg = sherpa_onnx.VadModelConfig()
                cfg.silero_vad.model = rutas["vad"]
                cfg.silero_vad.min_silence_duration = 0.25
                cfg.sample_rate = SAMPLE_RATE
                self._vad = sherpa_onnx.VoiceActivityDetector(
                    cfg, buffer_size_in_seconds=30
                )
            except Exception as e:  # noqa: BLE001
                print(f"[STT] VAD no disponible: {e}")

        self.segundos_carga = time.perf_counter() - t0
        print(
            f"[STT] modelo whisper-{self.tam} listo en {self.segundos_carga:.1f}s "
            f"({self.hilos} hilos, idioma={self.idioma})"
        )
        return True

    @property
    def listo(self):
        return self._recognizer is not None

    # ------------------------------------------------------------ transcribir

    def _normalizar(self, samples):
        """Sube el volumen del audio flojo antes de transcribir.

        Devuelve las muestras tal cual si la normalizacion esta desactivada,
        si el audio ya viene con buen nivel, o si es practicamente silencio
        (amplificar silencio solo genera ruido y empeora la transcripcion).
        """
        if not NORMALIZAR:
            return samples
        try:
            import numpy as np

            arr = np.asarray(samples, dtype=np.float32)
            pico = float(np.abs(arr).max())
            if pico < PICO_MINIMO:
                return samples

            ganancia = min(PICO_OBJETIVO / pico, GANANCIA_MAX)
            if ganancia <= 1.05:      # ya venia bien, no tocar
                return samples

            self.ultima_ganancia = ganancia
            return np.clip(arr * ganancia, -1.0, 1.0)
        except Exception as e:  # noqa: BLE001 - nunca romper por esto
            print(f"[STT] no se pudo normalizar: {e}")
            return samples

    def transcribir(self, samples, sample_rate=SAMPLE_RATE):
        """Transcribe un bloque de audio ya capturado.

        `samples` es un array de float32 en [-1, 1] (lo que devuelve
        sounddevice con dtype="float32", o un WAV normalizado).

        Devuelve el texto, o "" si no se pudo. Nunca lanza.
        """
        if not self.cargar():
            return ""
        if samples is None or len(samples) == 0:
            return ""

        samples = self._normalizar(samples)

        try:
            stream = self._recognizer.create_stream()
            stream.accept_waveform(sample_rate, samples)
            self._recognizer.decode_stream(stream)
            return stream.result.text.strip()
        except Exception as e:  # noqa: BLE001
            print(f"[STT] fallo al transcribir: {e}")
            return ""

    def transcribir_wav(self, ruta):
        """Transcribe un archivo WAV mono. Útil para pruebas y para el bench."""
        try:
            import wave

            import numpy as np

            with wave.open(ruta, "rb") as w:
                if w.getnchannels() != 1 or w.getsampwidth() != 2:
                    print(f"[STT] {ruta}: se espera WAV mono de 16 bits")
                    return ""
                sr = w.getframerate()
                crudo = w.readframes(w.getnframes())
            samples = np.frombuffer(crudo, dtype=np.int16).astype("float32") / 32768.0
            return self.transcribir(samples, sr)
        except Exception as e:  # noqa: BLE001
            print(f"[STT] no se pudo leer {ruta}: {e}")
            return ""

    # -------------------------------------------------------------- micrófono

    def escuchar(self, al_transcribir, detener=None, dispositivo=None):
        """Escucha el micrófono y llama `al_transcribir(texto)` por cada frase.

        Corta las frases con el VAD (silencio de 0.25s), así que no hace falta
        que el usuario pulse nada para terminar de hablar.

        `detener` es un callable opcional que devuelve True para salir del
        bucle -- pensado para que el nodo lo apague en `on_deactivate`.

        Esta función BLOQUEA. El llamador la corre en su propio hilo.
        """
        if not self.cargar():
            return
        if self._vad is None:
            print("[STT] falta silero_vad.onnx; no se puede escuchar en continuo")
            return

        try:
            import numpy as np
            import sounddevice as sd
        except ImportError:
            print("[STT] falta sounddevice (pip install sounddevice)")
            return

        ventana = 512  # tamaño de ventana que espera silero
        buffer = np.array([], dtype="float32")

        try:
            with sd.InputStream(
                channels=1,
                dtype="float32",
                samplerate=SAMPLE_RATE,
                device=dispositivo,
            ) as mic:
                print("[STT] escuchando...")
                while True:
                    if detener is not None and detener():
                        break

                    bloque, _ = mic.read(int(0.1 * SAMPLE_RATE))
                    buffer = np.concatenate([buffer, bloque.reshape(-1)])

                    while len(buffer) > ventana:
                        self._vad.accept_waveform(buffer[:ventana])
                        buffer = buffer[ventana:]

                    while not self._vad.empty():
                        frase = self._vad.front.samples
                        self._vad.pop()
                        texto = self.transcribir(frase)
                        if texto:
                            al_transcribir(texto)
        except Exception as e:  # noqa: BLE001
            print(f"[STT] el micrófono falló: {e}")


# Instancia compartida, igual que el patrón de los Clients del proyecto
# original: se crea al importar pero NO carga el modelo hasta el primer uso.
reconocedor = ReconocedorVoz()


def transcribir(samples, sample_rate=SAMPLE_RATE):
    return reconocedor.transcribir(samples, sample_rate)
