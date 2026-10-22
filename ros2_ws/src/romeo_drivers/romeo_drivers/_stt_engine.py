"""Reconocimiento de voz (STT) en español para Romeo, con sherpa-onnx.

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

    mkdir -p ~/.romeo/modelos && cd ~/.romeo/modelos
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2
    tar xvf sherpa-onnx-whisper-tiny.tar.bz2 && rm sherpa-onnx-whisper-tiny.tar.bz2
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

`tiny` es el default porque ganó MIDIENDO en la Pi con Ollama al lado (ver la
tabla más abajo y scripts/bench_stt.py), no por preferencia -- mismo criterio
que se usó para sacar el router y el corrector del LLM. Para probar `base`,
cambiar `tiny` por `base` en la URL y poner ROMEO_STT_TAM=base.

`silero_vad.onnx` solo hace falta para escuchar el micrófono en continuo, que
es como lo usa communication_node. Sin él, transcribir() y el bench siguen
funcionando igual.

Variables de entorno:

    ROMEO_STT_MODELOS_DIR  carpeta de modelos      (default ~/.romeo/modelos)
    ROMEO_STT_TAM          tiny | base | small     (default tiny)
    ROMEO_STT_HILOS        hilos de onnxruntime    (default 2)
    ROMEO_STT_IDIOMA       idioma del audio        (default es)
    ROMEO_STT_NORMALIZAR   0 desactiva el ajuste de volumen (default 1)

communication_node pasa `tam` y `hilos` por constructor (parámetros ROS2
stt_tam/stt_hilos), así que para ese nodo ROMEO_STT_TAM/ROMEO_STT_HILOS quedan
sin efecto -- las demás variables sí se leen de acá.
"""

import json
import os
import time

import numpy as np

MODELOS_DIR = os.environ.get(
    "ROMEO_STT_MODELOS_DIR", os.path.expanduser("~/.romeo/modelos")
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
# "Hola Romeo". El nombre del robot se pierde. Para Trivia da igual (el
# agente_trivia compara con difflib y lo absorbe), pero si algún día el nombre
# importa -- una palabra de activación, por ejemplo -- hay que volver a `base` y
# pagar los 2 segundos extra.
# Motor de reconocimiento: "nemo" o "whisper".
#
# nemo es NVIDIA FastConformer transducer entrenado SOLO en espanol. Medido en
# la Pi con voz real por el microfono de la placa (10 s de audio):
#
#   nemo-transducer-es   2.2 s   "llamo, Adrian, es una buena oportunidad de
#                                 estar aqui contigo y espero que nos llevamos
#                                 a llevar bien, y es lo mas importante que
#                                 estoy consciente."
#   whisper-base         8.6 s   "Ya madre es una buena oportunidad [...] que
#                                 estoy con si"
#   whisper-small       30.1 s   acierta el nombre, pero 30 s no sirve para
#                                conversar
#
# nemo es 4x mas rapido que whisper-base y transcribe mejor: whisper se come
# el nombre y el final de la frase. whisper queda como alternativa porque es
# multilingue (nemo solo entiende espanol).
# Servidor de STT remoto (un stt_server.py corriendo en otra maquina). Vacio
# = transcribir en local. El microfono sigue en la Pi: lo que viaja es el WAV
# ya recortado por el VAD. Util porque whisper-small tarda ~30 s en la Pi y
# 0.5 s en una PC con GPU.
SERVIDOR = os.environ.get("ROMEO_STT_SERVIDOR", "")
TIMEOUT_REMOTO = float(os.environ.get("ROMEO_STT_TIMEOUT", "30"))

MOTOR = os.environ.get("ROMEO_STT_MOTOR", "nemo")
NEMO_DIR = os.environ.get(
    "ROMEO_STT_NEMO_DIR",
    "sherpa-onnx-nemo-fast-conformer-transducer-es-1424-int8")

# Solo se usa con MOTOR="whisper".
TAM = os.environ.get("ROMEO_STT_TAM", "tiny")
HILOS = int(os.environ.get("ROMEO_STT_HILOS", "2"))

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
NORMALIZAR = os.environ.get("ROMEO_STT_NORMALIZAR", "1") not in ("0", "false", "False")
PICO_OBJETIVO = 0.9
GANANCIA_MAX = 20.0
PICO_MINIMO = 0.002  # por debajo de esto se asume silencio y no se toca

# Cuanto silencio tiene que haber para dar la frase por terminada.
#
# Estaba en 0.25 s y troceaba al hablar: una pausa normal entre palabras,
# o al pensar, ya pasaba de ese cuarto de segundo, asi que el VAD cerraba
# la frase a medias y la siguiente parte llegaba como otro mensaje. En el
# log se veia asi:
#
#     [mic] transcrito: 'conversar, decidiria conversar.'   <- "deseria conversar"
#     [mic] transcrito: 'y'
#     [mic] transcrito: '¿Pueso?'                           <- "¿puedo?"
#
# Y cada trozo suelto se mandaba al LLM, que contestaba cualquier cosa.
#
# 0.6 s deja respirar sin que la espera se note: es el tiempo que tarda
# alguien en darse cuenta de que ya termino de hablar.
MIN_SILENCIO = float(os.environ.get("ROMEO_MIN_SILENCIO", "0.6"))
IDIOMA = os.environ.get("ROMEO_STT_IDIOMA", "es")

SAMPLE_RATE = 16000


class ReconocedorVoz:
    """Convierte audio en texto. Se carga perezosamente en el primer uso."""

    def __init__(self, modelos_dir=None, tam=None, hilos=None, idioma=None,
                 motor=None, servidor=None):
        self.modelos_dir = modelos_dir or MODELOS_DIR
        self.tam = tam or TAM
        self.hilos = hilos if hilos is not None else HILOS
        self.idioma = idioma or IDIOMA
        self.motor = (motor or MOTOR).lower()
        self.servidor = (servidor if servidor is not None else SERVIDOR).rstrip("/")

        self._recognizer = None
        self._vad = None
        self._intento_fallido = False
        self.segundos_carga = 0.0
        self.ultima_ganancia = 1.0   # cuanto amplifico la ultima transcripcion

    # ------------------------------------------------------------------ carga

    def _rutas(self):
        vad = os.path.join(self.modelos_dir, "silero_vad.onnx")
        if self.motor == "nemo":
            base = os.path.join(self.modelos_dir, "asr", NEMO_DIR)
            return {
                "encoder": os.path.join(base, "encoder.int8.onnx"),
                "decoder": os.path.join(base, "decoder.int8.onnx"),
                "joiner": os.path.join(base, "joiner.int8.onnx"),
                "tokens": os.path.join(base, "tokens.txt"),
                "vad": vad,
            }
        # tiny y base estan en modelos/, small bajo modelos/asr/ (lo dejo ahi
        # el script de descarga): se mira en los dos sitios para no tener que
        # mover carpetas de 1.2 GB.
        base = os.path.join(self.modelos_dir, f"sherpa-onnx-whisper-{self.tam}")
        if not os.path.isdir(base):
            alterno = os.path.join(self.modelos_dir, "asr",
                                   f"sherpa-onnx-whisper-{self.tam}")
            if os.path.isdir(alterno):
                base = alterno
        return {
            "encoder": os.path.join(base, f"{self.tam}-encoder.int8.onnx"),
            "decoder": os.path.join(base, f"{self.tam}-decoder.int8.onnx"),
            "tokens": os.path.join(base, f"{self.tam}-tokens.txt"),
            "vad": vad,
        }

    def _nombre_modelo(self):
        return "nemo-transducer-es" if self.motor == "nemo" else f"whisper-{self.tam}"

    def cargar_vad(self):
        """Carga SOLO el VAD (6 MB), que es lo unico que hace falta para
        escuchar el microfono en continuo. El modelo de reconocimiento no se
        toca: con un servidor remoto configurado no se usa nunca.

        Devuelve True si el VAD quedo listo."""
        if self._vad is not None:
            return True
        try:
            import sherpa_onnx
        except ImportError:
            print("[STT] sherpa-onnx no instalado; el STT queda desactivado")
            return False
        vad = self._rutas()["vad"]
        if not os.path.exists(vad):
            print(f"[STT] falta {vad}; no se puede escuchar en continuo")
            return False
        try:
            cfg = sherpa_onnx.VadModelConfig()
            cfg.silero_vad.model = vad
            cfg.silero_vad.min_silence_duration = MIN_SILENCIO
            cfg.sample_rate = SAMPLE_RATE
            self._vad = sherpa_onnx.VoiceActivityDetector(
                cfg, buffer_size_in_seconds=30)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[STT] VAD no disponible: {e}")
            return False

    def cargar(self):
        """Carga el modelo de reconocimiento. Devuelve True si quedo listo.
        Idempotente.

        Con `servidor` configurado NO se carga NUNCA, ni como respaldo: en
        el robot el STT corre solo en la PC. La Pi solo carga el VAD, que es
        lo que corta las frases del microfono. Si la PC no contesta, esa
        frase se pierde (ver transcribir()).

        Sin `servidor` si se carga: es el modo de stt_server.py y del bench,
        que corren en la PC con el modelo local."""
        if self.servidor:
            return self.cargar_vad()
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
            if self.motor == "nemo":
                self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                    encoder=rutas["encoder"],
                    decoder=rutas["decoder"],
                    joiner=rutas["joiner"],
                    tokens=rutas["tokens"],
                    num_threads=self.hilos,
                    decoding_method="greedy_search",
                    model_type="nemo_transducer",
                )
            else:
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
        self.cargar_vad()

        self.segundos_carga = time.perf_counter() - t0
        print(
            f"[STT] modelo {self._nombre_modelo()} listo en "
            f"{self.segundos_carga:.1f}s ({self.hilos} hilos, "
            f"idioma={self.idioma})"
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

    def _transcribir_remoto(self, samples, sample_rate):
        """Manda el audio a un stt_server.py de otra maquina y devuelve su
        texto, o None si no contesta. NO hay respaldo local: en el robot el
        STT corre solo en la PC.

        El microfono sigue estando en la Pi -- lo que viaja es el WAV ya
        recortado por el VAD, unos 100 KB por frase. Whisper-small tarda
        ~30 s en la Pi y 0.5 s en una PC con GPU, asi que compensa de sobra
        el viaje por red.
        """
        import io as _io
        import urllib.request
        import wave as _wave

        try:
            buf = _io.BytesIO()
            with _wave.open(buf, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sample_rate)
                arr = np.asarray(samples, dtype=np.float32)
                w.writeframes((np.clip(arr, -1, 1) * 32767).astype(np.int16).tobytes())
            req = urllib.request.Request(
                f"{self.servidor}/transcribir", data=buf.getvalue(),
                headers={"Content-Type": "audio/wav"})
            with urllib.request.urlopen(req, timeout=TIMEOUT_REMOTO) as r:
                return json.loads(r.read()).get("texto", "")
        except Exception as e:  # noqa: BLE001 - sin respaldo, se pierde la frase
            print(f"[STT] el servidor {self.servidor} no contesta ({e}); "
                  f"la frase se pierde (no hay STT en la Pi)")
            return None

    def transcribir(self, samples, sample_rate=SAMPLE_RATE):
        """Transcribe un bloque de audio ya capturado.

        `samples` es un array de float32 en [-1, 1] (lo que devuelve
        sounddevice con dtype="float32", o un WAV normalizado).

        Devuelve el texto, o "" si no se pudo. Nunca lanza.
        """
        if samples is None or len(samples) == 0:
            return ""

        # Con un servidor configurado se transcribe SOLO alla. Si no
        # contesta, la frase se pierde: no hay modelo en la Pi.
        if self.servidor:
            samples = self._normalizar(samples)
            texto = self._transcribir_remoto(samples, sample_rate)
            return (texto or "").strip()

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

    def escuchar(self, al_transcribir, detener=None, dispositivo=None,
                 pausar=None):
        """Escucha el micrófono y llama `al_transcribir(texto)` por cada frase.

        Corta las frases con el VAD (silencio de 0.25s), así que no hace falta
        que el usuario pulse nada para terminar de hablar.

        `detener` es un callable opcional que devuelve True para salir del
        bucle -- pensado para que el nodo lo apague en `on_deactivate`.

        `pausar` es otro callable opcional: mientras devuelva True se tira el
        audio que entra en vez de transcribirlo. El nodo le pasa su
        `_sonando()`, para no transcribir la propia voz de Romeo.

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

        # sounddevice resuelve `dispositivo` como SUBCADENA del nombre y
        # exige que coincida con uno solo: si coincide con varios lanza
        # "Multiple input devices found" y no se escucha nada. Pasa en
        # cuanto corre pipewire, porque la misma placa USB aparece como
        # dispositivo ALSA y además como uno o dos nodos de PulseAudio con
        # un nombre parecido. Se avisa con la lista de candidatos, que es
        # lo que hace falta para corregir el parámetro.
        if isinstance(dispositivo, str) and dispositivo:
            try:
                candidatos = [
                    (i, d["name"]) for i, d in enumerate(sd.query_devices())
                    if dispositivo in d["name"] and d["max_input_channels"] > 0
                ]
            except Exception:  # noqa: BLE001 - si falla el listado, seguir igual
                candidatos = []
            if len(candidatos) > 1:
                print(f"[STT] {dispositivo!r} coincide con {len(candidatos)} "
                      f"dispositivos de entrada, hace falta uno más preciso:")
                for i, nombre in candidatos:
                    print(f"[STT]   [{i}] {nombre}")
                return
            if not candidatos:
                print(f"[STT] ningún dispositivo de entrada coincide con "
                      f"{dispositivo!r} -- no se escucha")
                return

        ventana = 512  # tamaño de ventana que espera silero
        buffer = np.array([], dtype="float32")
        silencios = 0  # bloques seguidos por debajo de PICO_MINIMO

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

                    # Mientras Romeo suena se SIGUE leyendo del micrófono
                    # (si no, el buffer de ALSA se desborda), pero el audio
                    # se tira y el VAD se deja vacío: el micro y el altavoz
                    # están en la misma placa, así que todo lo que capta
                    # mientras habla es su propia voz. Sin esto, whisper la
                    # transcribía y el orquestador la procesaba como si la
                    # hubiera dicho el usuario -- Romeo acababa conversando
                    # consigo mismo en bucle.
                    if pausar is not None and pausar():
                        buffer = np.array([], dtype="float32")
                        self._vad.reset()
                        continue

                    muestras = bloque.reshape(-1)

                    # Compuerta por energia ANTES del VAD: silero es una red
                    # neuronal y se le pasaban ~31 ventanas por segundo sin
                    # parar, tambien en silencio. Eso solo costaba CPU (76%
                    # medido en la Pi) para que dijera "aqui no hay voz".
                    #
                    # Un bloque por debajo de PICO_MINIMO no puede contener
                    # voz -- es el mismo umbral con el que _normalizar()
                    # decide que algo es silencio -- asi que se descarta sin
                    # molestar al VAD. Hace falta un contador: si llega voz y
                    # el VAD no vio el silencio previo, no sabe donde empieza
                    # la frase, asi que al volver a oir algo se le da un
                    # bloque de silencio para que cierre lo anterior.
                    if float(np.abs(muestras).max()) < PICO_MINIMO:
                        silencios += 1
                        if silencios == 1 and len(buffer):
                            buffer = np.concatenate([buffer, muestras])
                        else:
                            continue
                    else:
                        if silencios:
                            self._vad.accept_waveform(
                                np.zeros(ventana, dtype="float32"))
                        silencios = 0
                        buffer = np.concatenate([buffer, muestras])

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
