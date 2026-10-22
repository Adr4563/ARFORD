"""Detección de emoción facial (FER) -- YuNet para encontrar la cara + FER+
para clasificar la emoción, sobre onnxruntime en la CPU de la Pi.

Viene de ai-camera/reconocer_emocion.py y Clients/Camara_Client.py del
proyecto original (por qué CPU/ONNX y no el NPU del IMX500: ver
ai-camera/TODO-emociones-imx500.txt). Los .onnx (~230KB + ~35MB) se
instalan como data_files del paquete y se ubican con ament_index_python.

DOS MODOS DE USO

1. CONTINUO (iniciar_continuo): la cámara queda abierta y un hilo de fondo
   clasifica ~FPS cuadros por segundo, guardando la distribución COMPLETA de
   las 8 emociones de cada cuadro con su instante. Es el FER de la ventana
   de percepción afectiva de la tesis de Rocha et al. (sec. 3.4.1.2, fig.
   14): mientras el usuario habla se acumulan observaciones, y al terminar
   la frase resumen(t0, t1) las resume en UNA sola, con la mediana por
   emoción normalizada (sec. 4.1.4.1). Ese resumen es lo que va a PULSE.

   El juego de caras (detectar_emocion) también sale de este flujo: no
   puede abrir la cámara otra vez porque solo un proceso puede tenerla.

2. PUNTUAL (sin iniciar_continuo): detectar_emocion() abre la cámara,
   intenta hasta _MAX_INTENTOS cuadros y la cierra. Es el comportamiento de
   siempre, para cuando el modo continuo está desactivado.

`onnxruntime`/`opencv`/`picamera2` se importan de forma perezosa: el nodo
tiene que poder arrancar en una máquina sin cámara ni esas dependencias,
degradando a "sin detección" en vez de caerse.
"""
import collections
import os
import threading
import time

from ament_index_python.packages import get_package_share_directory

# Orden de salida de FER+. Las claves coinciden con ANCLAJES de
# romeo_brain/estado_afectivo.py, así PULSE convierte a VAD sin traducir.
ETIQUETAS = ["neutral", "felicidad", "sorpresa", "tristeza", "enojo", "asco", "miedo", "desprecio"]
# Las 4 que usa el juego de caras (columna 'cara' del dataset).
ETIQUETAS_RELEVANTES = {"felicidad": "Feliz", "tristeza": "Triste", "enojo": "Enojado", "neutral": "Neutral"}

_MAX_INTENTOS = 15
_PAUSA_ENTRE_INTENTOS = 0.2

# Modo continuo.
FPS_POR_DEFECTO = 2.0
HISTORIAL_SEG = 60.0       # cuántos segundos de cuadros se guardan
ESPERA_JUEGO_SEG = 4.0     # cuánto espera el juego de caras a ver una cara
CUADROS_JUEGO = 3          # con cuántos cuadros decide el juego de caras


class DetectorEmocion:
    """Estado del detector (sesión ONNX, detector de caras y, en modo
    continuo, la cámara y el historial). Se inicializa perezosamente: crear
    el objeto en on_configure() no toca nada."""

    def __init__(self, logger=None, fps=FPS_POR_DEFECTO):
        self.logger = logger
        self.modelos_dir = os.path.join(get_package_share_directory('romeo_drivers'), 'modelos')
        self._sess_emocion = None
        self._detector_cara = None
        self._input_name = None
        self._deps_ok = None  # None = no probado; True/False una vez intentado
        # Distribución completa de la última inferencia, en el orden de
        # ETIQUETAS. None mientras no haya habido ninguna.
        self.ultimas_probabilidades = None

        # Modo continuo: (instante time.time(), [8 probabilidades]) por
        # cada cuadro en que se vio una cara.
        self.fps = max(0.2, float(fps))
        self._observaciones = collections.deque(
            maxlen=int(HISTORIAL_SEG * self.fps) + 1)
        self._lock = threading.Lock()
        self._hilo = None
        self._continuo = False

    def _log(self, msg, warn=False):
        if self.logger is not None:
            # .warning, no .warn: RcutilsLogger de ROS 2 no tiene el alias.
            (self.logger.warning if warn else self.logger.info)(msg)
        else:
            print(msg)

    # ─── Modelos ─────────────────────────────────────────────────────

    def _asegurar_modelos_cargados(self):
        if self._deps_ok is not None:
            return self._deps_ok
        try:
            import cv2
            import onnxruntime as ort

            modelo_emocion = os.path.join(self.modelos_dir, 'emotion-ferplus-8.onnx')
            modelo_cara = os.path.join(self.modelos_dir, 'face_detection_yunet_2023mar.onnx')
            self._sess_emocion = ort.InferenceSession(modelo_emocion)
            self._input_name = self._sess_emocion.get_inputs()[0].name
            self._detector_cara = cv2.FaceDetectorYN_create(
                modelo_cara, "", (320, 320), score_threshold=0.7)
            self._deps_ok = True
        except Exception as e:
            self._log(f"[cámara] no disponible ({e}) -- sigue sin veredicto de cámara", warn=True)
            self._deps_ok = False
        return self._deps_ok

    @staticmethod
    def _softmax(x):
        import numpy as np
        e = np.exp(x - np.max(x))
        return e / e.sum()

    @staticmethod
    def _a_bgr(frame):
        import cv2
        return (cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.shape[-1] == 3
                else cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR))

    def _clasificar(self, frame_bgr):
        """Las 8 probabilidades de FER+ (orden ETIQUETAS) para la cara más
        clara del cuadro, o None si no hay cara."""
        import cv2
        import numpy as np

        alto, ancho = frame_bgr.shape[:2]
        self._detector_cara.setInputSize((ancho, alto))
        _, caras = self._detector_cara.detect(frame_bgr)
        if caras is None or len(caras) == 0:
            return None

        cara = max(caras, key=lambda c: c[14])
        x, y, w, h = [max(0, int(v)) for v in cara[:4]]
        gris = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        recorte = gris[y:y + h, x:x + w]
        if recorte.size == 0:
            return None
        recorte = cv2.resize(recorte, (64, 64)).astype(np.float32)
        salida = self._sess_emocion.run(
            None, {self._input_name: recorte.reshape(1, 1, 64, 64)})[0][0]
        probs = [float(p) for p in self._softmax(salida)]
        self.ultimas_probabilidades = probs
        return probs

    def _detectar_en_frame(self, frame_bgr):
        """(etiqueta, probabilidad) de la emoción ganadora, o (None, None)."""
        probs = self._clasificar(frame_bgr)
        if probs is None:
            return None, None
        idx = max(range(len(probs)), key=probs.__getitem__)
        return ETIQUETAS[idx], probs[idx]

    # ─── Modo continuo (ventana de percepción) ──────────────────────────

    @property
    def continuo_activo(self):
        return self._hilo is not None and self._hilo.is_alive()

    def iniciar_continuo(self):
        """Abre la cámara y empieza a clasificar de fondo. Idempotente.
        Devuelve False si no hay modelos o dependencias."""
        if self.continuo_activo:
            return True
        if not self._asegurar_modelos_cargados():
            return False
        self._continuo = True
        self._hilo = threading.Thread(target=self._bucle, daemon=True)
        self._hilo.start()
        return True

    def detener_continuo(self):
        self._continuo = False
        if self._hilo is not None:
            self._hilo.join(timeout=3.0)
        self._hilo = None

    def _bucle(self):
        """Cuerpo del hilo de la cámara. Si la cámara falla (desconectada,
        ocupada), reintenta cada 5 s sin tirar el nodo."""
        while self._continuo:
            picam2 = None
            try:
                from picamera2 import Picamera2
                picam2 = Picamera2()
                picam2.start(picam2.create_preview_configuration(), show_preview=False)
                time.sleep(2)  # que asiente la exposición
                self._log(f"[cámara] FER continuo a {self.fps:g} fps")
                periodo = 1.0 / self.fps
                while self._continuo:
                    inicio = time.time()
                    probs = self._clasificar(self._a_bgr(picam2.capture_array("main")))
                    if probs is not None:
                        with self._lock:
                            self._observaciones.append((inicio, probs))
                    time.sleep(max(0.0, periodo - (time.time() - inicio)))
            except Exception as e:  # noqa: BLE001 - nunca tirar el nodo
                self._log(f"[cámara] FER continuo falló ({e}), reintento en 5 s", warn=True)
                time.sleep(5)
            finally:
                if picam2 is not None:
                    try:
                        picam2.stop()
                        # close() es obligatorio: sin él la siguiente
                        # Picamera2() del proceso no puede adquirir la cámara.
                        picam2.close()
                    except Exception:  # noqa: BLE001
                        pass

    def _cuadros_entre(self, t0, t1):
        with self._lock:
            return [p for t, p in self._observaciones if t0 <= t <= t1]

    @staticmethod
    def resumir(cuadros):
        """Resume varios cuadros en UNA observación, como en la tesis
        (sec. 4.1.4.1): mediana de cada emoción y normalización para que
        las probabilidades vuelvan a sumar 1.

        La mediana y no la media: un cuadro suelto mal clasificado (la cara
        de perfil, un parpadeo) no arrastra el resultado.

        Devuelve la lista de 8 probabilidades, o None si no hay cuadros."""
        if not cuadros:
            return None
        import numpy as np
        mediana = np.median(np.asarray(cuadros, dtype=float), axis=0)
        total = float(mediana.sum())
        if total <= 0.0:
            return None
        return [float(p) / total for p in mediana]

    def resumen(self, t0, t1):
        """(probabilidades, n_cuadros) de la ventana [t0, t1], en segundos
        de time.time(). (None, 0) si en esa ventana no se vio ninguna cara."""
        cuadros = self._cuadros_entre(t0, t1)
        return self.resumir(cuadros), len(cuadros)

    # ─── Juego de caras ──────────────────────────────────────────────

    def detectar_emocion(self):
        """(emoción, confianza) para el juego de caras, con la emoción ya
        traducida a Feliz/Triste/Enojado/Neutral, o (None, None) si no se
        vio ninguna cara o la ganadora no es una de esas cuatro.

        Con el modo continuo activo usa los cuadros que lleguen desde ahora
        (la cámara ya está abierta por el hilo). Si no, abre la cámara."""
        if self.continuo_activo:
            return self._detectar_desde_continuo()
        return self._detectar_puntual()

    def _detectar_desde_continuo(self):
        desde = time.time()
        limite = desde + ESPERA_JUEGO_SEG
        cuadros = []
        while time.time() < limite:
            cuadros = self._cuadros_entre(desde, time.time())
            if len(cuadros) >= CUADROS_JUEGO:
                break
            time.sleep(0.1)
        probs = self.resumir(cuadros)
        if probs is None:
            self._log("[cámara] no se detectó ninguna cara -- sigue sin veredicto")
            return None, None
        idx = max(range(len(probs)), key=probs.__getitem__)
        return ETIQUETAS_RELEVANTES.get(ETIQUETAS[idx]), probs[idx]

    def _detectar_puntual(self):
        if not self._asegurar_modelos_cargados():
            return None, None
        try:
            from picamera2 import Picamera2
        except ImportError as e:
            self._log(f"[cámara] picamera2 no disponible ({e})", warn=True)
            return None, None

        try:
            picam2 = Picamera2()
            picam2.start(picam2.create_preview_configuration(), show_preview=False)
            time.sleep(2)  # deja asentar AE/AWB

            emocion, confianza = None, None
            for _ in range(_MAX_INTENTOS):
                emocion, confianza = self._detectar_en_frame(
                    self._a_bgr(picam2.capture_array("main")))
                if emocion is not None:
                    break
                time.sleep(_PAUSA_ENTRE_INTENTOS)

            picam2.stop()
            picam2.close()  # obligatorio, ver _bucle()
        except Exception as e:
            self._log(f"[cámara] falló la captura/detección ({e})", warn=True)
            return None, None

        if emocion is None:
            self._log("[cámara] no se detectó ninguna cara -- sigue sin veredicto")
            return None, None
        return ETIQUETAS_RELEVANTES.get(emocion), confianza
