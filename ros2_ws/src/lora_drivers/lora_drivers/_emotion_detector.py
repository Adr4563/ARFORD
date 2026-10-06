"""Detección de emoción facial -- fusión de ai-camera/reconocer_emocion.py
(la detección en sí, YuNet + FER+ sobre onnxruntime) y
deploy-raspberry-standalone/Clients/Camara_Client.py (el puente que
Orchestrator_Management.py usaba) del proyecto original.

En el monolito, Camara_Client.py importaba ai-camera/ vía sys.path porque
eran carpetas HERMANAS del mismo repo -- acá ya no hace falta ese puente:
visualization_faces_node vive en el mismo paquete que este archivo, así que
es un import normal. La lógica de detección (YuNet para encontrar la cara +
FER+ para clasificar la emoción, por qué CPU/ONNX y no el NPU del IMX500)
es EXACTAMENTE la de reconocer_emocion.py original -- ver
ai-camera/TODO-emociones-imx500.txt para la investigación completa detrás
de esa decisión.

Los .onnx (~230KB + ~35MB) se instalan como data_files del paquete (ver
setup.py) y se ubican con ament_index_python.

`onnxruntime`/`opencv`/`picamera2` se importan de forma perezosa (dentro de
las funciones, no al tope del módulo) -- mismo criterio que
Camara_Client.py original: visualization_faces_node tiene que poder
arrancar igual en una máquina sin cámara/esas dependencias instaladas
(ej. probando el resto del nodo en una PC de desarrollo), degradando
`detected=False` en vez de crashear.
"""
import os
import time

from ament_index_python.packages import get_package_share_directory

ETIQUETAS = ["neutral", "felicidad", "sorpresa", "tristeza", "enojo", "asco", "miedo", "desprecio"]
# Las 4 que importan para el dataset de trivia (columna 'cara': Feliz/
# Triste/Enojado/Neutral -- ver lora_brain/preguntas.py y
# lora_brain/orchestrator_node.py, Juego de emociones/imitación).
ETIQUETAS_RELEVANTES = {"felicidad": "Feliz", "tristeza": "Triste", "enojo": "Enojado", "neutral": "Neutral"}

_MAX_INTENTOS = 15
_PAUSA_ENTRE_INTENTOS = 0.2


class DetectorEmocion:
    """Encapsula el estado (sesión ONNX, detector de caras) -- se
    inicializa perezosamente en la primera detección, no en __init__, para
    que crear el objeto en on_configure() del nodo no falle si faltan
    dependencias; recién detectar_emocion() intenta cargar y devuelve
    detected=False si no puede."""

    def __init__(self, logger=None):
        self.logger = logger
        self.modelos_dir = os.path.join(get_package_share_directory('lora_drivers'), 'modelos')
        self._sess_emocion = None
        self._detector_cara = None
        self._input_name = None
        self._deps_ok = None  # None = no probado todavía; True/False una vez que se intenta

    def _log(self, msg, warn=False):
        if self.logger is not None:
            (self.logger.warn if warn else self.logger.info)(msg)
        else:
            print(msg)

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

    def _detectar_en_frame(self, frame_bgr):
        import cv2
        import numpy as np

        alto, ancho = frame_bgr.shape[:2]
        self._detector_cara.setInputSize((ancho, alto))
        _, caras = self._detector_cara.detect(frame_bgr)
        if caras is None or len(caras) == 0:
            return None, None

        cara = max(caras, key=lambda c: c[14])
        x, y, w, h = [max(0, int(v)) for v in cara[:4]]
        gris = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        recorte = gris[y:y + h, x:x + w]
        if recorte.size == 0:
            return None, None
        recorte = cv2.resize(recorte, (64, 64)).astype(np.float32)
        entrada = recorte.reshape(1, 1, 64, 64)

        salida = self._sess_emocion.run(None, {self._input_name: entrada})[0][0]
        probs = self._softmax(salida)
        idx = int(np.argmax(probs))
        return ETIQUETAS[idx], float(probs[idx])

    def detectar_emocion(self):
        """Captura un frame con la Raspberry Pi AI Camera (picamera2) y
        devuelve (emocion, confianza) -- emocion ya traducida a
        Feliz/Triste/Enojado/Neutral. Devuelve (None, None) ante CUALQUIER
        problema: dependencias faltantes, cámara no conectada, o sin cara
        detectada en los reintentos -- usado por visualization_faces_node
        para llenar DetectEmotion.Response."""
        if not self._asegurar_modelos_cargados():
            return None, None

        import cv2

        try:
            from picamera2 import Picamera2
        except ImportError as e:
            self._log(f"[cámara] picamera2 no disponible ({e})", warn=True)
            return None, None

        try:
            picam2 = Picamera2()
            config = picam2.create_preview_configuration()
            picam2.start(config, show_preview=False)
            time.sleep(2)  # deja asentar AE/AWB

            emocion, confianza = None, None
            for _ in range(_MAX_INTENTOS):
                frame = picam2.capture_array("main")
                frame_bgr = (cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                             if frame.shape[-1] == 3
                             else cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR))
                emocion, confianza = self._detectar_en_frame(frame_bgr)
                if emocion is not None:
                    break
                time.sleep(_PAUSA_ENTRE_INTENTOS)

            picam2.stop()
            # picam2.close() es obligatorio -- sin esto la cámara queda en
            # estado "Configured" (no "Available") y la siguiente
            # Picamera2() de este mismo proceso falla al acquire() (bug
            # real ya encontrado y arreglado en el proyecto original).
            picam2.close()
        except Exception as e:
            self._log(f"[cámara] falló la captura/detección ({e})", warn=True)
            return None, None

        if emocion is None:
            self._log("[cámara] no se detectó ninguna cara -- sigue sin veredicto")
            return None, None
        return ETIQUETAS_RELEVANTES.get(emocion), confianza
