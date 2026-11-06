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
   la frase resumen(t0, t1) las resume en UNA sola, con la media de cada
   emoción ponderada por el score de YuNet de cada cuadro y normalizada
   (paso 0, sec. 4.1.4.1). Ese resumen es lo que va a PULSE.

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

# Hilos para onnxruntime y OpenCV. Ver la nota en _asegurar_modelos_cargados.
HILOS_ONNX = int(os.environ.get("ROMEO_FER_HILOS", "2"))

# Buscar la cara dentro del sensor en vez de en la CPU (ver _yunet_camara).
# ROMEO_YUNET_CAMARA=0 lo desactiva y todo vuelve a la CPU.
YUNET_EN_CAMARA = os.environ.get("ROMEO_YUNET_CAMARA", "1") != "0"

from ._yunet_camara import YunetCamara  # noqa: E402

def _media_recortada(p, proporcion):
    """Media recortada por columnas (respaldo de DetectorEmocion.resumir).

    NUEVA. Reemplaza a la mediana como respaldo cuando no hay calidades:
    igual que ella ignora los cuadros extremos, pero promedia los del
    medio en vez de quedarse con uno solo.

    scipy no es dependencia de la Pi: si esta se usa trim_mean, si no se
    hace la misma cuenta -- ordenar cada columna y quitar int(proporcion*N)
    valores de cada extremo. Con N < 7 y 15% no se quita nada y queda la
    media simple, igual que en scipy."""
    import numpy as np
    try:
        from scipy.stats import trim_mean
        return np.asarray(trim_mean(p, proportiontocut=proporcion, axis=0), dtype=float)
    except ImportError:
        pass
    n = p.shape[0]
    k = int(proporcion * n)
    if n - 2 * k <= 0:
        return np.median(p, axis=0)
    return np.sort(p, axis=0)[k:n - k].mean(axis=0)


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

        # Modo continuo: (instante time.time(), [8 probabilidades], q) por
        # cada cuadro en que se vio una cara. q es el score de YuNet de esa
        # cara, en [0, 1], o None si no se conoce. ANTES la tupla era solo
        # (instante, probabilidades): q se agrego para la media ponderada
        # por calidad de resumir().
        self.fps = max(0.2, float(fps))
        self._observaciones = collections.deque(
            maxlen=int(HISTORIAL_SEG * self.fps) + 1)
        self._lock = threading.Lock()
        self._hilo = None
        self._continuo = False

    def _log(self, msg, warn=False):
        """Loguea sin poder tumbar al hilo del FER.

        SIEMPRE LA MISMA SEVERIDAD. Alternar .info y .warning sobre el
        mismo logger revienta en esta Pi (Python 3.14 + lyrical) con
        "Logger severity cannot be changed between calls", y como el
        primer aviso del FER es un warning ("YuNet en el sensor no
        disponible"), el .info siguiente tiraba el hilo de captura: la
        camara se quedaba sin FER toda la sesion.

        El try ademas evita que un fallo del logger vuelva a matar la
        captura por un mensaje de consola."""
        if self.logger is None:
            print(msg)
            return
        try:
            # .warning y no .warn: el RcutilsLogger de ROS 2 no tiene el
            # alias corto.
            self.logger.warning(msg)
        except Exception:  # noqa: BLE001 - loguear nunca es critico
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
            # LIMITAR LOS HILOS. Sin SessionOptions, onnxruntime abre un
            # hilo por nucleo: medido en la Pi, el nodo se iba a 347% de
            # CPU (3,5 de 4 nucleos) con cuatro ThreadPool al ~70% cada
            # uno, y el load average a 6.9. Eso deja sin CPU al STT, al
            # TTS y a ROS, que comparten la misma Pi.
            #
            # Con 2 hilos basta: el modelo es de 35 MB y solo hay que
            # clasificar FER_FPS cuadros por segundo, no un video entero.
            # Es lo mismo que ya hacen pruebas/fer/fer.py y _ter_engine.py.
            op = ort.SessionOptions()
            op.intra_op_num_threads = HILOS_ONNX
            op.inter_op_num_threads = 1
            self._sess_emocion = ort.InferenceSession(modelo_emocion, op)
            self._input_name = self._sess_emocion.get_inputs()[0].name
            self._detector_cara = cv2.FaceDetectorYN_create(
                modelo_cara, "", (320, 320), score_threshold=0.7)
            # OpenCV (YuNet) tiene su propio pool, aparte del de onnx.
            cv2.setNumThreads(HILOS_ONNX)
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

    def _clasificar(self, frame_bgr, caja=None):
        """Las 8 probabilidades de FER+ (orden ETIQUETAS) para la cara más
        clara del cuadro, o None si no hay cara.

        `caja` es (x, y, w, h) cuando la cara YA la encontro el sensor
        (ver _yunet_camara.py): entonces no se corre YuNet en la CPU, que
        es el 52% del coste por cuadro. Sin caja, se busca aqui como
        siempre."""
        import cv2
        import numpy as np

        return self._clasificar_con_calidad(frame_bgr, caja)[0]

    def _clasificar_con_calidad(self, frame_bgr, caja=None, score=None):
        """(probabilidades, q) como _clasificar, con q = score de YuNet de
        la cara elegida (columna 14 de cv2.FaceDetectorYN, o el que dio el
        sensor en `score`). q es None si no se conoce: resumir() cae
        entonces a la media recortada. (None, None) si no hay cara.

        NUEVA: es el cuerpo que antes tenia _clasificar(). Ese codigo ya
        leia cara[14] para elegir la cara mas clara, pero no devolvia el
        valor. _clasificar() sigue existiendo y devuelve solo las
        probabilidades, para el modo puntual y el juego de caras."""
        import cv2
        import numpy as np

        q = None
        if caja is not None:
            x, y, w, h = (max(0, int(v)) for v in caja)
            q = score
        else:
            alto, ancho = frame_bgr.shape[:2]
            self._detector_cara.setInputSize((ancho, alto))
            _, caras = self._detector_cara.detect(frame_bgr)
            if caras is None or len(caras) == 0:
                return None, None
            cara = max(caras, key=lambda c: c[14])
            x, y, w, h = [max(0, int(v)) for v in cara[:4]]
            q = float(cara[14])
        gris = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        recorte = gris[y:y + h, x:x + w]
        if recorte.size == 0:
            return None, None
        recorte = cv2.resize(recorte, (64, 64)).astype(np.float32)
        salida = self._sess_emocion.run(
            None, {self._input_name: recorte.reshape(1, 1, 64, 64)})[0][0]
        probs = [float(p) for p in self._softmax(salida)]
        self.ultimas_probabilidades = probs
        if q is not None:
            q = max(0.0, min(1.0, float(q)))
        return probs, q

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

                # YuNet dentro del sensor, si el firmware esta instalado.
                # Si no, se sigue con YuNet en la CPU: el robot funciona
                # igual, solo gasta mas.
                sensor = YunetCamara(log=self._log) if YUNET_EN_CAMARA else None
                if sensor is not None and not sensor.cargar():
                    self._log(f"[cámara] YuNet en el sensor no disponible "
                              f"({sensor.error}); se usa la CPU", warn=True)
                    sensor = None

                picam2 = (Picamera2(sensor.camera_num) if sensor is not None
                          else Picamera2())
                picam2.start(picam2.create_preview_configuration(), show_preview=False)
                time.sleep(2)  # que asiente la exposición
                donde = "sensor" if sensor is not None else "CPU"
                self._log(f"[cámara] FER continuo a {self.fps:g} fps "
                          f"(cara: {donde})")
                periodo = 1.0 / self.fps
                while self._continuo:
                    inicio = time.time()
                    if sensor is not None:
                        # La cara la busca el acelerador del IMX500; aqui
                        # solo se recorta y se clasifica.
                        pedido = picam2.capture_request()
                        try:
                            cuadro = pedido.make_array("main")
                            alto, ancho = cuadro.shape[:2]
                            caja, score = sensor.cara_y_score(
                                pedido.get_metadata(), ancho, alto)
                        finally:
                            pedido.release()
                        probs, q = (self._clasificar_con_calidad(
                                        self._a_bgr(cuadro), caja, score)
                                    if caja is not None else (None, None))
                    else:
                        probs, q = self._clasificar_con_calidad(
                            self._a_bgr(picam2.capture_array("main")))
                    if probs is not None:
                        with self._lock:
                            self._observaciones.append((inicio, probs, q))
                    time.sleep(max(0.0, periodo - (time.time() - inicio)))
            except Exception as e:  # noqa: BLE001 - nunca tirar el nodo
                # "Logger severity cannot be changed between calls" no se
                # cura reintentando: libcamera quedo tocado para todo el
                # proceso y cada intento da lo mismo. Antes eso era un
                # mensaje cada 5 s hasta que alguien matara el nodo.
                if "Logger severity" in str(e):
                    self._log(
                        "[cámara] libcamera quedó inservible en este proceso "
                        f"({e}). Se deja de intentar: hay que relanzar el "
                        "nodo. Si pasa al arrancar, suele ser el firmware "
                        "del IMX500 -- ver _yunet_camara._puede_leer_progreso.",
                        warn=True)
                    return
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
        """(probabilidades, calidades) de los cuadros en [t0, t1].

        ANTES devolvia solo la lista de probabilidades; ahora devuelve
        tambien las q, en el mismo orden, para pasarselas a resumir()."""
        with self._lock:
            sel = [(p, q) for t, p, q in self._observaciones if t0 <= t <= t1]
        return [p for p, _ in sel], [q for _, q in sel]

    @staticmethod
    def resumir(cuadros, calidades=None):
        """Resume varios cuadros en UNA observación (paso 0 de PULSE, tesis
        sec. 4.1.4.1), con la MEDIA PONDERADA POR CALIDAD de YuNet:

            m_k = sum_i q(i) * p_k(i) / sum_i q(i)
            p_k = m_k / sum_j m_j

        q(i) en [0, 1] es el score de la cara en el cuadro i. Un cuadro con
        la cara de perfil, borrosa o a medio salir del encuadre tiene un q
        bajo y pesa poco, en vez de valer lo mismo que uno frontal.

        RESPALDO. Si falta alguna q o todas suman 0, se usa la media
        recortada al 15% por cada lado (scipy.stats.trim_mean si esta
        instalado; si no, la misma cuenta en numpy). Como la mediana, eso
        descarta los cuadros sueltos mal clasificados.

        ANTES: m_k = mediana(p_k(1), ..., p_k(N)) y luego la misma
        normalizacion. La mediana protegia de cuadros sueltos mal
        clasificados, pero trataba igual una cara frontal y nitida que una
        de perfil o borrosa. La ponderacion usa el score que YuNet ya
        calculaba, asi que no cuesta nada mas por cuadro.

        Devuelve la lista de 8 probabilidades, o None si no hay cuadros."""
        if not cuadros:
            return None
        import numpy as np
        p = np.asarray(cuadros, dtype=float)
        if p.ndim != 2 or p.shape[0] == 0:
            return None
        m = None
        if calidades is not None and len(calidades) == p.shape[0] \
                and all(q is not None for q in calidades):
            q = np.clip(np.asarray(calidades, dtype=float), 0.0, 1.0)
            suma_q = float(q.sum())
            if np.isfinite(suma_q) and suma_q > 0.0:
                m = (q[:, None] * p).sum(axis=0) / suma_q
        if m is None:
            m = _media_recortada(p, 0.15)
        total = float(m.sum())
        if not np.isfinite(total) or total <= 0.0:
            return None
        return [float(x) / total for x in m]

    def resumen(self, t0, t1):
        """(probabilidades, n_cuadros) de la ventana [t0, t1], en segundos
        de time.time(). (None, 0) si en esa ventana no se vio ninguna cara."""
        cuadros, calidades = self._cuadros_entre(t0, t1)
        return self.resumir(cuadros, calidades), len(cuadros)

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
        cuadros, calidades = [], []
        while time.time() < limite:
            cuadros, calidades = self._cuadros_entre(desde, time.time())
            if len(cuadros) >= CUADROS_JUEGO:
                break
            time.sleep(0.1)
        probs = self.resumir(cuadros, calidades)
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
