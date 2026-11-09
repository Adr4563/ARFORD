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

# Hilos para onnxruntime y OpenCV. Ver la nota en _asegurar_modelos_cargados.
HILOS_ONNX = int(os.environ.get("ROMEO_FER_HILOS", "2"))

# Buscar la cara dentro del sensor en vez de en la CPU (ver _yunet_camara).
# ROMEO_YUNET_CAMARA=0 lo desactiva y todo vuelve a la CPU.
YUNET_EN_CAMARA = os.environ.get("ROMEO_YUNET_CAMARA", "1") != "0"

from ._yunet_camara import YunetCamara  # noqa: E402

_MAX_INTENTOS = 15
_PAUSA_ENTRE_INTENTOS = 0.2

# Modo continuo.
FPS_POR_DEFECTO = 2.0
HISTORIAL_SEG = 60.0       # cuántos segundos de cuadros se guardan
ESPERA_JUEGO_SEG = 4.0     # cuánto espera el juego de caras a ver una cara
CUADROS_JUEGO = 3          # con cuántos cuadros decide el juego de caras

# Tamaño del cuadro que se analiza. Antes la vista previa por defecto
# (640x480): YuNet buscaba la cara en el doble de pixeles de lo necesario
# para una persona a 1-2 m del robot, y FER+ igual recorta a 64x64.
RESOLUCION_POR_DEFECTO = (320, 240)
# Formato de captura. Por defecto picamera2 da XBGR8888 (RGBA, 4 canales),
# que obliga a convertir RGBA->BGR cada cuadro antes de YuNet. Pidiendo
# "BGR888" (3 canales) la camara entrega ya el formato que usa OpenCV, y
# _a_bgr no copia ni convierte nada: menos CPU por cuadro (el nodo FER es
# lo que mas calienta la Pi). YuNet y FER trabajan en BGR/gris igual.
FORMATO_CAPTURA = "BGR888"
# En reposo (bajo demanda, sin voz) el hilo solo mira el reloj cada tanto.
PAUSA_REPOSO = 0.1


class DetectorEmocion:
    """Estado del detector (sesión ONNX, detector de caras y, en modo
    continuo, la cámara y el historial). Se inicializa perezosamente: crear
    el objeto en on_configure() no toca nada."""

    def __init__(self, logger=None, fps=FPS_POR_DEFECTO, bajo_demanda=True,
                 resolucion=RESOLUCION_POR_DEFECTO):
        self.logger = logger
        # BAJO DEMANDA: la camara queda abierta, pero YuNet y FER+ solo
        # corren mientras alguien llama a activar() -- el communication_node
        # avisa cuando el microfono oye voz, y el juego de caras activa por
        # su cuenta. Antes clasificaba 5 cuadros por segundo SIEMPRE, aunque
        # nadie hablara, y el orquestador solo usa los de cada frase.
        self.bajo_demanda = bool(bajo_demanda)
        self.resolucion = tuple(int(v) for v in resolucion)
        self._activo_hasta = 0.0
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
        """Lleva el cuadro a BGR, el formato que esperan YuNet y FER.

        Con FORMATO_CAPTURA="BGR888" la camara ya entrega 3 canales en orden
        BGR: no hay nada que convertir y se devuelve tal cual (ahorra una
        conversion por cuadro). Si llegara RGBA (4 canales, formato viejo) o
        RGB, se convierte como antes."""
        import cv2
        canales = frame.shape[-1]
        if canales == 4:
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        if canales == 3 and FORMATO_CAPTURA != "BGR888":
            # 3 canales pero no pedimos BGR: el driver dio RGB, hay que girar.
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame  # ya es BGR888: sin conversion

    def _clasificar(self, frame_bgr, caja=None):
        """Las 8 probabilidades de FER+ (orden ETIQUETAS) para la cara más
        clara del cuadro, o None si no hay cara.

        `caja` es (x, y, w, h) cuando la cara YA la encontro el sensor
        (ver _yunet_camara.py): entonces no se corre YuNet en la CPU, que
        es el 52% del coste por cuadro. Sin caja, se busca aqui como
        siempre."""
        import cv2
        import numpy as np

        if caja is not None:
            x, y, w, h = (max(0, int(v)) for v in caja)
        else:
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

    def activar(self, segundos):
        """Clasifica cuadros durante los proximos `segundos` (bajo demanda).
        Llamarlo otra vez antes de que venza alarga la ventana."""
        with self._lock:
            self._activo_hasta = max(self._activo_hasta, time.time() + float(segundos))

    def _toca_clasificar(self):
        if not self.bajo_demanda:
            return True
        with self._lock:
            return time.time() < self._activo_hasta

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
                picam2.start(picam2.create_preview_configuration(
                    main={"size": self.resolucion, "format": FORMATO_CAPTURA}),
                    show_preview=False)
                time.sleep(2)  # que asiente la exposición
                donde = "sensor" if sensor is not None else "CPU"
                modo = "solo con voz" if self.bajo_demanda else "siempre"
                self._log(f"[cámara] FER a {self.fps:g} fps, {modo}, "
                          f"{self.resolucion[0]}x{self.resolucion[1]} (cara: {donde})")
                periodo = 1.0 / self.fps
                # Cuadros clasificados en la ventana activa en curso: al
                # cerrarse se loguean los fps REALES, para comparar.
                ventana_inicio, ventana_cuadros = None, 0
                while self._continuo:
                    if not self._toca_clasificar():
                        if ventana_inicio is not None:
                            dur = time.time() - ventana_inicio
                            self._log(f"[cámara] FER: {ventana_cuadros} cuadros en "
                                      f"{dur:.1f} s ({ventana_cuadros / max(dur, 0.1):.1f} fps reales)")
                            ventana_inicio, ventana_cuadros = None, 0
                        time.sleep(PAUSA_REPOSO)
                        continue
                    inicio = time.time()
                    if ventana_inicio is None:
                        ventana_inicio = inicio
                    ventana_cuadros += 1
                    if sensor is not None:
                        # La cara la busca el acelerador del IMX500; aqui
                        # solo se recorta y se clasifica.
                        pedido = picam2.capture_request()
                        try:
                            cuadro = pedido.make_array("main")
                            alto, ancho = cuadro.shape[:2]
                            caja = sensor.cara(pedido.get_metadata(), ancho, alto)
                        finally:
                            pedido.release()
                        probs = (self._clasificar(self._a_bgr(cuadro), caja)
                                 if caja is not None else None)
                    else:
                        probs = self._clasificar(self._a_bgr(picam2.capture_array("main")))
                    if probs is not None:
                        with self._lock:
                            self._observaciones.append((inicio, probs))
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
        # En el juego la persona pone la cara sin hablar: nadie mas activa
        # la camara, asi que se activa aqui lo que dura la espera.
        self.activar(ESPERA_JUEGO_SEG)
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
            picam2.start(picam2.create_preview_configuration(
                main={"size": self.resolucion, "format": FORMATO_CAPTURA}),
                show_preview=False)
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
