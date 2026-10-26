# -*- coding: utf-8 -*-
"""YuNet corriendo DENTRO de la camara (IMX500), no en la CPU de la Pi.

POR QUE. Medido en la Pi, cada cuadro costaba 332 ms: 174 ms de YuNet
buscando la cara y 157 ms de FER+ clasificandola. A 5 fps eso son 1,66 s
de CPU por segundo de reloj, o sea mas de un nucleo entero solo para la
camara (el nodo llegaba al 347% antes de limitar los hilos).

El sensor IMX500 lleva un acelerador que ejecuta la red el mismo: la Pi
recibe las coordenadas ya calculadas y no gasta CPU en buscarlas. FER+ se
queda en la CPU, porque el chip solo ejecuta UN modelo a la vez y los dos
van encadenados (YuNet encuentra la cara -> FER+ la clasifica).

EL MODELO. /usr/share/imx500-models/imx500_network_yunet.rpk, convertido
desde el mismo face_detection_yunet_2023mar.onnx que usa la CPU. Las dos
capas Resize del original traen la escala como tensor (dinamica) aunque
siempre valga x2; se reemplazaron por un upsample fijo para poder
cuantizar a int8, que es lo unico que el acelerador ejecuta.

LA SALIDA son 12 tensores, tres escalas (stride 8, 16 y 32) x cuatro
cosas: clasificacion, objectness, caja y 5 puntos de la cara. Aqui solo
se usan las cajas: el recorte que FER+ necesita.
"""
import os

import numpy as np

RPK = os.environ.get(
    "ROMEO_YUNET_RPK", "/usr/share/imx500-models/imx500_network_yunet.rpk")

# El modelo se entreno a 640x640 y los strides son los de YuNet.
LADO = 640
STRIDES = (8, 16, 32)

# Minimo para dar una cara por buena. El de la version CPU es 0.7 sobre
# la puntuacion ya combinada; aqui se compara contra lo mismo
# (clasificacion x objectness), asi que se mantiene.
UMBRAL = float(os.environ.get("ROMEO_YUNET_UMBRAL", "0.7"))


def _anclas(stride):
    """Centros de celda de una escala, en pixeles de la imagen 640x640."""
    n = LADO // stride
    ys, xs = np.mgrid[0:n, 0:n]
    return np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32) * stride


_ANCLAS = {s: _anclas(s) for s in STRIDES}


def decodificar(salidas, ancho, alto):
    """La mejor cara como (x, y, w, h) en pixeles de la imagen real.

    `salidas` son los 12 tensores de IMX500.get_outputs(). Devuelve None
    si ninguna cara pasa UMBRAL."""
    if not salidas or len(salidas) < 9:
        return None

    mejor, mejor_p = None, 0.0
    for i, stride in enumerate(STRIDES):
        cls = np.asarray(salidas[i]).reshape(-1)
        obj = np.asarray(salidas[3 + i]).reshape(-1)
        caja = np.asarray(salidas[6 + i]).reshape(-1, 4)
        n = min(len(cls), len(obj), len(caja), len(_ANCLAS[stride]))
        if n == 0:
            continue
        # YuNet multiplica las dos cabezas; con una sola no basta, porque
        # la clasificacion sola da ~0.5 en celdas vacias.
        puntos = cls[:n] * obj[:n]
        k = int(puntos.argmax())
        if puntos[k] <= mejor_p:
            continue
        mejor_p = float(puntos[k])
        cx, cy = _ANCLAS[stride][k]
        dx, dy, dw, dh = caja[k]
        # Centro desplazado por el stride; tamaño exponencial, como YuNet.
        w = float(np.exp(dw) * stride)
        h = float(np.exp(dh) * stride)
        mejor = (cx + dx * stride - w / 2.0, cy + dy * stride - h / 2.0, w, h)

    if mejor is None or mejor_p < UMBRAL:
        return None

    # De 640x640 a la resolucion real del cuadro.
    ex, ey = ancho / LADO, alto / LADO
    x, y, w, h = mejor
    return (int(max(0, x * ex)), int(max(0, y * ey)),
            int(w * ex), int(h * ey))


class YunetCamara:
    """Envuelve el IMX500. Carga perezosa: el firmware tarda en subir."""

    def __init__(self, rpk=None, log=None):
        self._rpk = rpk or RPK
        self._log = log or (lambda *a, **k: None)
        self._imx = None
        self.error = None

    @property
    def listo(self):
        return self._imx is not None

    @property
    def camera_num(self):
        """Indice de camara que Picamera2 tiene que abrir, o None."""
        return getattr(self._imx, "camera_num", None) if self._imx else None

    def cargar(self):
        """True si el firmware quedo en el sensor. Nunca lanza: sin esto
        el detector sigue con YuNet en la CPU."""
        if self._imx is not None:
            return True
        if not os.path.exists(self._rpk):
            self.error = f"no existe {self._rpk}"
            return False
        try:
            from picamera2.devices import IMX500
            self._imx = IMX500(self._rpk)
            self._log(f"[cámara] YuNet en el sensor ({self._rpk})")
            return True
        except Exception as e:                               # noqa: BLE001
            self.error = str(e)[:200]
            self._imx = None
            return False

    def cara(self, metadata, ancho, alto):
        """(x, y, w, h) de la cara en este cuadro, o None."""
        if self._imx is None:
            return None
        try:
            salidas = self._imx.get_outputs(metadata, add_batch=True)
        except Exception:                                    # noqa: BLE001
            return None
        return decodificar(salidas, ancho, alto)
