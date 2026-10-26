#!/usr/bin/env python3
"""FER: emocion de la cara con la camara de la Pi, a 5 cuadros por segundo.

    python3 fer.py                 # ventanas de 3 s, Ctrl+C para salir
    python3 fer.py --ventana 5     # ventanas de 5 s
    python3 fer.py --fps 2         # como lo tiene hoy el robot (fer_fps: 2.0)

Hace lo mismo que la tesis de Rocha (sec. 4.1.4.1): clasifica cada cuadro y,
al cerrar la ventana de percepcion, la resume en UNA observacion con la
mediana de cada emocion normalizada. Despues aplica lo de PULSE: confianza
por entropia (ec. 4.1-4.2) y conversion a VAD (ec. 4.3).

Diferencias con la tesis: alla DeepFace en una laptop; aca los mismos
modelos del robot (YuNet para encontrar la cara + FER+ para la emocion,
ONNX en la CPU de la Pi), que dan 8 clases en vez de 7.

Independiente del robot: no importa ROS ni romeo_drivers. Si el robot esta
corriendo, visualization_faces_node tiene la camara: paralo antes.
"""

import argparse
import math
import os
import sys
import time

import numpy as np

# Orden de salida de FER+ (igual que _emotion_detector.ETIQUETAS).
ETIQUETAS = ["neutral", "felicidad", "sorpresa", "tristeza", "enojo", "asco",
             "miedo", "desprecio"]
# Anclajes VAD de De Bruyne et al. (2020), los de la tesis. "desprecio" no
# esta en el paper: es la coordenada propia que usa estado_afectivo.py.
ANCLAJES = {
    "neutral": (0.00, 0.00, 0.00), "felicidad": (0.76, 0.48, 0.35),
    "sorpresa": (0.40, 0.67, -0.13), "tristeza": (-0.63, -0.27, -0.33),
    "enojo": (-0.51, 0.59, 0.25), "asco": (-0.60, 0.35, 0.11),
    "miedo": (-0.64, 0.60, -0.43), "desprecio": (-0.55, 0.20, 0.30),
}

AQUI = os.path.dirname(os.path.abspath(__file__))
# Los .onnx viven en el paquete del robot: ros2_ws/src/... en el repo y
# src/... en la Pi (alli ~/ros2_ws ES el repo).
CANDIDATOS = [os.path.normpath(os.path.join(AQUI, r)) for r in (
    "../../ros2_ws/src/romeo_drivers/romeo_drivers/data/modelos",
    "../../src/romeo_drivers/romeo_drivers/data/modelos")]


def _dir_modelos():
    for d in CANDIDATOS:
        if os.path.exists(os.path.join(d, "emotion-ferplus-8.onnx")):
            return d
    sys.exit(f"No encuentro emotion-ferplus-8.onnx en {CANDIDATOS}")


def confianza(p):
    """1 - entropia normalizada (tesis ec. 4.1-4.2)."""
    h = -sum(x * math.log(x) for x in p if x > 0) / math.log(len(p))
    return 1.0 - h


def a_vad(p):
    """Suma ponderada de anclajes (tesis ec. 4.3)."""
    return tuple(sum(pi * ANCLAJES[e][k] for pi, e in zip(p, ETIQUETAS)) for k in range(3))


class Fer:

    def __init__(self, hilos=2):
        import cv2
        import onnxruntime as ort
        d = _dir_modelos()
        op = ort.SessionOptions()
        op.intra_op_num_threads = hilos
        self._sesion = ort.InferenceSession(os.path.join(d, "emotion-ferplus-8.onnx"), op)
        self._entrada = self._sesion.get_inputs()[0].name
        self._caras = cv2.FaceDetectorYN_create(
            os.path.join(d, "face_detection_yunet_2023mar.onnx"), "", (320, 320),
            score_threshold=0.7)

    def clasificar(self, bgr):
        """Las 8 probabilidades de la cara mas clara del cuadro, o None."""
        import cv2
        alto, ancho = bgr.shape[:2]
        self._caras.setInputSize((ancho, alto))
        _, caras = self._caras.detect(bgr)
        if caras is None or len(caras) == 0:
            return None
        x, y, w, h = [max(0, int(v)) for v in max(caras, key=lambda c: c[14])[:4]]
        recorte = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)[y:y + h, x:x + w]
        if recorte.size == 0:
            return None
        recorte = cv2.resize(recorte, (64, 64)).astype(np.float32)
        z = self._sesion.run(None, {self._entrada: recorte.reshape(1, 1, 64, 64)})[0][0]
        e = np.exp(z - z.max())
        return e / e.sum()


def resumir(cuadros):
    """Mediana por emocion, normalizada (tesis sec. 4.1.4.1)."""
    m = np.median(np.asarray(cuadros), axis=0)
    return m / m.sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--ventana", type=float, default=3.0, help="segundos por ventana")
    ap.add_argument("--hilos", type=int, default=2)
    args = ap.parse_args()

    import cv2
    from picamera2 import Picamera2

    fer = Fer(args.hilos)
    cam = Picamera2()
    cam.start(cam.create_preview_configuration(), show_preview=False)
    time.sleep(2)  # que asiente la exposicion
    print(f"FER a {args.fps:g} fps, ventanas de {args.ventana:g} s. "
          f"Mira a la camara y pon caras. Ctrl+C para salir.\n", flush=True)
    periodo = 1.0 / args.fps
    try:
        while True:
            cuadros, tomados, t_proc = [], 0, []
            fin = time.time() + args.ventana
            while time.time() < fin:
                t0 = time.time()
                frame = cam.capture_array("main")
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR if frame.shape[-1] == 4
                                   else cv2.COLOR_RGB2BGR)
                p = fer.clasificar(bgr)
                t_proc.append(time.time() - t0)
                tomados += 1
                if p is not None:
                    cuadros.append(p)
                time.sleep(max(0.0, periodo - (time.time() - t0)))
            real = tomados / args.ventana
            if not cuadros:
                print(f"  sin cara en {tomados} cuadros ({real:.1f} fps reales)", flush=True)
                continue
            r = resumir(cuadros)
            orden = sorted(zip(ETIQUETAS, r), key=lambda kv: -kv[1])[:3]
            v, a, d = a_vad(r)
            print(f"  {', '.join(f'{e} {x:.0%}' for e, x in orden)}\n"
                  f"      {len(cuadros)}/{tomados} cuadros con cara, {real:.1f} fps reales "
                  f"(pedidos {args.fps:g}), {np.mean(t_proc) * 1000:.0f} ms por cuadro\n"
                  f"      confianza {confianza(r):.2f}, VAD ({v:+.2f}, {a:+.2f}, {d:+.2f})",
                  flush=True)
    except KeyboardInterrupt:
        print()
    finally:
        cam.stop()
        cam.close()  # sin close() la siguiente Picamera2() no adquiere la camara
    return 0


if __name__ == "__main__":
    sys.exit(main())
