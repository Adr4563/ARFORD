#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""efecto_color.py -- mide cuanto cambia YuNet (CPU) y el FER si el cuadro
llega con los canales rojo y azul intercambiados, como le pasa a ROMEO
(pide "BGR888" a picamera2, que entrega R,G,B, y lo trata como BGR).

Usa los cuadros ya grabados (datos/<res>/<escenario>/*.png, guardados en BGR
correcto). Para cada cuadro con cara:
  correcto   = el PNG tal cual (BGR real)
  invertido  = img[..., ::-1]  (lo que ROMEO le pasa hoy a YuNet y al FER)
La caja de la cara sale de YuNet sobre el cuadro CORRECTO y se usa la MISMA en
las dos versiones, para aislar el efecto del color en el clasificador.

Preproceso de cada FER copiado de _emotion_detector.py (_clasificar), sin
importarlo ni cambiarlo:
  ferplus        gris 64x64 crudo, (1,1,64,64), 8 clases (orden ETIQUETAS)
  deepface       gris 48x48 /255, (1,48,48,1), 7 clases
  hsemotion      color 224x224, /255, BGR->RGB, norm ImageNet, (1,3,N,N), 8 clases
  mobilefacenet  igual que hsemotion con N=112 (se usan las 8 primeras salidas)

USO (en la Pi, desde la raiz del repo)
    python3 pruebas/caras/efecto_color.py --res 320x240
"""
import argparse
import json
import os
import statistics
from pathlib import Path

AQUI = Path(__file__).resolve().parent
REPO = AQUI.parents[1]
DRV = REPO / "ros2_ws/src/romeo_drivers/romeo_drivers"
YUNET = DRV / "data/modelos/face_detection_yunet_2023mar.onnx"
ETIQUETAS = ["neutral", "felicidad", "sorpresa", "tristeza", "enojo", "asco", "miedo", "desprecio"]
HSE = ["enojo", "desprecio", "asco", "miedo", "felicidad", "neutral", "tristeza", "sorpresa"]
DEEP = ["enojo", "asco", "miedo", "felicidad", "tristeza", "sorpresa", "neutral"]
EXTRA = Path.home() / "ARFORD/pruebas/fer_deepface/modelos"
MODELOS = {
    "ferplus": [REPO / "ros2_ws/install/romeo_drivers/share/romeo_drivers/modelos/emotion-ferplus-8.onnx",
                DRV / "data/modelos/emotion-ferplus-8.onnx"],
    "deepface": [EXTRA / "deepface_emotion.onnx"],
    "hsemotion": [EXTRA / "hsemotion_b0.onnx"],
    "mobilefacenet": [EXTRA / "mobilefacenet_emotion.onnx"],
}


def softmax(x):
    import numpy as np
    e = np.exp(x - np.max(x))
    return e / e.sum()


class FER:
    def __init__(self, nombre, ruta, hilos=2):
        import onnxruntime as ort
        op = ort.SessionOptions()
        op.intra_op_num_threads, op.inter_op_num_threads = hilos, 1
        self.s = ort.InferenceSession(str(ruta), op, providers=["CPUExecutionProvider"])
        self.i, self.nombre = self.s.get_inputs()[0].name, nombre

    def __call__(self, bgr, caja):
        """8 probabilidades en el orden ETIQUETAS. `bgr` es lo que el FER
        CREE que es BGR (en la version invertida en realidad es RGB)."""
        import cv2
        import numpy as np
        x, y, w, h = caja
        if self.nombre in ("hsemotion", "mobilefacenet"):
            lado = 112 if self.nombre == "mobilefacenet" else 224
            r = bgr[y:y + h, x:x + w]
            c = cv2.resize(r, (lado, lado)).astype(np.float32) / 255.0
            c = c[:, :, ::-1]
            c = (c - np.array((0.485, 0.456, 0.406), np.float32)) / np.array((0.229, 0.224, 0.225), np.float32)
            p8 = softmax(self.s.run(None, {self.i: c.transpose(2, 0, 1)[None].astype(np.float32)})[0][0][:8])
            p = [0.0] * 8
            for k, e in enumerate(HSE):
                p[ETIQUETAS.index(e)] = float(p8[k])
            return p
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)[y:y + h, x:x + w]
        if self.nombre == "deepface":
            c = cv2.resize(g, (48, 48)).astype(np.float32) / 255.0
            p7 = softmax(self.s.run(None, {self.i: c.reshape(1, 48, 48, 1)})[0][0])
            p = [0.0] * 8
            for k, e in enumerate(DEEP):
                p[ETIQUETAS.index(e)] = float(p7[k])
            return p
        c = cv2.resize(g, (64, 64)).astype(np.float32)
        return [float(v) for v in softmax(self.s.run(None, {self.i: c.reshape(1, 1, 64, 64)})[0][0])]


def main():
    import cv2
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", default=str(AQUI / "datos"))
    ap.add_argument("--res", default="320x240")
    ap.add_argument("--salida", default=str(AQUI / "resultados" / "efecto_color.json"))
    a = ap.parse_args()
    cv2.setNumThreads(2)
    det = cv2.FaceDetectorYN_create(str(YUNET), "", (320, 320), score_threshold=0.01)
    fers = {}
    for n, rutas in MODELOS.items():
        r = next((p for p in rutas if p.exists()), None)
        print(f"FER {n}: {r if r else 'no encontrado, se omite'}")
        if r:
            fers[n] = FER(n, r)

    base = Path(a.datos) / a.res
    escs = sorted(p.name for p in base.iterdir() if p.is_dir() and p.name not in ("sin_cara", "rojo"))
    yunet = {"correcto": [], "invertido": []}
    fer = {n: {"correcto": [], "invertido": []} for n in fers}
    por_esc = {}
    for esc in escs:
        for png in sorted((base / esc).glob("*.png")):
            img = cv2.imread(str(png))
            inv = img[..., ::-1].copy()
            h, w = img.shape[:2]
            det.setInputSize((w, h))
            res = {}
            for nombre, im in (("correcto", img), ("invertido", inv)):
                _, caras = det.detect(im)
                mejor = None if caras is None or len(caras) == 0 else max(caras, key=lambda c: c[14])
                res[nombre] = mejor
                yunet[nombre].append(0.0 if mejor is None else float(mejor[14]))
            pe = por_esc.setdefault(esc, {"n": 0, "det_ok": 0, "det_inv": 0})
            pe["n"] += 1
            pe["det_ok"] += int(yunet["correcto"][-1] >= 0.7)
            pe["det_inv"] += int(yunet["invertido"][-1] >= 0.7)
            if res["correcto"] is None or res["correcto"][14] < 0.7:
                continue
            caja = [max(0, int(v)) for v in res["correcto"][:4]]
            for n, f in fers.items():
                fer[n]["correcto"].append(f(img, caja))
                fer[n]["invertido"].append(f(inv, caja))

    out = {"res": a.res, "escenarios": por_esc, "yunet": {}, "fer": {}}
    n = len(yunet["correcto"])
    for k in ("correcto", "invertido"):
        v = yunet[k]
        out["yunet"][k] = {"n": n, "detecta_0.7": sum(x >= 0.7 for x in v) / max(n, 1),
                           "puntaje_mediano": statistics.median(v) if v else None}
    print(f"\nYuNet CPU ({a.res}, {n} cuadros con cara):")
    for k, m in out["yunet"].items():
        print(f"  {k:9}  detecta (>=0.7) {100 * m['detecta_0.7']:.1f} %   puntaje mediano {m['puntaje_mediano']:.3f}")
    for nf, d in fer.items():
        ok, inv = d["correcto"], d["invertido"]
        if not ok:
            continue
        am = lambda p: max(range(8), key=lambda i: p[i])  # noqa: E731
        cambia = sum(am(x) != am(y) for x, y in zip(ok, inv)) / len(ok)
        tv = statistics.mean(0.5 * sum(abs(x[i] - y[i]) for i in range(8)) for x, y in zip(ok, inv))
        dist = {k: {e: sum(am(p) == i for p in v) / len(v) for i, e in enumerate(ETIQUETAS)}
                for k, v in (("correcto", ok), ("invertido", inv))}
        media = {k: {e: statistics.mean(p[i] for p in v) for i, e in enumerate(ETIQUETAS)}
                 for k, v in (("correcto", ok), ("invertido", inv))}
        out["fer"][nf] = {"n": len(ok), "cambia_emocion_principal": cambia, "distancia_tv_media": tv,
                          "distribucion_argmax": dist, "prob_media": media}
        print(f"\nFER {nf}: {len(ok)} caras | la emocion principal cambia en {100 * cambia:.1f} % de los cuadros"
              f" | distancia TV media {tv:.3f}")
        print("  emocion     " + "".join(f"{e[:8]:>10}" for e in ETIQUETAS))
        for k in ("correcto", "invertido"):
            print(f"  {k:10}  " + "".join(f"{100 * dist[k][e]:>9.1f}%" for e in ETIQUETAS))
    Path(a.salida).parent.mkdir(parents=True, exist_ok=True)
    Path(a.salida).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado: {a.salida}")


if __name__ == "__main__":
    main()
