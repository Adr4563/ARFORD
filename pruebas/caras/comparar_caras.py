#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""comparar_caras.py -- compara detectores de cara sobre los MISMOS cuadros
grabados con grabar_caras.py (datos/<ANCHOxALTO>/<escenario>/NNN.png).

Cada detector corre en un PROCESO APARTE (RAM y CPU por separado) y con los
MISMOS hilos (2, como ROMEO: HILOS_ONNX=2).

DETECTORES
  yunet_cpu     OpenCV FaceDetectorYN + face_detection_yunet_2023mar.onnx del
                repo, umbral 0.7, entrada = tamano del cuadro (como
                _emotion_detector.py).
  yunet_sensor  salida CRUDA del YuNet int8 que corrio EN el IMX500 al grabar
                (datos/sensor/<escenario>/NNN.npz), decodificada con
                _yunet_camara.decodificar del repo. Su latencia aqui es solo
                decodificar: la red corre en el sensor. Cuadros sin npz = "sin
                dato" (se reportan aparte, no cuentan como "sin cara").
  scrfd_2g5     det_2.5g.onnx (InsightFace, buffalo_m). Solo investigacion NO
                comercial.
  yolov8n_face  yolov8n-face.onnx (akanametov/yolo-face 1.0.0). GPL-3.0.
  yolo11n_face  yolov11n-face.onnx (mismo origen). GPL-3.0.

DOS VISTAS
  1 "Como en uso real": cada detector con su umbral de operacion.
  2 "Igualdad de condiciones": por cada cuadro se guarda el puntaje MAXIMO
    sin filtrar del detector. Con los cuadros de sin_cara se fija, para cada
    detector, el umbral que deja como maximo 1 % (y 5 %) de esos cuadros con
    una caja; con ese umbral se mide la deteccion en los cuadros con cara.
    "Deteccion" = el puntaje maximo del cuadro pasa el umbral (no hay cajas
    de referencia: no distingue una caja bien puesta de una mal puesta; para
    eso estan los mosaicos).

ESTADISTICA
  Intervalo de Wilson del 95 %. Los cuadros seguidos de una pose no son
  independientes: se estima la muestra efectiva n_ef = n(1-r)/(1+r), con r la
  autocorrelacion de lag 1 de la serie detectado/no detectado, y se da
  tambien el intervalo con n_ef. La regla de decision usa el de n_ef.

USO (en la Pi, desde la raiz del repo)
    python3 pruebas/caras/comparar_caras.py --prueba
    python3 pruebas/caras/comparar_caras.py
"""
import argparse
import csv
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
REPO = AQUI.parents[1]
MODELOS_DIR = AQUI / "modelos"
YUNET_ONNX = REPO / "ros2_ws/src/romeo_drivers/romeo_drivers/data/modelos/face_detection_yunet_2023mar.onnx"

DETECTORES = {
    "yunet_cpu":    {"archivo": YUNET_ONNX, "umbral": 0.7, "licencia": "MIT (OpenCV Zoo)"},
    "yunet_sensor": {"archivo": None, "umbral": 0.7,
                     "licencia": "MIT (convertido a .rpk por el equipo)"},
    "scrfd_2g5":    {"archivo": MODELOS_DIR / "det_2.5g.onnx", "umbral": 0.5,
                     "licencia": "InsightFace: solo investigacion NO comercial"},
    "yolov8n_face": {"archivo": MODELOS_DIR / "yolov8n-face.onnx", "umbral": 0.25,
                     "licencia": "GPL-3.0"},
    "yolo11n_face": {"archivo": MODELOS_DIR / "yolov11n-face.onnx", "umbral": 0.25,
                     "licencia": "GPL-3.0"},
}
POR_DEFECTO = list(DETECTORES)
COMPETIDORES = ("scrfd_2g5", "yolov8n_face", "yolo11n_face")
SIN_CARA = "sin_cara"
USO_REAL = ("dist_0.5m", "dist_1m", "dist_1.5m", "dist_2m", "ang_30")
NO_DETECCION = {SIN_CARA, "rojo"}           # escenarios que no miden deteccion
NIVELES_FP = (0.01, 0.05)


# ------------------------------------------------------------------ utiles
def temperatura():
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
    except Exception:  # noqa: BLE001
        return None


def listar_cuadros(datos, max_por_carpeta=None):
    """[(resolucion, escenario, ruta_png)] -- solo carpetas ANCHOxALTO."""
    out = []
    for dres in sorted(p for p in Path(datos).iterdir()
                       if p.is_dir() and re.fullmatch(r"\d+x\d+", p.name)):
        for desc in sorted(p for p in dres.iterdir() if p.is_dir() and p.name != "rojo"):
            pngs = sorted(desc.glob("*.png"))
            if max_por_carpeta:
                pngs = pngs[:max_por_carpeta]
            out += [(dres.name, desc.name, p) for p in pngs]
    return out


def nms(cajas, puntos, iou):
    import cv2
    if not cajas:
        return []
    idx = cv2.dnn.NMSBoxes([list(map(float, c)) for c in cajas],
                           [float(s) for s in puntos], 0.0, iou)
    return [int(i) for i in (idx.flatten() if hasattr(idx, "flatten") else idx)]


def sesion_onnx(ruta, hilos):
    import onnxruntime as ort
    op = ort.SessionOptions()
    op.intra_op_num_threads = hilos
    op.inter_op_num_threads = 1
    return ort.InferenceSession(str(ruta), op, providers=["CPUExecutionProvider"])


def wilson(k, n, z=1.96):
    if not n:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - m), min(1.0, c + m))


def n_efectiva(serie, continua=None):
    """n(1-r)/(1+r) con r = autocorrelacion lag 1 (r<0 se toma como 0).
    Si la serie binaria es constante (p. ej. 100/100 detectados) r no esta
    definido: se usa la serie del puntaje crudo, si la hay."""
    n = len(serie)
    if n < 3:
        return float(n)
    m = sum(serie) / n
    var = sum((x - m) ** 2 for x in serie)
    if var == 0 and continua and len(continua) == n:
        serie, m = continua, sum(continua) / n
        var = sum((x - m) ** 2 for x in serie)
    if var == 0:
        return float(n)                     # constante: r indefinido
    r = sum((serie[i] - m) * (serie[i + 1] - m) for i in range(n - 1)) / var
    r = max(0.0, min(r, 0.99))
    return max(1.0, n * (1 - r) / (1 + r))


def p95(v):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))] if v else None


def pct(x):
    return "-" if x is None else f"{100 * x:.0f}"


def ic(par):
    return "-" if par[0] is None else f"{100 * par[0]:.0f}-{100 * par[1]:.0f}"


# --------------------------------------------------------------- detectores
# __call__(img, ruta) -> cajas [(x,y,w,h,score)] con el umbral de operacion,
#                        o None si no hay dato (solo yunet_sensor).
# max_crudo(img, ruta) -> puntaje maximo SIN umbral del ultimo cuadro (fuera
#                        del tiempo medido).
class YunetCPU:
    def __init__(self, umbral, hilos):
        import cv2
        cv2.setNumThreads(hilos)
        self.det = cv2.FaceDetectorYN_create(str(YUNET_ONNX), "", (320, 320),
                                             score_threshold=umbral)
        # Segunda instancia con umbral ~0 SOLO para el puntaje crudo (vista 2);
        # no entra en la latencia ni en la CPU medidas.
        self.crudo = cv2.FaceDetectorYN_create(str(YUNET_ONNX), "", (320, 320),
                                               score_threshold=0.01)

    def __call__(self, img, ruta):
        alto, ancho = img.shape[:2]
        self.det.setInputSize((ancho, alto))
        _, caras = self.det.detect(img)
        if caras is None:
            return []
        return [(float(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[14])) for c in caras]

    def max_crudo(self, img, ruta):
        alto, ancho = img.shape[:2]
        self.crudo.setInputSize((ancho, alto))
        _, caras = self.crudo.detect(img)
        return 0.0 if caras is None or len(caras) == 0 else float(max(c[14] for c in caras))


class YunetSensor:
    def __init__(self, umbral, hilos):
        sys.path.insert(0, str(REPO / "ros2_ws" / "src" / "romeo_drivers"))
        from romeo_drivers import _yunet_camara as yc
        yc.UMBRAL = umbral
        self.yc, self._max = yc, None

    def __call__(self, img, ruta):
        import numpy as np
        npz = ruta.parents[2] / "sensor" / ruta.parent.name / (ruta.stem + ".npz")
        self._max = None
        if not npz.exists():
            return None
        with np.load(npz) as z:
            s = [z[k] for k in sorted(z.files, key=lambda k: int(k.split("_")[1]))]
        # mismo puntaje que decodificar(): clasificacion x objectness
        self._max = float(max((np.asarray(s[i]).reshape(-1) * np.asarray(s[3 + i]).reshape(-1)).max()
                              for i in range(3)))
        alto, ancho = img.shape[:2]
        c = self.yc.decodificar(s, ancho, alto)
        return [] if c is None else [(c[0], c[1], c[2], c[3], self._max)]

    def max_crudo(self, img, ruta):
        return self._max


class SCRFD:
    """Reimplementa insightface/python-package/insightface/model_zoo/scrfd.py
    (deteccion + NMS; sin puntos de la cara)."""

    def __init__(self, umbral, hilos, lado=640, iou=0.4):
        self.s = sesion_onnx(DETECTORES["scrfd_2g5"]["archivo"], hilos)
        self.entrada = self.s.get_inputs()[0].name
        self.salidas = [o.name for o in self.s.get_outputs()]
        if len(self.salidas) not in (6, 9):
            raise RuntimeError(f"SCRFD con {len(self.salidas)} salidas: formato no reconocido")
        self.fmc, self.strides, self.n_anclas = 3, (8, 16, 32), 2
        self.umbral, self.lado, self.iou = umbral, lado, iou
        self._anclas, self._max = {}, None

    def anclas(self, h, w, stride):
        import numpy as np
        k = (h, w, stride)
        if k not in self._anclas:
            c = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
            c = (c * stride).reshape(-1, 2)
            self._anclas[k] = np.stack([c] * self.n_anclas, axis=1).reshape(-1, 2)
        return self._anclas[k]

    def __call__(self, img, ruta):
        import cv2
        import numpy as np
        alto, ancho = img.shape[:2]
        esc = min(self.lado / alto, self.lado / ancho)
        nh, nw = int(alto * esc), int(ancho * esc)
        lienzo = np.zeros((self.lado, self.lado, 3), np.uint8)
        lienzo[:nh, :nw] = cv2.resize(img, (nw, nh))
        blob = cv2.dnn.blobFromImage(lienzo, 1.0 / 128, (self.lado, self.lado),
                                     (127.5, 127.5, 127.5), swapRB=True)
        outs = self.s.run(self.salidas, {self.entrada: blob})
        cajas, puntos, mx = [], [], 0.0
        for i, st in enumerate(self.strides):
            sc = (outs[i][0] if outs[i].ndim == 3 else outs[i]).reshape(-1)
            mx = max(mx, float(sc.max()))
            bb = (outs[i + self.fmc][0] if outs[i + self.fmc].ndim == 3
                  else outs[i + self.fmc]) * st
            a = self.anclas(self.lado // st, self.lado // st, st)
            for j in np.where(sc >= self.umbral)[0]:
                cx, cy = a[j]
                x1, y1 = cx - bb[j, 0], cy - bb[j, 1]
                x2, y2 = cx + bb[j, 2], cy + bb[j, 3]
                cajas.append((x1 / esc, y1 / esc, (x2 - x1) / esc, (y2 - y1) / esc))
                puntos.append(float(sc[j]))
        self._max = mx
        keep = nms(cajas, puntos, self.iou)
        return [(*cajas[k], puntos[k]) for k in keep]

    def max_crudo(self, img, ruta):
        return self._max


class YOLOFace:
    """ONNX exportado por Ultralytics: entrada 1x3xLxL RGB/255 con letterbox
    (relleno 114). Dos formatos de salida (MEDIDO en los metadatos):
      crudo     (1, 4+1[+kpts], N): cx, cy, w, h, score -> NMS aqui
                (yolov11n-face.onnx: task=detect, salida (1, 5, 8400))
      con NMS   (1, 300, 6+kpts): x1, y1, x2, y2, score, clase, ... ya filtrado
                (yolov8n-face.onnx: task=pose, args 'nms': True, salida
                (1, 300, 21)). Las filas bajo el umbral interno del export
                vienen en 0, asi que su puntaje crudo no baja de ese umbral."""

    def __init__(self, nombre, umbral, hilos, iou=0.7):
        self.s = sesion_onnx(DETECTORES[nombre]["archivo"], hilos)
        i = self.s.get_inputs()[0]
        self.entrada = i.name
        self.lado = i.shape[2] if isinstance(i.shape[2], int) else 640
        self.umbral, self.iou, self._max = umbral, iou, None
        meta = self.s.get_modelmeta().custom_metadata_map
        self.con_nms = "'nms': True" in meta.get("args", "")
        self.meta = {k: meta.get(k) for k in ("task", "version", "license", "description")}

    def __call__(self, img, ruta):
        import cv2
        import numpy as np
        alto, ancho = img.shape[:2]
        esc = min(self.lado / alto, self.lado / ancho)
        nh, nw = int(round(alto * esc)), int(round(ancho * esc))
        px, py = (self.lado - nw) // 2, (self.lado - nh) // 2
        lienzo = np.full((self.lado, self.lado, 3), 114, np.uint8)
        lienzo[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh))
        x = cv2.cvtColor(lienzo, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255
        out = self.s.run(None, {self.entrada: x})[0][0]
        if self.con_nms:                                # (300, 6+kpts), xyxy
            self._max = float(out[:, 4].max())
            cajas = []
            for d in out[out[:, 4] >= self.umbral]:
                x1, y1, x2, y2 = (float(v) for v in d[:4])
                cajas.append(((x1 - px) / esc, (y1 - py) / esc, (x2 - x1) / esc, (y2 - y1) / esc,
                              float(d[4])))
            return cajas
        if out.shape[0] > out.shape[1]:
            out = out.T
        sc = out[4]
        self._max = float(sc.max())
        cajas, puntos = [], []
        for j in np.where(sc >= self.umbral)[0]:
            cx, cy, w, h = out[0, j], out[1, j], out[2, j], out[3, j]
            cajas.append(((cx - w / 2 - px) / esc, (cy - h / 2 - py) / esc, w / esc, h / esc))
            puntos.append(float(sc[j]))
        keep = nms(cajas, puntos, self.iou)
        return [(*cajas[k], puntos[k]) for k in keep]

    def max_crudo(self, img, ruta):
        return self._max


def crear(nombre, hilos):
    u = DETECTORES[nombre]["umbral"]
    if nombre == "yunet_cpu":
        return YunetCPU(u, hilos)
    if nombre == "yunet_sensor":
        return YunetSensor(u, hilos)
    if nombre == "scrfd_2g5":
        return SCRFD(u, hilos)
    return YOLOFace(nombre, u, hilos)


# --------------------------------------------------------------- trabajador
def trabajador(nombre, datos, hilos, max_por_carpeta):
    import cv2
    import psutil
    proc = psutil.Process()
    mb = lambda: proc.memory_info().rss / 1e6  # noqa: E731

    arch = DETECTORES[nombre]["archivo"]
    if arch is not None and not Path(arch).exists():
        print("RESULTADO:" + json.dumps({"detector": nombre, "error": f"no existe {arch}"}))
        return
    ram0, t0 = mb(), time.perf_counter()
    try:
        det = crear(nombre, hilos)
    except Exception as e:  # noqa: BLE001
        print("RESULTADO:" + json.dumps({"detector": nombre, "error": str(e)[:300]}))
        return
    carga = time.perf_counter() - t0
    ram_carga = ram_pico = mb()

    cuadros = listar_cuadros(datos, max_por_carpeta)
    if cuadros:                                         # calentamiento
        img = cv2.imread(str(cuadros[0][2]))
        for _ in range(3):
            det(img, cuadros[0][2])
    temps = [temperatura()]
    cpu_por_res, filas = {}, []
    for k, (res, esc, ruta) in enumerate(cuadros):
        img = cv2.imread(str(ruta))
        c0, t = time.process_time(), time.perf_counter()
        cajas = det(img, ruta)
        lat = time.perf_counter() - t
        cpu = time.process_time() - c0                  # todos los hilos del proceso
        acc = cpu_por_res.setdefault(res, [0.0, 0.0])
        acc[0] += cpu
        acc[1] += lat
        smax = det.max_crudo(img, ruta)                 # fuera de lo medido
        ram_pico = max(ram_pico, mb())
        if k % 50 == 0:
            temps.append(temperatura())
        fila = {"resolucion": res, "escenario": esc, "archivo": ruta.relative_to(datos).as_posix(),
                "lat_ms": round(lat * 1000, 2),
                "max_crudo": None if smax is None else round(float(smax), 4)}
        if cajas is None:
            fila.update(n_cajas=None, ancho=None, score=None, cajas=None)
        else:
            cajas = [tuple(float(v) for v in c) for c in cajas]
            mejor = max(cajas, key=lambda c: c[4], default=None)
            fila.update(n_cajas=len(cajas),
                        ancho=None if mejor is None else round(mejor[2], 1),
                        score=None if mejor is None else round(mejor[4], 3),
                        cajas=[[round(v, 1) for v in c[:4]] for c in cajas])
        filas.append(fila)
    temps.append(temperatura())
    temps = [x for x in temps if x is not None]
    no_cero = [f["max_crudo"] for f in filas if f["max_crudo"]]
    print("RESULTADO:" + json.dumps({
        "meta_modelo": getattr(det, "meta", None), "nms_en_modelo": getattr(det, "con_nms", False),
        "puntaje_min_no_cero": min(no_cero) if no_cero else None,
        "detector": nombre, "umbral": DETECTORES[nombre]["umbral"],
        "licencia": DETECTORES[nombre]["licencia"], "hilos": hilos,
        "carga_s": round(carga, 2), "ram_base_mb": round(ram0),
        "ram_carga_mb": round(ram_carga), "ram_pico_mb": round(ram_pico),
        "cpu_pct_por_res": {r: round(100 * c / max(w, 1e-9), 1) for r, (c, w) in cpu_por_res.items()},
        "temp_ini_c": temps[0] if temps else None, "temp_max_c": max(temps) if temps else None,
        "filas": filas}))


# ------------------------------------------------------------- metricas
def orden(f):
    return int(Path(f["archivo"]).stem)


def vista1(filas):
    """{(res, esc): metricas} con el umbral de operacion."""
    grupos = {}
    for f in filas:
        grupos.setdefault((f["resolucion"], f["escenario"]), []).append(f)
    out = {}
    for key, fs in sorted(grupos.items()):
        fs = sorted(fs, key=orden)
        val = [f for f in fs if f["n_cajas"] is not None]
        serie = [1 if f["n_cajas"] > 0 else 0 for f in val]
        k, n = sum(serie), len(serie)
        nef = n_efectiva(serie, [f["max_crudo"] or 0.0 for f in val])
        anchos = [f["ancho"] for f in val if f["n_cajas"] and f["ancho"] is not None]
        out[key] = {"n": n, "sin_dato": len(fs) - n, "k": k, "tasa": k / n if n else None,
                    "ic": wilson(k, n), "n_ef": round(nef, 1),
                    "ic_nef": wilson(k / n * nef, nef) if n else (None, None),
                    "ancho_med": statistics.median(anchos) if anchos else None,
                    "multi": sum(1 for f in val if f["n_cajas"] > 1)}
    return out


def umbral_para_fp(neg, alfa):
    """Umbral minimo t tal que, como maximo, alfa*n de los puntajes de sin_cara
    son >= t. Devuelve (t, fp_real, degenerado)."""
    if not neg:
        return None, None, False
    s = sorted(neg, reverse=True)
    k = int(math.floor(alfa * len(s)))
    if s[0] <= 0:
        return 1e-6, 0.0, True               # 0 FP a cualquier umbral > 0
    t = s[k] + 1e-6 if k < len(s) else 1e-6
    fp = sum(1 for v in s if v >= t) / len(s)
    return t, fp, False


def vista2(filas):
    """{res: {alfa: {"umbral", "fp", "degenerado", "por_esc": {...}, "uso": {...}}}}"""
    out = {}
    for res in sorted({f["resolucion"] for f in filas}):
        fr = [f for f in filas if f["resolucion"] == res and f["max_crudo"] is not None]
        neg = [f["max_crudo"] for f in fr if f["escenario"] == SIN_CARA]
        out[res] = {"n_sin_cara": len(neg)}
        for alfa in NIVELES_FP:
            t, fp, deg = umbral_para_fp(neg, alfa)
            if t is None:
                continue
            por_esc, uso = {}, [0, 0, 0.0]
            for esc in sorted({f["escenario"] for f in fr} - NO_DETECCION):
                fe = sorted((f for f in fr if f["escenario"] == esc), key=orden)
                serie = [1 if f["max_crudo"] >= t else 0 for f in fe]
                k, n = sum(serie), len(serie)
                nef = n_efectiva(serie, [f["max_crudo"] for f in fe])
                por_esc[esc] = {"n": n, "k": k, "tasa": k / n, "ic": wilson(k, n),
                                "n_ef": round(nef, 1), "ic_nef": wilson(k / n * nef, nef)}
                if esc in USO_REAL:
                    uso[0] += k
                    uso[1] += n
                    uso[2] += nef
            k, n, nef = uso
            out[res][alfa] = {"umbral": round(t, 4), "fp": fp, "degenerado": deg, "por_esc": por_esc,
                              "uso": {"n": n, "k": k, "tasa": k / n if n else None,
                                      "ic": wilson(k, n), "n_ef": round(nef, 1),
                                      "ic_nef": wilson(k / n * nef, nef) if n else (None, None)}}
    return out


def solapan(a, b):
    return not (a[1] < b[0] or b[1] < a[0])


def decision(v2, lat, cpu, temp):
    """Aplica la regla escrita en INFORME_CARAS.md (seccion 2) sobre la vista 2,
    escenarios de uso real agrupados, intervalo con n efectiva."""
    lineas = []
    for res in sorted({r for d in v2.values() for r in d}):
        for alfa in NIVELES_FP:
            def uso(d):
                return v2.get(d, {}).get(res, {}).get(alfa, {}).get("uso")
            y = uso("yunet_cpu")
            comps = [(d, uso(d)) for d in COMPETIDORES if uso(d) and uso(d)["n"]]
            if not y or not y["n"] or not comps:
                continue
            mejor, m = max(comps, key=lambda x: x[1]["tasa"])
            ly, lm = lat.get(("yunet_cpu", res)), lat.get((mejor, res))
            cab = (f"{res}, FP <= {alfa:.0%}: yunet_cpu {pct(y['tasa'])} % [{ic(y['ic_nef'])}] vs "
                   f"{mejor} {pct(m['tasa'])} % [{ic(m['ic_nef'])}]; latencia {ly:.0f} vs {lm:.0f} ms")
            if m["ic_nef"][0] > y["ic_nef"][1]:
                v = (f"{mejor} SUPERA a YuNet sin solaparse. Costo: latencia {lm:.0f} ms (YuNet {ly:.0f}),"
                     f" CPU {cpu.get((mejor, res))} % (YuNet {cpu.get(('yunet_cpu', res))}),"
                     f" temp max {temp.get(mejor)} C (YuNet {temp.get('yunet_cpu')}),"
                     f" licencia {DETECTORES[mejor]['licencia']}. NO se decide el cambio.")
            elif solapan(y["ic_nef"], m["ic_nef"]) or y["tasa"] >= m["tasa"]:
                if ly <= lm:
                    v = ("YuNet SE QUEDA (intervalos se solapan o YuNet arriba, y su latencia es menor o igual)."
                         + (" Solape = EMPATE estadistico." if solapan(y["ic_nef"], m["ic_nef"]) else ""))
                else:
                    v = (f"EMPATE y YuNet es mas lento: desempate por CPU ({cpu.get(('yunet_cpu', res))} vs "
                         f"{cpu.get((mejor, res))} %), temperatura ({temp.get('yunet_cpu')} vs "
                         f"{temp.get(mejor)} C) y licencia (MIT vs {DETECTORES[mejor]['licencia']}).")
            else:
                v = "caso no previsto por la regla: revisar a mano"
            lineas.append(cab + "\n    -> " + v)
    return lineas


def hallazgo_pixeles(v2):
    out = []
    for d, porres in v2.items():
        if "320x240" not in porres or "640x480" not in porres:
            continue
        for alfa in NIVELES_FP:
            a = porres["320x240"].get(alfa, {}).get("uso")
            b = porres["640x480"].get(alfa, {}).get("uso")
            if a and b and a["n"] and b["n"] and b["ic_nef"][0] > a["ic_nef"][1]:
                out.append(f"{d}, FP <= {alfa:.0%}: 640x480 {pct(b['tasa'])} % > 320x240 "
                           f"{pct(a['tasa'])} % sin solaparse -> el limite son los pixeles")
    return out


def mosaicos(resultados, datos, n, carpeta, semilla=0):
    import cv2
    import numpy as np
    ok = [r for r in resultados if "error" not in r]
    if not ok:
        return []
    por_det = {r["detector"]: {f["archivo"]: f for f in r["filas"]} for r in ok}
    archivos = sorted(set.intersection(*[set(d) for d in por_det.values()]))
    random.Random(semilla).shuffle(archivos)
    carpeta.mkdir(parents=True, exist_ok=True)
    hechos = []
    for arch in archivos[:n]:
        img = cv2.imread(str(Path(datos) / arch))
        casillas = []
        for det, fs in por_det.items():
            c, f = img.copy(), fs[arch]
            for b in f["cajas"] or []:
                x, y, w, h = map(int, b)
                cv2.rectangle(c, (x, y), (x + w, y + h), (0, 255, 0), 2)
            et = det + (" (sin dato)" if f["cajas"] is None else
                        f" {f['score']}" if f["score"] is not None else "")
            cv2.rectangle(c, (0, 0), (c.shape[1], 18), (0, 0, 0), -1)
            cv2.putText(c, et, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            casillas.append(c)
        destino = carpeta / arch.replace("/", "__")
        cv2.imwrite(str(destino), np.hstack(casillas))
        hechos.append(destino.name)
    return hechos


# ------------------------------------------------------------- orquestador
def principal():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", default=str(AQUI / "datos"))
    ap.add_argument("--detectores", nargs="+", default=POR_DEFECTO, choices=list(DETECTORES))
    ap.add_argument("--hilos", type=int, default=2, help="igual que ROMEO (HILOS_ONNX=2)")
    ap.add_argument("--prueba", action="store_true", help="solo 5 cuadros por carpeta")
    ap.add_argument("--max-por-carpeta", type=int)
    ap.add_argument("--muestras", type=int, default=20, help="mosaicos para revisar a ojo")
    ap.add_argument("--salida", default=str(AQUI / "resultados" / "caras"))
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    a = ap.parse_args()
    maxc = 5 if a.prueba else a.max_por_carpeta

    if a.worker:
        trabajador(a.worker, Path(a.datos), a.hilos, maxc)
        return

    cuadros = listar_cuadros(a.datos, maxc)
    if not cuadros:
        sys.exit(f"No hay cuadros en {a.datos}")
    print(f"Cuadros: {len(cuadros)} | hilos: {a.hilos} | detectores: {a.detectores}"
          + (" | MODO PRUEBA" if a.prueba else ""))
    env = dict(os.environ, OMP_NUM_THREADS=str(a.hilos))
    resultados = []
    for nombre in a.detectores:
        print(f"\n=== {nombre} (umbral {DETECTORES[nombre]['umbral']}) ===", flush=True)
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", nombre,
               "--datos", a.datos, "--hilos", str(a.hilos)]
        if maxc:
            cmd += ["--max-por-carpeta", str(maxc)]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, env=env)
        linea = next((l for l in p.stdout.splitlines() if l.startswith("RESULTADO:")), None)
        r = (json.loads(linea[10:]) if linea else
             {"detector": nombre, "error": f"el proceso termino sin resultado (codigo {p.returncode})"})
        resultados.append(r)
        print("  ERROR: " + r["error"] if "error" in r else
              f"  carga {r['carga_s']} s | RAM {r['ram_carga_mb']} MB | CPU {r['cpu_pct_por_res']}"
              f" | temp {r['temp_ini_c']} -> {r['temp_max_c']} C")

    ok = [r for r in resultados if "error" not in r]
    v1 = {r["detector"]: vista1(r["filas"]) for r in ok}
    v2 = {r["detector"]: vista2(r["filas"]) for r in ok}
    lat, cpu, temp = {}, {}, {}
    for r in ok:
        temp[r["detector"]] = r["temp_max_c"]
        for res in {f["resolucion"] for f in r["filas"]}:
            ls = [f["lat_ms"] for f in r["filas"] if f["resolucion"] == res]
            lat[(r["detector"], res)] = statistics.median(ls)
            lat[(r["detector"], res, "p95")] = p95(ls)
            cpu[(r["detector"], res)] = r["cpu_pct_por_res"].get(res)

    print("\n\n======================== COSTO ========================")
    print(f"{'detector':13} {'RAM MB':>7} {'temp C':>13}  por resolucion: latencia mediana / p95 ms, CPU %")
    for r in ok:
        d = r["detector"]
        partes = [f"{res}: {lat[(d, res)]:.1f} / {lat[(d, res, 'p95')]:.1f}, {cpu[(d, res)]} %"
                  for res in sorted(r["cpu_pct_por_res"])]
        print(f"{d:13} {r['ram_carga_mb']:>7} {str(r['temp_ini_c']) + '->' + str(r['temp_max_c']):>13}  "
              + "  ".join(partes))
    print("  yunet_sensor: latencia y CPU = solo decodificar; la red corre en el IMX500.")
    for r in ok:
        if r.get("meta_modelo"):
            print(f"  {r['detector']}: metadatos {r['meta_modelo']}")
        if r.get("nms_en_modelo"):
            print(f"  {r['detector']}: NMS dentro del modelo; puntajes bajo su umbral interno salen en 0."
                  f" Minimo no cero visto: {r['puntaje_min_no_cero']} (piso del barrido de la vista 2).")

    print("\n==================== VISTA 1: como en uso real ====================")
    print("tasa % [IC95 Wilson con n | con n_ef] (ancho mediano px); sin_cara = FP")
    for res in sorted({k[0] for g in v1.values() for k in g}):
        print(f"\n--- {res} ---")
        escs = sorted({k[1] for g in v1.values() for k in g if k[0] == res})
        for esc in escs:
            print(f"  {esc}")
            for d, g in v1.items():
                m = g.get((res, esc))
                if not m:
                    continue
                an = "-" if m["ancho_med"] is None else f"{m['ancho_med']:.0f}"
                extra = f", sin dato {m['sin_dato']}" if m["sin_dato"] else ""
                print(f"    {d:13} {'FP ' if esc == SIN_CARA else '   '}{pct(m['tasa']):>3} % "
                      f"[{ic(m['ic'])} | {ic(m['ic_nef'])}] ({an} px) n={m['n']} n_ef={m['n_ef']}{extra}")

    print("\n============ VISTA 2: igualdad de condiciones (barrido) ============")
    for res in sorted({r for g in v2.values() for r in g}):
        for alfa in NIVELES_FP:
            print(f"\n--- {res}, FP en sin_cara <= {alfa:.0%} ---")
            for d, g in v2.items():
                m = g.get(res, {}).get(alfa)
                if not m:
                    continue
                u = m["uso"]
                print(f"  {d:13} umbral {m['umbral']:<8} FP real {pct(m['fp'])} % "
                      f"(n sin_cara {g[res]['n_sin_cara']})  USO REAL {pct(u['tasa'])} % "
                      f"[{ic(u['ic'])} | {ic(u['ic_nef'])}] n={u['n']} n_ef={u['n_ef']}"
                      + ("  ** 0 FP a cualquier umbral: criterio degenerado **" if m["degenerado"] else ""))
                print("      " + "  ".join(f"{e} {pct(x['tasa'])}" for e, x in m["por_esc"].items()))

    print("\n===================== REGLA DE DECISION =====================")
    for l in decision(v2, lat, cpu, temp) or ["(sin datos suficientes)"]:
        print("  " + l)
    for l in hallazgo_pixeles(v2):
        print("  HALLAZGO: " + l)
    for r in resultados:
        if "error" in r:
            print(f"\n{r['detector']}: ERROR -> {r['error']}")

    salida = Path(a.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    hechos = mosaicos(resultados, a.datos, a.muestras, salida.parent / "mosaicos")
    with open(f"{salida}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["detector", "resolucion", "escenario", "archivo", "n_cajas", "ancho_px",
                    "score", "max_crudo", "lat_ms", "cajas"])
        for r in ok:
            for f in r["filas"]:
                w.writerow([r["detector"], f["resolucion"], f["escenario"], f["archivo"], f["n_cajas"],
                            f["ancho"], f["score"], f["max_crudo"], f["lat_ms"], json.dumps(f["cajas"])])
    with open(f"{salida}.json", "w", encoding="utf-8") as fh:
        json.dump({"config": vars(a),
                   "costo": {f"{k[0]}|{k[1]}": v for k, v in lat.items() if len(k) == 2},
                   "vista1": {d: {f"{k[0]}/{k[1]}": m for k, m in g.items()} for d, g in v1.items()},
                   "vista2": {d: {res: {str(al): m for al, m in x.items()} for res, x in g.items()}
                              for d, g in v2.items()},
                   "decision": decision(v2, lat, cpu, temp), "hallazgo_pixeles": hallazgo_pixeles(v2),
                   "resultados": resultados}, fh, ensure_ascii=False, indent=1, default=str)
    print(f"\nGuardado: {salida}.csv, {salida}.json y {len(hechos)} mosaicos en {salida.parent / 'mosaicos'}")


if __name__ == "__main__":
    principal()
