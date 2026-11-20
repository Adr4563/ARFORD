#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""grabar_caras.py -- graba cuadros de la AI Camera (IMX500) por escenario,
para comparar detectores de cara sobre las MISMAS imagenes.

Cada cuadro sale de UN pedido de la camara con dos flujos del mismo instante:
    main  640x480 (BGR888)   ->  datos/640x480/<escenario>/NNN.png
    lores 320x240 (YUV420)   ->  datos/320x240/<escenario>/NNN.png
Asi las dos resoluciones tienen exactamente la misma pose. Ademas:
    datos/sensor/<escenario>/NNN.npz   los 12 tensores CRUDOS del YuNet que
                                       corre en el IMX500 (sin umbral: sirven
                                       para barrer umbrales despues)
    datos/meta/<escenario>.jsonl       metadatos de la camara por cuadro
                                       (ExposureTime, AnalogueGain, ColourGains...)
El IMX500 solo procesa lo que ve en vivo y adjunta su salida a ~1 de cada 2
cuadros (la red va a la mitad del ritmo de la camara). Se toma el siguiente
pedido que la traiga (max 6); si no llega, el cuadro se guarda SIN npz y
comparar_caras.py lo cuenta aparte ("sin dato"), no como "sin cara".

COLOR: picamera2 sigue los nombres de libcamera: "BGR888" deja los bytes en
orden R,G,B (MEDIDO: la cara salia azul). Aqui se convierte a BGR real antes
de guardar. ROMEO no lo hace (ver INFORME_CARAS.md).

ANTES DE CORRER: detener los nodos de ROMEO que usan la camara.

USO (en la Pi, desde la raiz del repo)
    python3 pruebas/caras/grabar_caras.py --escenarios dist_1m sin_cara
    python3 pruebas/caras/grabar_caras.py --servidor   # lee escenarios por stdin,
                                                       # uno por linea; "q" sale
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
REPO = AQUI.parents[1]

# Una variable a la vez; la base es 1 m, frontal, luz normal.
ESCENARIOS = {
    "dist_0.5m": "a 0,5 m, frontal, luz normal",
    "dist_1m":   "a 1 m, frontal, luz normal (base)",
    "dist_1.5m": "a 1,5 m, frontal, luz normal",
    "dist_2m":   "a 2 m, frontal, luz normal",
    "dist_3m":   "a 3 m, frontal, luz normal (si hay espacio)",
    "ang_30":    "a 1 m, cara girada ~30 grados",
    "ang_60":    "a 1 m, cara girada ~60 grados",
    "luz_baja":  "a 1 m, frontal, luz del cuarto baja",
    "lentes_si": "a 1 m, frontal, CON lentes",
    "lentes_no": "a 1 m, frontal, la misma persona SIN lentes",
    "sin_cara":  "escena SIN caras, fondos variados",
    "rojo":      "objeto rojo llenando el centro (prueba de orden de color)",
}
METADATOS = ("ExposureTime", "AnalogueGain", "DigitalGain", "ColourGains",
             "ColourTemperature", "Lux", "FrameDuration", "SensorTimestamp")
PROCESOS_CAMARA = ("expression_faces_node", "romeo_bringup", "fer.py",
                   "servidor_deepface", "libcamera-", "rpicam-")


def temperatura():
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
    except Exception:  # noqa: BLE001
        return None


def procesos_con_camara():
    try:
        s = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
    except Exception:  # noqa: BLE001
        return []
    yo = str(os.getpid())
    return [l.strip() for l in s.splitlines()[1:]
            if any(p in l for p in PROCESOS_CAMARA) and not l.strip().startswith(yo)
            and "grabar_caras" not in l]


def a_json(v):
    if isinstance(v, (list, tuple)):
        return [a_json(x) for x in v]
    if isinstance(v, (int, float, str)) or v is None:
        return v
    return str(v)


class Grabadora:
    def __init__(self, a):
        import cv2
        from picamera2 import Picamera2
        self.cv2, self.a = cv2, a
        self.sensor = None
        if not a.sin_sensor:
            sys.path.insert(0, str(REPO / "ros2_ws" / "src" / "romeo_drivers"))
            from romeo_drivers._yunet_camara import YunetCamara
            s = YunetCamara(log=print)
            if s.cargar():
                self.sensor = s
            else:
                print(f"AVISO: YuNet en el sensor no disponible ({s.error}).")
        self.picam2 = (Picamera2(self.sensor.camera_num) if self.sensor is not None
                       else Picamera2())
        self.picam2.configure(self.picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "BGR888"},
            lores={"size": (320, 240), "format": "YUV420"}))
        self.picam2.start(show_preview=False)
        time.sleep(2)                                   # exposicion
        if self.sensor is not None:                     # el IMX500 tarda en dar salida
            con, limite = 0, time.time() + 20
            while con < 5 and time.time() < limite:
                con += 1 if self._pedido()[3] else 0
            self._info_sensor()
        print("CAMARA_LISTA", flush=True)

    def _info_sensor(self):
        """Pregunta (a): con que tamano entra la imagen a la red del IMX500, y
        si el firmware entrega su tensor de entrada (sirve para ver su orden
        de color con el objeto rojo)."""
        imx, info = self.sensor._imx, {}
        for nombre in ("get_input_size", "get_input_w_h"):
            try:
                info[nombre] = a_json(getattr(imx, nombre)())
            except Exception as e:  # noqa: BLE001
                info[nombre] = f"error: {e}"[:120]
        try:
            info["network_intrinsics"] = str(getattr(imx, "network_intrinsics", None))[:600]
        except Exception:  # noqa: BLE001
            pass
        p = self.picam2.capture_request()
        try:
            md = p.get_metadata()
        finally:
            p.release()
        info["claves_metadata"] = sorted(md.keys())
        info["ScalerCrop"] = a_json(md.get("ScalerCrop"))
        info["hay_CnnInputTensor"] = md.get("CnnInputTensor") is not None
        info["camera_controls_ScalerCrop"] = a_json(self.picam2.camera_controls.get("ScalerCrop"))
        d = Path(self.a.salida) / "meta"
        d.mkdir(parents=True, exist_ok=True)
        (d / "sensor_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
        print("SENSOR_INFO " + json.dumps({k: v for k, v in info.items() if k != "claves_metadata"},
                                          default=str)[:900], flush=True)

    def _tensor_entrada(self, md, destino):
        """Guarda la imagen que ve la red del IMX500, si el firmware la da.
        Devuelve la media de sus 3 canales en el centro, en el orden en que
        llegan, o None."""
        t = md.get("CnnInputTensor")
        if t is None:
            return None
        import numpy as np
        try:
            img = np.asarray(self.sensor._imx.input_tensor_image(t))
        except Exception as e:  # noqa: BLE001
            return f"error: {e}"[:120]
        np.save(destino, img)
        h, w = img.shape[:2]
        c = img[h // 3:2 * h // 3, w // 3:2 * w // 3].reshape(-1, img.shape[-1])
        return {"forma": list(img.shape), "media_canales_centro": [round(float(v), 1) for v in c.mean(axis=0)[:3]]}

    def _pedido(self):
        p = self.picam2.capture_request()
        try:
            main = p.make_array("main")
            lores = p.make_array("lores")
            md = p.get_metadata()
            salidas = None
            if self.sensor is not None:
                try:
                    salidas = self.sensor._imx.get_outputs(md, add_batch=True)
                except Exception:  # noqa: BLE001
                    salidas = None
        finally:
            p.release()
        return main, lores, md, salidas

    def grabar(self, esc, n):
        import numpy as np
        cv2, base = self.cv2, Path(self.a.salida)
        d640, d320 = base / "640x480" / esc, base / "320x240" / esc
        dsen, dmeta = base / "sensor" / esc, base / "meta"
        for d in (d640, d320, dsen, dmeta):
            d.mkdir(parents=True, exist_ok=True)
        periodo, con_sensor, crudo_centro, tensores_entrada = 1.0 / self.a.fps, 0, [], []
        al_primero = 0
        t_ini = temperatura()
        print(f"GRABANDO {esc} {n}", flush=True)
        with open(dmeta / f"{esc}.jsonl", "w", encoding="utf-8") as fm:
            for i in range(n):
                t = time.time()
                intentos = 0
                for _ in range(6 if self.sensor is not None else 1):
                    main, lores, md, salidas = self._pedido()
                    intentos += 1
                    if salidas:
                        break
                if self.sensor is not None and i in (0, n // 2):
                    te = self._tensor_entrada(md, dsen / f"entrada_{i:03d}.npy")
                    if te is not None:
                        tensores_entrada.append({"i": i, **(te if isinstance(te, dict) else {"error": te})})
                h, w = main.shape[:2]
                c = main[h // 3:2 * h // 3, w // 3:2 * w // 3].reshape(-1, main.shape[-1])
                crudo_centro.append(c.mean(axis=0)[:3].tolist())   # canales tal como llegan
                # "BGR888" llega en orden R,G,B -> BGR real.
                bgr = cv2.cvtColor(main, cv2.COLOR_RGBA2BGR if main.shape[-1] == 4
                                   else cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(d640 / f"{i:03d}.png"), bgr)
                cv2.imwrite(str(d320 / f"{i:03d}.png"), cv2.cvtColor(lores, cv2.COLOR_YUV2BGR_I420))
                if salidas:
                    np.savez_compressed(dsen / f"{i:03d}.npz", *[np.asarray(s) for s in salidas])
                    con_sensor += 1
                # intentos == 1: el sensor dio salida al primer pedido, que es
                # lo unico que ROMEO mira (un pedido por ciclo de 5 fps).
                fila = {"i": i, "t": round(t, 3), "sensor": bool(salidas), "intentos": intentos}
                fila.update({k: a_json(md.get(k)) for k in METADATOS})
                fm.write(json.dumps(fila) + "\n")
                al_primero += int(bool(salidas) and intentos == 1)
                if (i + 1) % 25 == 0:
                    print(f"  {i + 1}/{n}", flush=True)
                time.sleep(max(0.0, periodo - (time.time() - t)))
        canal = [round(sum(v[k] for v in crudo_centro) / len(crudo_centro), 1) for k in range(3)]
        info = {"escenario": esc, "descripcion": ESCENARIOS.get(esc, ""), "cuadros": n,
                "con_salida_sensor": con_sensor, "salida_al_primer_pedido": al_primero,
                "tensor_entrada_sensor": tensores_entrada, "fps": self.a.fps,
                "media_canales_crudos_centro": canal,
                "nota_canales": "orden tal como lo entrega make_array('main') con BGR888",
                "temp_ini_c": t_ini, "temp_fin_c": temperatura(),
                "fecha": time.strftime("%Y-%m-%d %H:%M:%S")}
        (dmeta / f"{esc}.info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        print(f"LISTO {esc}: {n} cuadros, sensor {con_sensor}/{n} (al primer pedido {al_primero}/{n}),"
              f" canales crudos centro {canal}, tensor entrada {tensores_entrada or 'no disponible'}",
              flush=True)

    def cerrar(self):
        try:
            self.picam2.stop()
        finally:
            self.picam2.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--escenarios", nargs="+", choices=list(ESCENARIOS))
    ap.add_argument("--servidor", action="store_true",
                    help="lee 'escenario [cuadros]' por stdin, uno por linea; q sale")
    ap.add_argument("--cuadros", type=int, default=100)
    ap.add_argument("--fps", type=float, default=5.0, help="como ROMEO (fer_fps: 5)")
    ap.add_argument("--salida", default=str(AQUI / "datos"))
    ap.add_argument("--sin-sensor", action="store_true")
    ap.add_argument("--forzar", action="store_true")
    a = ap.parse_args()
    if not a.escenarios and not a.servidor:
        ap.error("usa --escenarios ... o --servidor")
    otros = procesos_con_camara()
    if otros and not a.forzar:
        print("Hay procesos que pueden estar usando la camara:")
        for l in otros:
            print("   ", l)
        sys.exit("Detenlos primero o usa --forzar.")

    g = Grabadora(a)
    try:
        if a.servidor:
            for linea in sys.stdin:
                partes = linea.split()
                if not partes:
                    continue
                if partes[0] == "q":
                    break
                if partes[0] not in ESCENARIOS:
                    print(f"ERROR escenario desconocido: {partes[0]}", flush=True)
                    continue
                g.grabar(partes[0], int(partes[1]) if len(partes) > 1 else a.cuadros)
        else:
            for esc in a.escenarios:
                g.grabar(esc, a.cuadros)
    finally:
        g.cerrar()
        print("FIN", flush=True)


if __name__ == "__main__":
    main()
