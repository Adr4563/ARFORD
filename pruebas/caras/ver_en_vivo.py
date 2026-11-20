#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ver_en_vivo.py -- muestra EN VIVO, en el navegador del PC, como cada
detector de cara procesa los cuadros y como se arman las metricas.

Usa EXACTAMENTE los detectores de comparar_caras.py (se importa ese modulo),
asi que lo que se ve es lo mismo que produjo INFORME_CARAS.md.

MODOS
  grabado  Repite la prueba sobre los cuadros guardados (datos/). Por cada
           cuadro corren los 5 detectores UNO TRAS OTRO (como en
           comparar_caras.py, latencias comparables). En pantalla: la caja,
           el puntaje y la latencia de cada uno, y el panel con la tasa de
           deteccion, el IC de Wilson 95 % y los falsos positivos por
           escenario. Al terminar imprime las vistas 1 y 2.
  camara   En vivo con la AI Camera. Cada detector en su PROPIO hilo con el
           ultimo cuadro: los lentos se quedan atras a la vista. OJO: aqui
           compiten por la CPU, las latencias NO son las del informe.
           Enciende la camara: detener ROMEO antes.

VER: el script imprime una direccion http://<ip-de-la-pi>:8080/<clave>/ ;
abrirla en el navegador del PC (misma red). La clave es aleatoria en cada
ejecucion para que nadie mas en la red vea las caras. Nada se guarda.

USO (en la Pi, desde la raiz del repo)
    python3 pruebas/caras/ver_en_vivo.py grabado
    python3 pruebas/caras/ver_en_vivo.py grabado --res 640x480 --escenarios ang_60 dist_1.5m
    python3 pruebas/caras/ver_en_vivo.py grabado --cada 5          # 1 de cada 5 cuadros (rapido)
    python3 pruebas/caras/ver_en_vivo.py camara --detectores yunet_cpu yunet_sensor yolo11n_face
"""
import argparse
import http.server
import secrets
import socket
import socketserver
import statistics
import sys
import threading
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import comparar_caras as cc  # noqa: E402  (mismos detectores y metricas)

COLOR = {"yunet_cpu": (0, 220, 0), "yunet_sensor": (0, 200, 255), "scrfd_2g5": (255, 160, 0),
         "yolov8n_face": (255, 0, 200), "yolo11n_face": (60, 60, 255)}
TW, TH = 320, 240                                   # tamano de cada casilla
# Que compara cada umbral (MEDIDO, ver INFORME_CARAS.md 5.4)
ESCALA = {"yunet_cpu": "sqrt(cls*obj)", "yunet_sensor": "cls*obj", "scrfd_2g5": "score",
          "yolov8n_face": "score, NMS interno", "yolo11n_face": "score"}


def umbral_txt(nombre):
    u = cc.DETECTORES[nombre]["umbral"]
    extra = f" ~{u ** 0.5:.2f} en escala CPU" if nombre == "yunet_sensor" else ""
    return f"umbral {u:.2f} ({ESCALA.get(nombre, '')}){extra}"


# ------------------------------------------------------------ transmision
class Pantalla:
    """Ultimo JPEG + controles (pausa / saltar escenario)."""

    def __init__(self):
        self.jpeg, self.n = b"", 0
        self.cond = threading.Condition()
        self.pausa = False
        self.saltar = False
        self.comando = None                         # modo camara: (escenario, segundos)

    def publicar(self, img):
        import cv2
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self.cond:
                self.jpeg, self.n = buf.tobytes(), self.n + 1
                self.cond.notify_all()


HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Detectores en vivo</title>
<style>body{background:#111;color:#ddd;font-family:sans-serif;margin:12px}
img{max-width:100%;border:1px solid #444}button{font-size:16px;margin:6px 6px 6px 0;padding:6px 14px}</style>
</head><body><h3>Detectores de cara -- MODO</h3>
<button onclick="fetch('pausa')">Pausa / seguir</button>
<button onclick="fetch('saltar')">Siguiente escenario</button>
<span id="estado" style="margin-left:12px;color:#8c8"></span>
<div><img id="vista" alt="cargando la camara..."></div>
<script>
// Pide un cuadro nuevo ~5 veces por segundo (funciona aunque haya proxy;
// el video continuo multipart a veces queda retenido por el proxy).
const img = document.getElementById('vista'), est = document.getElementById('estado');
let n = 0, fallos = 0;
function siguiente() {
  const t = new Image();
  t.onload = () => { img.src = t.src; n++; fallos = 0; est.textContent = 'en vivo (' + n + ' cuadros)'; setTimeout(siguiente, 150); };
  t.onerror = () => { fallos++; est.textContent = 'sin conexion con la Pi, reintentando (' + fallos + ')'; setTimeout(siguiente, 1000); };
  t.src = 'cuadro.jpg?t=' + Date.now();
}
siguiente();
</script>
<p>Verde = yunet_cpu, naranja claro = yunet_sensor, azul = scrfd_2g5, rosa = yolov8n_face, rojo = yolo11n_face.</p>
</body></html>"""


def servir(pantalla, clave, modo, puerto):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            p = self.path
            if p == f"/{clave}":                    # sin la barra final: redirigir
                self.send_response(301)
                self.send_header("Location", f"/{clave}/")
                self.end_headers()
                return
            if not p.startswith(f"/{clave}/"):
                self.send_error(404)
                return
            p = p[len(clave) + 2:]
            if p.startswith("cuadro.jpg"):          # un cuadro suelto (la pagina lo pide en bucle)
                with pantalla.cond:
                    jpeg = pantalla.jpeg
                if not jpeg:
                    self.send_error(503)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.end_headers()
                self.wfile.write(jpeg)
                return
            if p.startswith("iniciar?"):            # modo camara: abre una ventana de medicion
                from urllib.parse import parse_qs
                q = parse_qs(p.split("?", 1)[1])
                pantalla.comando = {"esc": q.get("esc", ["escenario"])[0],
                                    "seg": float(q.get("seg", ["20"])[0]),
                                    "n": int(q.get("n", ["100"])[0])}
                self.send_response(204)
                self.end_headers()
                return
            if p == "":
                b = HTML.replace("MODO", modo).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(b)
            elif p in ("pausa", "saltar"):
                if p == "pausa":
                    pantalla.pausa = not pantalla.pausa
                else:
                    pantalla.saltar = True
                self.send_response(204)
                self.end_headers()
            elif p == "stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
                self.end_headers()
                visto = -1
                try:
                    while True:
                        with pantalla.cond:
                            pantalla.cond.wait_for(lambda: pantalla.n != visto, timeout=5)
                            jpeg, visto = pantalla.jpeg, pantalla.n
                        if jpeg:
                            self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

    class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    srv = S(("0.0.0.0", puerto), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    except OSError:
        ip = "<ip-de-la-pi>"
    print(f"\n>>> Abre en el navegador del PC:  http://{ip}:{puerto}/{clave}/\n", flush=True)


# ------------------------------------------------------------- dibujo
def casilla(img, nombre, cajas, extra, linea2=""):
    import cv2
    h, w = img.shape[:2]
    c = cv2.resize(img, (TW, TH))
    ex, ey = TW / w, TH / h
    for b in cajas or []:
        x, y, bw, bh = b[:4]
        cv2.rectangle(c, (int(x * ex), int(y * ey)), (int((x + bw) * ex), int((y + bh) * ey)),
                      COLOR.get(nombre, (0, 255, 0)), 2)
    cv2.rectangle(c, (0, 0), (TW, 48), (0, 0, 0), -1)
    cv2.putText(c, f"{nombre}  {extra}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                COLOR.get(nombre, (255, 255, 255)), 1, cv2.LINE_AA)
    if nombre in cc.DETECTORES:
        cv2.putText(c, umbral_txt(nombre), (4, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.37,
                    (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(c, linea2, (4, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (230, 230, 230), 1, cv2.LINE_AA)
    return c


def panel(lineas, ancho=TW, alto=TH):
    import cv2
    import numpy as np
    p = np.zeros((alto, ancho, 3), np.uint8)
    for i, (txt, col) in enumerate(lineas[: alto // 15]):
        cv2.putText(p, txt, (4, 14 + 15 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.37, col, 1, cv2.LINE_AA)
    return p


def mosaico(casillas):
    import numpy as np
    while len(casillas) % 3:
        casillas.append(np.zeros((TH, TW, 3), np.uint8))
    return np.vstack([np.hstack(casillas[i:i + 3]) for i in range(0, len(casillas), 3)])


def pct(k, n):
    return "-" if not n else f"{100 * k / n:.0f}%"


def ic(k, n):
    lo, hi = cc.wilson(k, n)
    return "-" if lo is None else f"[{100 * lo:.0f}-{100 * hi:.0f}]"


# ------------------------------------------------------------ modo grabado
def modo_grabado(a, pantalla):
    import cv2
    datos = Path(a.datos)
    cuadros = [c for c in cc.listar_cuadros(datos) if c[0] == a.res
               and (not a.escenarios or c[1] in a.escenarios)][:: a.cada]
    if not cuadros:
        sys.exit(f"No hay cuadros en {datos}/{a.res}")
    print(f"Cargando detectores: {a.detectores}", flush=True)
    dets = {}
    for d in a.detectores:
        try:
            dets[d] = cc.crear(d, a.hilos)
        except Exception as e:  # noqa: BLE001
            print(f"  {d}: no carga ({e})")
    print(f"{len(cuadros)} cuadros de {a.res}; corren uno tras otro, como comparar_caras.py", flush=True)

    filas = {d: [] for d in dets}
    est = {}                                        # (esc, det) -> [k, n, sin_dato, lats]
    hechos = []                                     # resumen de escenarios terminados
    esc_actual, i_esc = None, 0
    n_esc = {}
    for _, e, _ in cuadros:
        n_esc[e] = n_esc.get(e, 0) + 1
    for k, (res, esc, ruta) in enumerate(cuadros):
        if esc != esc_actual:
            if esc_actual is not None:
                hechos.append(resumen_esc(esc_actual, dets, est))
            esc_actual, i_esc, pantalla.saltar = esc, 0, False
        if pantalla.saltar:
            continue
        while pantalla.pausa:
            time.sleep(0.1)
        i_esc += 1
        img = cv2.imread(str(ruta))
        tiles = []
        for d, det in dets.items():
            t = time.perf_counter()
            cajas = det(img, ruta)
            lat = (time.perf_counter() - t) * 1000
            smax = det.max_crudo(img, ruta)
            s = est.setdefault((esc, d), [0, 0, 0, []])
            if cajas is None:
                s[2] += 1
                txt, l2 = "SIN DATO del sensor", f"{lat:.0f} ms"
            else:
                s[1] += 1
                s[0] += int(len(cajas) > 0)
                s[3].append(lat)
                mejor = max(cajas, key=lambda c: c[4], default=None)
                txt = (f"{len(cajas)} caja(s)  p={mejor[4]:.2f}" if mejor else "sin caja") + f"  {lat:.0f} ms"
                etiqueta = "FP" if esc == cc.SIN_CARA else "det"
                l2 = f"{etiqueta} {s[0]}/{s[1]} {pct(s[0], s[1])} {ic(s[0], s[1])}  crudo {smax if smax is None else round(smax, 3)}"
            filas[d].append({"resolucion": res, "escenario": esc, "archivo": ruta.relative_to(datos).as_posix(),
                             "lat_ms": round(lat, 2), "max_crudo": None if smax is None else float(smax),
                             "n_cajas": None if cajas is None else len(cajas),
                             "ancho": None if not cajas else round(max(cajas, key=lambda c: c[4])[2], 1),
                             "score": None, "cajas": None})
            tiles.append(casilla(img, d, cajas, txt, l2))
        lineas = [(f"MODO GRABADO  {res}  cuadro {k + 1}/{len(cuadros)}", (255, 255, 255)),
                  (f"escenario: {esc}  ({i_esc}/{n_esc[esc]})", (0, 255, 255)),
                  ("det = cuadros con >=1 caja; IC Wilson 95%", (160, 160, 160))]
        for d in dets:
            s = est.get((esc, d), [0, 0, 0, []])
            med = f"{statistics.median(s[3]):.0f}ms" if s[3] else "-"
            lineas.append((f"{d[:12]:12} {pct(s[0], s[1]):>4} {ic(s[0], s[1]):>9} {med:>6}"
                           + (f" sd{s[2]}" if s[2] else ""), COLOR.get(d, (255, 255, 255))))
        lineas.append(("-- escenarios terminados --", (160, 160, 160)))
        lineas += [(h, (200, 200, 200)) for h in hechos[-6:]]
        pantalla.publicar(mosaico(tiles + [panel(lineas)]))
        if a.pausa_ms:
            time.sleep(a.pausa_ms / 1000)
    if esc_actual is not None:
        hechos.append(resumen_esc(esc_actual, dets, est))

    print("\n================ RESULTADO (mismas funciones que comparar_caras.py) ================")
    for h in hechos:
        print("  " + h)
    print("\nVista 1 (umbral de operacion): tasa % [IC Wilson n | n_ef]")
    for d, fs in filas.items():
        v1 = cc.vista1(fs)
        print(f"  {d}: " + "  ".join(f"{k[1]} {cc.pct(m['tasa'])}%[{cc.ic(m['ic'])}|{cc.ic(m['ic_nef'])}]"
                                    for k, m in v1.items()))
    print("\nVista 2 (umbral con FP <= 1 % en sin_cara; requiere sin_cara entre los escenarios):")
    for d, fs in filas.items():
        v2 = cc.vista2(fs).get(a.res, {}).get(0.01)
        if v2:
            u = v2["uso"]
            print(f"  {d}: umbral {v2['umbral']}  FP {cc.pct(v2['fp'])}%  uso real {cc.pct(u['tasa'])}%"
                  f" [{cc.ic(u['ic'])} | {cc.ic(u['ic_nef'])}]" + ("  (criterio degenerado)" if v2["degenerado"] else ""))
    print("\nListo. La pagina sigue mostrando el ultimo cuadro; Ctrl+C para salir.")
    while True:
        time.sleep(1)


def resumen_esc(esc, dets, est):
    partes = []
    for d in dets:
        s = est.get((esc, d), [0, 0, 0, []])
        partes.append(f"{d.split('_')[0][:5]}{'S' if d == 'yunet_sensor' else ''} {pct(s[0], s[1])}")
    return f"{esc[:10]:10} " + " ".join(partes)


# ------------------------------------------------------------- modo camara
class SensorEnVivo:
    """YuNet del IMX500 sobre la salida en vivo (misma decodificacion que el robot)."""

    def __init__(self, yc, umbral):
        self.yc = yc
        yc.UMBRAL = umbral

    def evaluar(self, salidas, ancho, alto):
        import numpy as np
        if not salidas:
            return None, None
        smax = float(max((np.asarray(salidas[i]).reshape(-1) * np.asarray(salidas[3 + i]).reshape(-1)).max()
                         for i in range(3)))
        c = self.yc.decodificar(salidas, ancho, alto)
        return ([] if c is None else [(c[0], c[1], c[2], c[3], smax)]), smax


def modo_camara(a, pantalla):
    import cv2
    from picamera2 import Picamera2
    otros = cc_procesos_camara()
    if otros:
        sys.exit("Hay procesos usando la camara (detener ROMEO):\n  " + "\n  ".join(otros))
    ancho, alto = (int(v) for v in a.res.split("x"))
    sensor = imx = None
    if "yunet_sensor" in a.detectores:
        sys.path.insert(0, str(cc.REPO / "ros2_ws" / "src" / "romeo_drivers"))
        from romeo_drivers import _yunet_camara as yc
        cam = yc.YunetCamara(log=print)
        if cam.cargar():
            imx, sensor = cam, SensorEnVivo(yc, cc.DETECTORES["yunet_sensor"]["umbral"])
        else:
            print(f"yunet_sensor no disponible: {cam.error}")
    picam2 = Picamera2(imx.camera_num) if imx else Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (ancho, alto), "format": "BGR888"}))
    picam2.start(show_preview=False)
    time.sleep(2)

    ultimo = {"id": 0, "img": None}
    res = {}                                        # det -> dict(img, cajas, lat, n, k, t0)
    cand = threading.Lock()
    vivo = True
    # Ventana de medicion por escenario: solo cuenta lo procesado entre
    # inicio y fin. wn/wk = cuadros procesados / con cara en la ventana.
    ventana = {"esc": None, "ini": 0.0, "fin": 0.0}
    hechos = []

    def contar(r, con_cara, lat=None):
        if ventana["ini"] <= time.time() < ventana["fin"]:
            r["wn"] = r.get("wn", 0) + 1
            r["wk"] = r.get("wk", 0) + int(con_cara)
            if lat is not None:
                r.setdefault("wlats", []).append(lat)

    def capturar():
        while vivo:
            p = picam2.capture_request()
            try:
                f = p.make_array("main")
                sal = imx._imx.get_outputs(p.get_metadata(), add_batch=True) if imx else None
            except Exception:  # noqa: BLE001
                sal = None
            finally:
                p.release()
            # "BGR888" de picamera2 llega en orden R,G,B (MEDIDO): se pasa a BGR real
            img = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            with cand:
                ultimo["id"] += 1
                ultimo["img"] = img
            if sensor is not None:
                t = time.perf_counter()
                cajas, smax = sensor.evaluar(sal, ancho, alto)
                r = res.setdefault("yunet_sensor", {"n": 0, "k": 0, "sd": 0, "t0": time.time(), "lats": []})
                if cajas is None:
                    r["sd"] += 1
                    if ventana["ini"] <= time.time() < ventana["fin"]:
                        r["wsd"] = r.get("wsd", 0) + 1
                else:
                    r.update(img=img, cajas=cajas, lat=(time.perf_counter() - t) * 1000, smax=smax)
                    r["n"] += 1
                    r["k"] += int(len(cajas) > 0)
                    contar(r, len(cajas) > 0)

    def trabajar(nombre):
        det = cc.crear(nombre, a.hilos)
        r = res.setdefault(nombre, {"n": 0, "k": 0, "sd": 0, "t0": time.time(), "lats": []})
        visto = 0
        while vivo:
            with cand:
                i, img = ultimo["id"], ultimo["img"]
            if img is None or i == visto:
                time.sleep(0.005)
                continue
            visto = i
            t = time.perf_counter()
            cajas = det(img, Path("vivo.png"))
            lat = (time.perf_counter() - t) * 1000
            r.update(img=img, cajas=cajas, lat=lat, smax=det.max_crudo(img, None))
            r["n"] += 1
            r["k"] += int(len(cajas) > 0)
            r["lats"] = (r["lats"] + [lat])[-50:]
            contar(r, len(cajas) > 0, lat)

    threading.Thread(target=capturar, daemon=True).start()
    for d in a.detectores:
        if d != "yunet_sensor":
            threading.Thread(target=trabajar, args=(d,), daemon=True).start()
    print("Camara encendida. Ctrl+C para apagar.", flush=True)
    try:
        while True:
            with cand:
                base = ultimo["img"]
            if base is None:
                time.sleep(0.05)
                continue
            ahora = time.time()
            if pantalla.comando:                    # nuevo escenario: reinicia contadores de ventana
                esc, seg = pantalla.comando["esc"], pantalla.comando["seg"]
                pantalla.comando = None
                for r in res.values():
                    for k in ("wn", "wk", "wsd", "wlats"):
                        r.pop(k, None)
                ventana.update(esc=esc, ini=ahora, fin=ahora + seg)
                print(f"VENTANA {esc}: {seg:.0f} s", flush=True)
            midiendo = ventana["ini"] <= ahora < ventana["fin"]
            if ventana["esc"] and not midiendo and ventana.get("cerrada") != ventana["ini"]:
                ventana["cerrada"] = ventana["ini"]  # cierra la ventana: resumen fijo
                partes = []
                for d in a.detectores:
                    r = res.get(d, {})
                    wn, wk = r.get("wn", 0), r.get("wk", 0)
                    partes.append(f"{d}={wk}/{wn}" + (f"(sd{r['wsd']})" if r.get("wsd") else ""))
                    hechos.append((f"{ventana['esc'][:9]:9} {d.split('_')[0][:5]}"
                                   f"{'S' if d == 'yunet_sensor' else ''} {pct(wk, wn):>4} {ic(wk, wn)} n={wn}",
                                   COLOR.get(d, (255, 255, 255))))
                print(f"RESUMEN {ventana['esc']}: " + "  ".join(partes), flush=True)
            if midiendo:
                cab = (f"MIDIENDO '{ventana['esc']}'  quedan {ventana['fin'] - ahora:4.1f} s", (0, 0, 255))
            else:
                cab = ("EN ESPERA (sin medir)", (0, 255, 255))
            tiles, lineas = [], [(f"MODO CAMARA EN VIVO  {a.res}", (255, 255, 255)), cab,
                                 ("fps reales con CPU compartida; det = ventana", (160, 160, 160))]
            for d in a.detectores:
                r = res.get(d)
                if not r or "img" not in r:
                    tiles.append(casilla(base, d, [], "cargando..."))
                    continue
                fps = r["n"] / max(ahora - r["t0"], 1e-6)
                wn, wk = r.get("wn", 0), r.get("wk", 0)
                sm = r.get("smax")
                txt = (f"{len(r['cajas'])} caja  crudo {'-' if sm is None else f'{sm:.2f}'}"
                       f"  {r['lat']:.0f}ms {fps:.1f}fps")
                l2 = (f"ventana: cara {wk}/{wn} {pct(wk, wn)} {ic(wk, wn)}" if (midiendo or wn)
                      else "esperando 'listo'") + (f" sd{r['wsd']}" if r.get("wsd") else "")
                tiles.append(casilla(r["img"], d, r["cajas"], txt, l2))
                lineas.append((f"{d[:12]:12} {fps:4.1f}fps {r['lat']:5.0f}ms {pct(wk, wn):>4} n={wn}",
                               COLOR.get(d, (255, 255, 255))))
            lineas.append(("-- escenarios medidos --", (160, 160, 160)))
            if len(tiles) == 1:                     # un solo modelo: vista grande para la demostracion
                import numpy as np
                lineas += hechos[-(480 // 15 - len(lineas) - 1):]
                pantalla.publicar(np.hstack([cv2.resize(tiles[0], (640, 480)), panel(lineas, TW, 480)]))
            else:
                lineas += hechos[-(TH // 15 - len(lineas) - 1):]
                pantalla.publicar(mosaico(tiles + [panel(lineas)]))
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        vivo = False
        time.sleep(0.3)
        picam2.stop()
        picam2.close()
        print("Camara apagada.")


# --------------------------------------------------------- modo prueba
def modo_prueba(a, pantalla):
    """Prueba NUEVA a la vista: graba cada escenario (como grabar_caras.py) y
    enseguida corre los detectores UNO TRAS OTRO sobre esos mismos cuadros
    (como comparar_caras.py), mostrando todo en el dashboard."""
    import json
    import cv2
    import numpy as np
    from picamera2 import Picamera2
    otros = cc_procesos_camara()
    if otros:
        sys.exit("Hay procesos usando la camara (detener ROMEO):\n  " + "\n  ".join(otros))
    base = Path(a.salida)
    sys.path.insert(0, str(cc.REPO / "ros2_ws" / "src" / "romeo_drivers"))
    from romeo_drivers import _yunet_camara as yc
    cam = yc.YunetCamara(log=print)
    imx = cam if cam.cargar() else None
    if imx is None:
        print(f"AVISO: yunet_sensor no disponible ({cam.error})")
    picam2 = Picamera2(imx.camera_num) if imx else Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "BGR888"}, lores={"size": (320, 240), "format": "YUV420"}))
    picam2.start(show_preview=False)
    time.sleep(2)
    print(f"Cargando detectores: {a.detectores}", flush=True)
    dets = {}
    for d in a.detectores:
        if d == "yunet_sensor" and imx is None:
            continue
        try:
            dets[d] = cc.crear(d, a.hilos)
        except Exception as e:  # noqa: BLE001
            print(f"  {d}: no carga ({e})")
    vivo_sensor = SensorEnVivo(yc, cc.DETECTORES["yunet_sensor"]["umbral"]) if imx else None
    est, filas, hechos = {}, {d: [] for d in dets}, []
    print(f"Listo. Datos nuevos en {base}. Procesa a {a.res}.", flush=True)

    def pedido():
        p = picam2.capture_request()
        try:
            main, lores, md = p.make_array("main"), p.make_array("lores"), p.get_metadata()
            try:
                sal = imx._imx.get_outputs(md, add_batch=True) if imx else None
            except Exception:  # noqa: BLE001
                sal = None
        finally:
            p.release()
        return main, lores, md, sal

    def grande(img, cajas, lineas, titulo, color=(0, 255, 255)):
        g = cv2.resize(img, (640, 480))
        ex, ey = 640 / img.shape[1], 480 / img.shape[0]
        for b in cajas or []:
            x, y, w, h = b[:4]
            cv2.rectangle(g, (int(x * ex), int(y * ey)), (int((x + w) * ex), int((y + h) * ey)),
                          COLOR["yunet_sensor"], 2)
        cv2.rectangle(g, (0, 0), (640, 26), (0, 0, 0), -1)
        cv2.putText(g, titulo, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        return np.hstack([g, panel(lineas, TW, 480)])

    def lineas_estado(esc=None):
        ls = [("MODO PRUEBA (datos nuevos)", (255, 255, 255)),
              (f"guarda en {base.name}/  procesa {a.res}", (160, 160, 160)),
              ("-- escenarios medidos --", (160, 160, 160))]
        return ls + hechos[-26:]

    try:
        while True:
            cmd, pantalla.comando = pantalla.comando, None
            if cmd is None:                                   # EN ESPERA: vista en vivo
                main, _, _, sal = pedido()
                img = cv2.cvtColor(main, cv2.COLOR_RGB2BGR)   # "BGR888" llega R,G,B (MEDIDO)
                cajas = vivo_sensor.evaluar(sal, 640, 480)[0] if vivo_sensor else None
                pantalla.publicar(grande(img, cajas, lineas_estado(),
                                         "EN VIVO - EN ESPERA (escribe 'listo' para medir)"))
                time.sleep(0.05)
                continue
            esc, n = cmd["esc"], cmd["n"]
            if esc == "FIN":
                break
            # ---------------- 1) grabar (igual que grabar_caras.py)
            d640, d320 = base / "640x480" / esc, base / "320x240" / esc
            dsen, dmeta = base / "sensor" / esc, base / "meta"
            for d in (d640, d320, dsen, dmeta):
                d.mkdir(parents=True, exist_ok=True)
            print(f"GRABANDO {esc} {n}", flush=True)
            with open(dmeta / f"{esc}.jsonl", "w", encoding="utf-8") as fm:
                for i in range(n):
                    t = time.time()
                    intentos = 0
                    for _ in range(6 if imx else 1):          # el sensor da salida en ~1 de 2
                        main, lores, md, sal = pedido()
                        intentos += 1
                        if sal:
                            break
                    bgr = cv2.cvtColor(main, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(d640 / f"{i:03d}.png"), bgr)
                    cv2.imwrite(str(d320 / f"{i:03d}.png"), cv2.cvtColor(lores, cv2.COLOR_YUV2BGR_I420))
                    if sal:
                        np.savez_compressed(dsen / f"{i:03d}.npz", *[np.asarray(s) for s in sal])
                    fm.write(json.dumps({"i": i, "sensor": bool(sal), "intentos": intentos,
                                         **{k: str(md.get(k)) for k in ("ExposureTime", "AnalogueGain",
                                                                         "ColourGains", "Lux")}}) + "\n")
                    cajas = vivo_sensor.evaluar(sal, 640, 480)[0] if vivo_sensor else None
                    pantalla.publicar(grande(bgr, cajas, lineas_estado(),
                                             f"GRABANDO '{esc}'  {i + 1}/{n}  (muevete apenas)", (0, 0, 255)))
                    time.sleep(max(0.0, 1.0 / a.fps - (time.time() - t)))
            # Forzar la escritura a la tarjeta: si se corta la corriente, lo
            # grabado no queda en 0 bytes (paso el 19-11 con dist_0.5m).
            import os
            os.sync()
            # ---------------- 2) procesar (igual que comparar_caras.py)
            print(f"PROCESANDO {esc}", flush=True)
            rutas = sorted((base / a.res / esc).glob("*.png"))
            for k, ruta in enumerate(rutas):
                img = cv2.imread(str(ruta))
                tiles = []
                for d, det in dets.items():
                    t = time.perf_counter()
                    cajas = det(img, ruta)
                    lat = (time.perf_counter() - t) * 1000
                    smax = det.max_crudo(img, ruta)
                    s = est.setdefault((esc, d), [0, 0, 0, []])
                    if cajas is None:
                        s[2] += 1
                        txt, l2 = f"SIN DATO del sensor  {lat:.0f}ms", ""
                    else:
                        s[1] += 1
                        s[0] += int(len(cajas) > 0)
                        s[3].append(lat)
                        txt = f"{len(cajas)} caja  crudo {'-' if smax is None else f'{smax:.2f}'}  {lat:.0f}ms"
                        l2 = f"{'FP' if esc == cc.SIN_CARA else 'det'} {s[0]}/{s[1]} {pct(s[0], s[1])} {ic(s[0], s[1])}"
                    filas[d].append({"resolucion": a.res, "escenario": esc,
                                     "archivo": ruta.relative_to(base).as_posix(), "lat_ms": round(lat, 2),
                                     "max_crudo": None if smax is None else float(smax),
                                     "n_cajas": None if cajas is None else len(cajas),
                                     "ancho": None if not cajas else round(max(cajas, key=lambda c: c[4])[2], 1),
                                     "score": None, "cajas": None})
                    tiles.append(casilla(img, d, cajas, txt, l2))
                lin = [(f"PROCESANDO '{esc}'  cuadro {k + 1}/{len(rutas)}", (0, 255, 255)),
                       ("mismos cuadros, detectores uno tras otro", (160, 160, 160))]
                for d in dets:
                    s = est.get((esc, d), [0, 0, 0, []])
                    med = f"{statistics.median(s[3]):.0f}ms" if s[3] else "-"
                    lin.append((f"{d[:12]:12} {pct(s[0], s[1]):>4} {ic(s[0], s[1]):>9} {med:>6}"
                                + (f" sd{s[2]}" if s[2] else ""), COLOR.get(d, (255, 255, 255))))
                lin.append(("-- escenarios medidos --", (160, 160, 160)))
                lin += hechos[-(TH // 15 - len(lin) - 1):]
                pantalla.publicar(mosaico(tiles + [panel(lin)]))
            for d in dets:
                s = est.get((esc, d), [0, 0, 0, []])
                hechos.append((f"{esc[:9]:9} {d.split('_')[0][:5]}{'S' if d == 'yunet_sensor' else ''}"
                               f" {pct(s[0], s[1]):>4} {ic(s[0], s[1])}" + (f" sd{s[2]}" if s[2] else ""),
                               COLOR.get(d, (255, 255, 255))))
            print(f"RESUMEN {esc}: " + "  ".join(
                f"{d}={est.get((esc, d), [0, 0])[0]}/{est.get((esc, d), [0, 0])[1]}" for d in dets), flush=True)
    finally:
        picam2.stop()
        picam2.close()
        print("Camara apagada.", flush=True)
    print("\n======== RESULTADO DE LA PRUEBA NUEVA (funciones de comparar_caras.py) ========")
    for d, fs in filas.items():
        v1 = cc.vista1(fs)
        print(f"  {d}: " + "  ".join(f"{k[1]} {cc.pct(m['tasa'])}%[{cc.ic(m['ic'])}|{cc.ic(m['ic_nef'])}]"
                                    for k, m in v1.items()))
        v2 = cc.vista2(fs).get(a.res, {}).get(0.01)
        if v2:
            u = v2["uso"]
            print(f"     vista 2 (FP<=1%): umbral {v2['umbral']}  uso real {cc.pct(u['tasa'])}%"
                  f" [{cc.ic(u['ic_nef'])}]" + ("  (degenerado)" if v2["degenerado"] else ""))
    print(f"\nDatos nuevos en {base}. Para el informe oficial: "
          f"python3 pruebas/caras/comparar_caras.py --datos {base}", flush=True)
    while True:
        time.sleep(1)


def cc_procesos_camara():
    import os
    import subprocess
    s = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
    claves = ("expression_faces_node", "romeo_bringup", "fer.py", "grabar_caras", "rpicam-", "libcamera-")
    return [l.strip() for l in s.splitlines()[1:] if any(c in l for c in claves)
            and not l.strip().startswith(str(os.getpid()))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("modo", choices=["grabado", "camara", "prueba"])
    ap.add_argument("--salida", default=str(AQUI / "datos_vivo"), help="modo prueba: donde guarda los cuadros nuevos")
    ap.add_argument("--fps", type=float, default=5.0, help="modo prueba: como ROMEO (fer_fps 5)")
    ap.add_argument("--detectores", nargs="+", default=list(cc.DETECTORES), choices=list(cc.DETECTORES))
    ap.add_argument("--res", default="320x240", help="320x240 (la de ROMEO) o 640x480")
    ap.add_argument("--escenarios", nargs="+", help="solo estos (modo grabado)")
    ap.add_argument("--cada", type=int, default=1, help="usar 1 de cada N cuadros (modo grabado)")
    ap.add_argument("--pausa-ms", type=int, default=0, help="espera extra entre cuadros (modo grabado)")
    ap.add_argument("--hilos", type=int, default=2, help="igual que ROMEO")
    ap.add_argument("--datos", default=str(AQUI / "datos"))
    ap.add_argument("--puerto", type=int, default=8080)
    ap.add_argument("--clave", help="clave fija para la direccion (por defecto, una aleatoria en cada ejecucion)")
    ap.add_argument("--umbral-sensor", type=float, default=cc.DETECTORES["yunet_sensor"]["umbral"],
                    help="umbral de yunet_sensor sobre cls*obj (0.7 = ROMEO; 0.49 = equivalente a la CPU)")
    ap.add_argument("--umbral-todos", type=float,
                    help="mismo umbral para los 5 detectores (cada uno sobre SU puntaje)")
    a = ap.parse_args()
    cc.DETECTORES["yunet_sensor"]["umbral"] = a.umbral_sensor
    if a.umbral_todos is not None:
        for d in cc.DETECTORES.values():
            d["umbral"] = a.umbral_todos
    print("Umbrales: " + ", ".join(f"{k}={v['umbral']}" for k, v in cc.DETECTORES.items()), flush=True)
    import cv2
    cv2.setNumThreads(a.hilos)
    pantalla = Pantalla()
    servir(pantalla, a.clave or secrets.token_urlsafe(8), a.modo, a.puerto)
    try:
        {"grabado": modo_grabado, "camara": modo_camara, "prueba": modo_prueba}[a.modo](a, pantalla)
    except KeyboardInterrupt:
        print("\nSalida.")


if __name__ == "__main__":
    main()
