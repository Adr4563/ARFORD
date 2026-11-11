# -*- coding: utf-8 -*-
"""Prueba en vivo del modelo de emocion de DeepFace (el de Rocha/FRED) en la Pi.

Abre la camara con picamera2, encuentra la cara con YuNet y clasifica la
emocion con el modelo ONNX de DeepFace (CNN FER-2013, las 7 clases de la
tesis de Rocha p.126). Sirve una pagina con el video a la izquierda y la
emocion capturada en tiempo real a la derecha.

    python3 servidor_deepface.py              # http://<IP-de-la-Pi>:8070
    ROMEO_FER_PUERTO=9000 python3 servidor_deepface.py

IMPORTANTE: solo un proceso puede abrir la camara. Apaga Romeo antes
(no debe correr expression_faces_node).

Modelo DeepFace: entrada 48x48 gris /255 shape (1,48,48,1); salida 7 clases
en este orden fijo: angry, disgust, fear, happy, sad, surprise, neutral.
"""
import io
import os
import threading
import time

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, Response, jsonify, render_template_string

# --- rutas de los modelos ---
AQUI = os.path.dirname(os.path.abspath(__file__))
MODELO_EMOCION = os.path.join(AQUI, "modelos", "deepface_emotion.onnx")
# YuNet ya vive en el paquete de Romeo; lo reusamos.
_YUNET_CANDIDATOS = [
    os.path.expanduser(
        "~/ARFORD/ros2_ws/install/romeo_drivers/share/romeo_drivers/"
        "modelos/face_detection_yunet_2023mar.onnx"),
    os.path.expanduser(
        "~/ARFORD/ros2_ws/src/romeo_drivers/romeo_drivers/data/"
        "modelos/face_detection_yunet_2023mar.onnx"),
]
MODELO_CARA = next((p for p in _YUNET_CANDIDATOS if os.path.exists(p)),
                   _YUNET_CANDIDATOS[0])

PUERTO = int(os.environ.get("ROMEO_FER_PUERTO", "8070"))
RESOLUCION = (640, 480)
HILOS = int(os.environ.get("ROMEO_FER_HILOS", "2"))

# Las 7 clases de DeepFace (FER-2013), en el orden exacto de la salida.
ETIQUETAS = ["enojo", "asco", "miedo", "felicidad", "tristeza", "sorpresa", "neutral"]
# Color por emocion para el recuadro (BGR).
COLOR = {
    "enojo": (0, 0, 220), "asco": (0, 140, 0), "miedo": (180, 0, 180),
    "felicidad": (0, 200, 0), "tristeza": (200, 120, 0),
    "sorpresa": (0, 200, 220), "neutral": (160, 160, 160),
}

# --- carga de modelos ---
_op = ort.SessionOptions()
_op.intra_op_num_threads = HILOS
_op.inter_op_num_threads = 1
sess_emocion = ort.InferenceSession(MODELO_EMOCION, _op,
                                    providers=["CPUExecutionProvider"])
_in_emocion = sess_emocion.get_inputs()[0].name
detector_cara = cv2.FaceDetectorYN_create(MODELO_CARA, "", (320, 320),
                                           score_threshold=0.7)
cv2.setNumThreads(HILOS)

# --- estado compartido con la pagina ---
_estado = {"emocion": "--", "confianza": 0.0, "probs": {}, "cara": False,
           "fps": 0.0, "t": 0.0}
_lock = threading.Lock()
_ultimo_jpeg = {"data": None}


def _softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def _clasificar_cara(gris48):
    """gris48: (48,48) float32 /255 -> (etiqueta, confianza, dict probs)."""
    ent = gris48.reshape(1, 48, 48, 1).astype("float32")
    salida = sess_emocion.run(None, {_in_emocion: ent})[0][0]
    probs = _softmax(salida)
    idx = int(np.argmax(probs))
    d = {ETIQUETAS[i]: float(probs[i]) for i in range(len(ETIQUETAS))}
    return ETIQUETAS[idx], float(probs[idx]), d


def _procesar(bgr):
    """Dibuja sobre el frame la cara + emocion y actualiza el estado."""
    h, w = bgr.shape[:2]
    detector_cara.setInputSize((w, h))
    _, caras = detector_cara.detect(bgr)
    emocion, conf, probs, hay = "--", 0.0, {}, False
    if caras is not None and len(caras):
        # la cara mas grande
        cara = max(caras, key=lambda c: c[2] * c[3])
        x, y, cw, ch = [int(v) for v in cara[:4]]
        x, y = max(0, x), max(0, y)
        recorte = bgr[y:y + ch, x:x + cw]
        if recorte.size:
            gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
            gris = cv2.resize(gris, (48, 48)).astype("float32") / 255.0
            emocion, conf, probs = _clasificar_cara(gris)
            hay = True
            col = COLOR.get(emocion, (255, 255, 255))
            cv2.rectangle(bgr, (x, y), (x + cw, y + ch), col, 2)
            cv2.putText(bgr, f"{emocion} {conf*100:.0f}%", (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    return bgr, emocion, conf, probs, hay


def _bucle_camara():
    from picamera2 import Picamera2
    picam2 = Picamera2()
    picam2.start(picam2.create_preview_configuration(
        main={"size": RESOLUCION, "format": "BGR888"}), show_preview=False)
    time.sleep(2)  # AE/AWB
    t_prev = time.time()
    try:
        while True:
            frame = picam2.capture_array("main")
            if frame.shape[2] == 4:                       # XBGR -> BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            frame, emocion, conf, probs, hay = _procesar(frame)
            ahora = time.time()
            fps = 1.0 / max(1e-3, ahora - t_prev)
            t_prev = ahora
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                _ultimo_jpeg["data"] = buf.tobytes()
            with _lock:
                _estado.update(emocion=emocion, confianza=conf, probs=probs,
                               cara=hay, fps=fps, t=ahora)
    finally:
        picam2.stop()
        picam2.close()


app = Flask(__name__)

PAGINA = """<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DeepFace en vivo - Romeo</title>
<style>
 :root{--bg:#0f1115;--card:#1a1f2b;--fg:#e8ecf3;--sub:#8a94a6;--acc:#f06a3f}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);
   font-family:system-ui,Segoe UI,Roboto,sans-serif}
 header{padding:14px 20px;border-bottom:1px solid #222838}
 header h1{margin:0;font-size:18px} header small{color:var(--sub)}
 .wrap{display:flex;gap:16px;padding:16px;flex-wrap:wrap}
 .video{flex:1 1 520px;min-width:320px;background:#000;border-radius:12px;
   overflow:hidden;border:1px solid #222838}
 .video img{width:100%;display:block}
 .panel{flex:1 1 300px;min-width:280px}
 .card{background:var(--card);border:1px solid #222838;border-radius:12px;
   padding:16px;margin-bottom:14px}
 .big{font-size:40px;font-weight:700;letter-spacing:.5px}
 .conf{color:var(--sub);font-size:14px;margin-top:4px}
 .row{display:flex;align-items:center;gap:8px;margin:7px 0}
 .row .lab{width:86px;font-size:13px;color:var(--sub);text-transform:capitalize}
 .bar{flex:1;height:12px;background:#121722;border-radius:6px;overflow:hidden}
 .bar>span{display:block;height:100%;background:var(--acc);width:0%}
 .pct{width:44px;text-align:right;font-size:12px;color:var(--sub)}
 .meta{color:var(--sub);font-size:12px}
 .nocara{color:#d88} .modelo{color:var(--sub);font-size:12px;margin-top:6px}
</style></head><body>
<header><h1>Reconocimiento facial en vivo &mdash; modelo de DeepFace (Rocha/FRED)</h1>
<small>CNN FER-2013 &middot; 7 emociones &middot; onnxruntime en la Raspberry</small></header>
<div class="wrap">
 <div class="video"><img src="/video" alt="camara"></div>
 <div class="panel">
   <div class="card">
     <div class="meta" id="estadoCara">esperando camara...</div>
     <div class="big" id="emo">--</div>
     <div class="conf" id="conf"></div>
   </div>
   <div class="card" id="barras"></div>
   <div class="card meta">
     <div>FPS: <b id="fps">0</b></div>
     <div class="modelo">Modelo: deepface_emotion.onnx (facial_expression_model, Serengil)</div>
   </div>
 </div>
</div>
<script>
const ORD=["felicidad","tristeza","enojo","sorpresa","miedo","asco","neutral"];
function barras(p){let h="";for(const k of ORD){const v=(p[k]||0)*100;
  h+=`<div class="row"><div class="lab">${k}</div><div class="bar">
  <span style="width:${v.toFixed(0)}%"></span></div><div class="pct">${v.toFixed(0)}%</div></div>`;}
  document.getElementById("barras").innerHTML=h;}
async function tick(){try{const r=await fetch("/estado");const d=await r.json();
  const ec=document.getElementById("estadoCara");
  if(d.cara){ec.textContent="cara detectada";ec.className="meta";}
  else{ec.textContent="no se ve ninguna cara";ec.className="meta nocara";}
  document.getElementById("emo").textContent=d.cara?d.emocion:"--";
  document.getElementById("conf").textContent=d.cara?("confianza "+(d.confianza*100).toFixed(0)+"%"):"";
  document.getElementById("fps").textContent=d.fps.toFixed(1);
  barras(d.probs||{});}catch(e){}}
setInterval(tick,300);tick();
</script></body></html>"""


@app.route("/")
def index():
    return render_template_string(PAGINA)


@app.route("/estado")
def estado():
    with _lock:
        return jsonify(dict(_estado))


@app.route("/video")
def video():
    def gen():
        while True:
            data = _ultimo_jpeg["data"]
            if data is None:
                time.sleep(0.05)
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
            time.sleep(0.04)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    print(f"[deepface] modelo emocion: {MODELO_EMOCION}")
    print(f"[deepface] modelo cara   : {MODELO_CARA}")
    print(f"[deepface] abre http://<IP-de-la-Pi>:{PUERTO}  (apaga Romeo antes)")
    threading.Thread(target=_bucle_camara, daemon=True).start()
    app.run(host="0.0.0.0", port=PUERTO, threaded=True)
