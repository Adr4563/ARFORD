#!/usr/bin/env python3
r"""Pagina para probar la capa afectiva sin la Pi: FER + SER + TER, la fusion
en valencia/arousal (PULSE) y el animo de Romeo (ROSE). La camara y el micro
son los del navegador.

Usa el MISMO stt_server.py que atiende a la Pi (STT, SER y TER) y el MISMO
estado_afectivo.py que el orquestador; solo le agrega la pagina en "/" y el
endpoint /turno, que hace por cada frase lo que hace orchestrator_node:

    TER  acumular_ter + resumir_ter (ventana de 5 frases) -> percibir_categorias
    FER  cuadros de la frase -> mediana normalizada -> percibir_categorias
    SER  percibir_categorias
    si todas salen neutral no se toca el animo; si no, fusionar y empatizar

Se corre en la PC:

    python pruebas/web/servidor.py

y se abre desde cualquier equipo de la red en https://<ip-de-la-PC>:8443.

Va por HTTPS porque el navegador solo deja usar micro y camara en https o en
localhost: por http://<ip> la pagina carga pero no graba. El certificado es
autofirmado, asi que la primera vez el navegador avisa ("Avanzado ->
continuar"). Usa el 8443 y no el 8082 para no chocar con un stt_server.py
normal (http) que este corriendo para la Pi.

El FER no se importa de _emotion_detector.py porque ese modulo necesita ROS
(ament_index_python); se repite aqui su _clasificar y su resumir, con los
mismos dos modelos de romeo_drivers/data/modelos/.

El TER se lee de modelos/ter/ del repo, asi no hace falta correr antes
modelos/instalar_modelos.sh. El SER sale de la cache de HuggingFace.

La respuesta (/responder) sigue el orden del orquestador: aeb.mapear sobre
el animo ya actualizado da el tono, y agente_chat.responder contesta con
ese tono. El chat va por Ollama con CHAT_MODEL, que aqui es gemma3:4b:
romeo-llama3b-v3 no esta creado en esta maquina, y como los tonos del AEB
ya no traen frases para copiar, es el unico de los probados que responde
bien a su juicio (gemma2:2b decia "no te preocupes" a alguien a quien le
pegan e inventaba nombres).
Gemma esta en el Ollama de D:\ollama-models, que se levanta aparte en el
11435 porque el de siempre (11434) guarda en C:, que esta lleno:

    set OLLAMA_MODELS=D:\ollama-models& set OLLAMA_HOST=127.0.0.1:11435& ollama serve

/hablar lee la respuesta con la voz de tts_server.py.
"""

import functools
import os
import re
import socket
import unicodedata
import sys
import threading

AQUI = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(AQUI, "..", ".."))
SRC = os.path.join(REPO, "ros2_ws", "src")
os.environ.setdefault("ROMEO_TER_DIR", os.path.join(REPO, "modelos", "ter"))
os.environ.setdefault("ROMEO_STT_HOST", "0.0.0.0")
os.environ.setdefault("ROMEO_STT_PUERTO", "8443")
os.environ.setdefault("CHAT_MODEL", "gemma3:4b")
os.environ.setdefault("CHAT_SERVER_HOST", "http://127.0.0.1:11435")
sys.path.insert(0, os.path.join(SRC, "romeo_drivers", "scripts"))
sys.path.insert(0, os.path.join(SRC, "romeo_brain"))
# El juez de la trivia usa el mismo modelo que el chat: "romeo" (el de
# agente_trivia) no esta creado en esta maquina.
os.environ.setdefault("TRIVIA_CORRECTOR_MODEL", os.environ["CHAT_MODEL"])

# preguntas.py y frases.py buscan sus datos con get_package_share_directory
# de ROS, que aqui no esta. Se sustituye por uno que apunta a la carpeta del
# paquete en el repo, donde esta el mismo data/ que se instala.
import types  # noqa: E402

_ament = types.ModuleType("ament_index_python")
_ament.packages = types.ModuleType("ament_index_python.packages")
_ament.packages.get_package_share_directory = lambda nombre: os.path.join(SRC, nombre, nombre)
sys.modules.setdefault("ament_index_python", _ament)
sys.modules.setdefault("ament_index_python.packages", _ament.packages)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from flask import Response, jsonify, request, send_from_directory  # noqa: E402

from romeo_brain import (aeb, agente_chat, agente_trivia, estado_afectivo,  # noqa: E402
                         frases, llama_client, preguntas)
import stt_server  # noqa: E402
import tts_server  # noqa: E402

# ── FER: lo mismo que _emotion_detector.py ──────────────────────────────
MODELOS_FER = os.path.join(SRC, "romeo_drivers", "romeo_drivers", "data", "modelos")
ETIQUETAS_FER = ["neutral", "felicidad", "sorpresa", "tristeza", "enojo", "asco", "miedo", "desprecio"]
_op = ort.SessionOptions()
_op.intra_op_num_threads = 2
_op.inter_op_num_threads = 1
_sess_fer = ort.InferenceSession(os.path.join(MODELOS_FER, "emotion-ferplus-8.onnx"), _op)
_entrada_fer = _sess_fer.get_inputs()[0].name
_yunet = cv2.FaceDetectorYN_create(
    os.path.join(MODELOS_FER, "face_detection_yunet_2023mar.onnx"), "", (320, 320),
    score_threshold=0.7)


def clasificar_cara(bgr):
    """Las 8 probabilidades de FER+ de la cara mas clara, o None sin cara."""
    alto, ancho = bgr.shape[:2]
    _yunet.setInputSize((ancho, alto))
    _, caras = _yunet.detect(bgr)
    if caras is None or len(caras) == 0:
        return None
    x, y, w, h = [max(0, int(v)) for v in max(caras, key=lambda c: c[14])[:4]]
    recorte = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)[y:y + h, x:x + w]
    if recorte.size == 0:
        return None
    recorte = cv2.resize(recorte, (64, 64)).astype(np.float32)
    salida = _sess_fer.run(None, {_entrada_fer: recorte.reshape(1, 1, 64, 64)})[0][0]
    e = np.exp(salida - np.max(salida))
    return [float(p) for p in e / e.sum()]


def resumir_cuadros(cuadros):
    """Mediana de cada emocion y normalizacion (tesis, sec. 4.1.4.1)."""
    if not cuadros:
        return None
    mediana = np.median(np.asarray(cuadros, dtype=float), axis=0)
    total = float(mediana.sum())
    return [float(p) / total for p in mediana] if total > 0.0 else None


# ── ROSE: un animo por servidor, como una sesion del robot ──────────────
afecto = estado_afectivo.EstadoAfectivo(perfil=estado_afectivo.PERFIL_POR_DEFECTO)
_lock_afecto = threading.Lock()


def _vad(v):
    return None if v is None else {"v": round(v[0], 3), "a": round(v[1], 3), "d": round(v[2], 3)}


def _animo():
    # La cara la decide el AEB, no afecto.cara(): con tristeza el robot pone
    # la cara normal (aeb.OCULTAR_CARA_TRISTE), y aqui tiene que verse igual.
    return {**_vad(afecto.vector), "cara": aeb.mapear(afecto)["cara"]}


def _cuatro(probs, etiquetas):
    """Las cuatro emociones del robot, para mostrarlas igual en las tres fuentes."""
    if not probs:
        return None
    claves, valores = estado_afectivo.agrupar_en_cuatro(probs, etiquetas)
    total = sum(valores) or 1.0
    return {"etiquetas": claves, "probabilidades": [v / total for v in valores]}


@stt_server.app.get("/")
def pagina():
    return send_from_directory(AQUI, "index.html")


@stt_server.app.get("/animo")
def animo():
    with _lock_afecto:
        afecto.decaer()
        return jsonify({"animo": _animo(), "base": _vad(afecto.animo_base),
                        "perfil": afecto.perfil, "empatia": afecto.empatia,
                        "aeb": _aeb()})


@stt_server.app.post("/reiniciar")
def reiniciar():
    """Sesion nueva: animo base y ventana del TER vacia."""
    with _lock_afecto:
        afecto.reiniciar()
        estado_afectivo.olvidar_ter()
        agente_chat.reiniciar()
        trivia.update(activa=False, pendiente=None, cola=[], usados=set(), aciertos=0, total=0)
        return jsonify({"animo": _animo(), "aeb": _aeb()})


def _aeb():
    """El AEB del animo actual, sin las claves que la pagina no usa."""
    comp = aeb.mapear(afecto)
    return {k: comp[k] for k in ("estado", "emocion", "nivel", "cara", "tono", "movimiento")}


# ── Trivia: el mismo flujo que orchestrator_node ───────────────────────
# Se entra SOLO nombrando la trivia (_PEDIR_TRIVIA de orchestrator_node,
# con las variantes que suelta el STT); las preguntas salen mezcladas de
# preguntas.jsonl y se encadenan hasta que se pide salir (frases.SALIR_TRIVIA
# o TEMA_PERSONAL). El juez es agente_trivia.juzgar. Se repite aqui porque
# orchestrator_node importa rclpy.
_PEDIR_TRIVIA = re.compile(r"\b(trivia|trivias|tribia|tribias|trivi|tribu)\b", re.I)
_PALABRAS_RESPUESTA_CORTA = 4  # mismo umbral que orchestrator_node
trivia = {"activa": False, "pendiente": None, "cola": [], "usados": set(),
          "aciertos": 0, "total": 0}


def _sin_tildes(texto):
    t = unicodedata.normalize("NFD", (texto or "").lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def _siguiente_pregunta():
    """La proxima pregunta (y la deja pendiente), o None si se acabaron.
    Las de reconocimiento musical se saltan: la cancion es el enunciado y
    solo suena en el robot."""
    while True:
        if not trivia["cola"]:
            mas = preguntas.preguntas_mezcladas(trivia["usados"], cantidad=5)
            if not mas:
                return None
            trivia["usados"].update(q["id"] for q in mas)
            trivia["cola"] = mas
        q = trivia["cola"].pop(0)
        if not q.get("musical"):
            trivia["pendiente"] = q
            return q


def _estado_trivia(**extra):
    return {"activa": trivia["activa"], "aciertos": trivia["aciertos"],
            "total": trivia["total"],
            "pregunta": trivia["pendiente"]["pregunta"] if trivia["pendiente"] else None,
            **extra}


def _turno_trivia(texto):
    """(respuesta, info) si el mensaje es de la trivia, o None si va al chat."""
    if trivia["activa"]:
        salir = texto.lower()
        if any(f in salir for f in frases.SALIR_TRIVIA) or any(f in salir for f in frases.TEMA_PERSONAL):
            trivia["activa"] = False
            trivia["pendiente"] = None
            marcador = (f"Acertaste {trivia['aciertos']} de {trivia['total']}. "
                        if trivia["total"] else "")
            # Como en el robot: si ademas de salir dijo algo de fondo, eso
            # lo contesta el chat.
            if len(texto.split()) > _PALABRAS_RESPUESTA_CORTA:
                return marcador, _estado_trivia(salio=True, sigue_chat=True)
            return (marcador + "Listo, dejamos la trivia. ¿De qué quieres que hablemos?",
                    _estado_trivia(salio=True))
        q = trivia["pendiente"]
        acerto, comentario = agente_trivia.juzgar(q["pregunta"], q["respuesta_esperada"], texto)
        trivia["total"] += 1
        trivia["aciertos"] += acerto
        if not comentario:
            comentario = frases.una(frases.RESPUESTAS_TRIVIA_ACIERTO if acerto
                                    else frases.RESPUESTAS_TRIVIA_ERROR,
                                    esperada=q["respuesta_esperada"])
        cara = agente_trivia.elegir_cara_pregunta(q, acerto) or ("happy" if acerto else "sad")
        siguiente = _siguiente_pregunta()
        if siguiente is None:
            trivia["activa"] = False
            trivia["pendiente"] = None
            cierre = f" Se acabaron las preguntas. Acertaste {trivia['aciertos']} de {trivia['total']}."
            return comentario + cierre, _estado_trivia(acerto=acerto, esperada=q["respuesta_esperada"], cara=cara)
        return (f"{comentario} {siguiente['pregunta']}",
                _estado_trivia(acerto=acerto, esperada=q["respuesta_esperada"], cara=cara))
    if _PEDIR_TRIVIA.search(_sin_tildes(texto)):
        trivia.update(activa=True, cola=[], aciertos=0, total=0)
        q = _siguiente_pregunta()
        return f"{frases.una(frases.INICIO_TRIVIA)} {q['pregunta']}", _estado_trivia(empezo=True)
    return None


@stt_server.app.post("/responder")
def responder():
    """Lo que contesta Romeo a `texto`. Si es la trivia, el flujo de arriba;
    si no, como _manejar_chat_libre: el tono sale del AEB sobre el animo
    actual (el que dejo el /turno de esta misma frase), y aeb.tono_para le
    suma su animo si le preguntan como esta."""
    texto = ((request.get_json(silent=True) or {}).get("texto") or "").strip()
    if not texto:
        return jsonify({"error": "falta el texto"}), 400
    with _lock_afecto:
        afecto.decaer()
        p = aeb.mapear(afecto)
    comp = {k: p[k] for k in ("estado", "emocion", "nivel", "cara", "tono", "movimiento")}
    pregunta_animo = bool(aeb.PREGUNTA_ANIMO.search(texto))
    try:
        de_trivia = _turno_trivia(texto)
        info = None
        if de_trivia and not de_trivia[1].get("sigue_chat"):
            respuesta, info = de_trivia
        else:
            respuesta = agente_chat.responder(texto, tono=aeb.tono_para(p, texto))
            if de_trivia:
                info = de_trivia[1]
                respuesta = f"{de_trivia[0]}{respuesta}"
    except Exception as e:  # noqa: BLE001 - igual que orchestrator_node.responder
        return jsonify({"error": f"el chat no respondió ({e})"}), 502
    return jsonify({"respuesta": respuesta, "modelo": llama_client.CHAT_MODEL,
                    "aeb": comp, "pregunta_animo": pregunta_animo, "trivia": info})


@stt_server.app.get("/hablar")
def hablar():
    texto = (request.args.get("texto") or "").strip()
    if not texto or tts_server._tts is None:
        return Response(b"", mimetype="audio/wav")
    with tts_server._lock:
        audio = tts_server._tts.generate(texto, sid=0)
    muestras = np.asarray(audio.samples, dtype=np.float32)
    if muestras.size == 0:
        return Response(b"", mimetype="audio/wav")
    return Response(tts_server._a_wav(muestras, audio.sample_rate), mimetype="audio/wav")


@stt_server.app.post("/turno")
def turno():
    if not stt_server.reconocedor.listo:
        return jsonify({"error": "el modelo no está cargado"}), 503
    if "audio" not in request.files:
        return jsonify({"error": "no llegó audio"}), 400
    datos, motivo = stt_server._wav_a_samples(request.files["audio"].read())
    if datos is None:
        return jsonify({"error": motivo}), 400
    samples, sr = datos

    with stt_server._lock:
        texto = stt_server.reconocedor.transcribir(samples, sr)
        ser = stt_server.clasificador_voz.clasificar(samples, sr) if texto else None
        ter = stt_server.clasificador_texto.clasificar(texto) if texto else None
        cuadros = []
        for foto in request.files.getlist("fotos"):
            bgr = cv2.imdecode(np.frombuffer(foto.read(), np.uint8), cv2.IMREAD_COLOR)
            probs = clasificar_cara(bgr) if bgr is not None else None
            if probs is not None:
                cuadros.append(probs)
    n_fotos = len(request.files.getlist("fotos"))
    fer = resumir_cuadros(cuadros)

    with _lock_afecto:
        afecto.decaer()
        antes = _animo()
        lecturas = []

        def agregar(fuente, probs, etiquetas):
            vector, c, dom = estado_afectivo.percibir_categorias(probs, etiquetas)
            if vector is not None:
                lecturas.append((fuente, vector, c, dom))

        # Mismo orden que orchestrator_node: TER, FER, SER.
        ter_ventana = None
        if ter:
            estado_afectivo.acumular_ter(ter[1], ter[0])
            etiq, probs = estado_afectivo.resumir_ter()
            ter_ventana = _cuatro(probs, etiq)
            agregar("TER", probs, etiq)
        if fer:
            agregar("FER", fer, ETIQUETAS_FER)
        if ser:
            agregar("SER", ser[1], ser[0])

        fusion, motivo_fusion = None, None
        if not lecturas:
            motivo_fusion = "no hubo ninguna lectura"
        elif all(dom == "neutral" for *_x, dom in lecturas):
            motivo_fusion = "todas las fuentes salieron neutral: el ánimo no se mueve"
        else:
            fusion = estado_afectivo.fusionar([(v, c) for _f, v, c, _d in lecturas])
            afecto.empatizar(fusion, max(c for *_x, c, _d in lecturas))
        despues = _animo()
        comp = _aeb()

    return jsonify({
        "texto": texto,
        "ser": _cuatro(ser[1], ser[0]) if ser else None,
        "ter": _cuatro(ter[1], ter[0]) if ter else None,
        "ter_ventana": ter_ventana,
        "fer": _cuatro(fer, ETIQUETAS_FER),
        "fer_cuadros": len(cuadros),
        "fer_fotos": n_fotos,
        "lecturas": [{"fuente": f, **_vad(v), "confianza": round(c, 3), "dominante": d}
                     for f, v, c, d in lecturas],
        "fusion": _vad(fusion),
        "motivo_fusion": motivo_fusion,
        "animo_antes": antes,
        "animo": despues,
        "aeb": comp,
        "base": _vad(afecto.animo_base),
        "duracion_audio": round(len(samples) / float(sr), 2),
    })


def ip_local():
    # No manda nada: solo pregunta que interfaz usaria para salir a la red.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("192.168.0.1", 80))
        return s.getsockname()[0]


if __name__ == "__main__":
    # stt_server.main() carga los modelos y llama a app.run(); aqui solo se le
    # agrega el certificado, para no copiar toda la carga.
    stt_server.app.run = functools.partial(stt_server.app.run, ssl_context="adhoc")
    if not tts_server.cargar():
        print("[WEB] sin voz: Romeo responde solo con texto")
    print(f"[WEB] chat con {llama_client.CHAT_MODEL} en {llama_client.CHAT_SERVER_HOST}")
    print(f"[WEB] abrir https://{ip_local()}:{stt_server.PUERTO}")
    sys.exit(stt_server.main())
