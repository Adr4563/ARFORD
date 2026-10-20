# -*- coding: utf-8 -*-
"""agente_trivia -- todo lo del juego de preguntas y del juego de caras.

Reune lo que antes estaba repartido en agent_corrector.py (evaluar la
respuesta del usuario) y agent_behavior.py (elegir cara, musica y
desplazamiento segun la pregunta). Son dos caras del mismo trabajo y nunca
se usan por separado: el orquestador llamaba a los dos en el mismo turno.

No hay ningun LLM aca: la correccion es determinista (comparacion
normalizada de texto y numeros) y la expresion sale de los campos de la
propia pregunta en preguntas.jsonl.
"""

# ─── Correccion de respuestas (venia de agent_corrector.py) ─────────
import difflib
import os
import re
import unicodedata

from . import llama_client

# Modelo que revisa las respuestas de trivia que la comparacion de texto da
# por incorrectas (ver _evaluar_con_llm). Vacio = sin segunda opinion.
CORRECTOR_LLM = os.environ.get("TRIVIA_CORRECTOR_MODEL", "romeo")

# Los ejemplos no son adorno: sin ellos un modelo de 1B DESCRIBE la
# instruccion en vez de cumplirla ("Felicita a tu interlocutor...") o se
# disculpa como si el error fuera suyo. Con dos ejemplos contesta en el tono
# correcto.
_PROMPT_COMENTARIO = (
    "Eres Romeo, un robot que juega trivia. Reacciona en UNA frase corta a lo "
    "que acaba de contestar la persona, hablandole de tu. Si acerto, "
    "felicitala; si fallo, dile cual era la correcta sin regañarla. No "
    "expliques nada mas.\n"
    "\n"
    "Ejemplo 1:\n"
    "Pregunta: Cual es la capital de Francia?\n"
    "Respuesta correcta: Paris\n"
    "La persona dijo: Paris, obvio\n"
    "(acerto)\n"
    "Romeo: ¡Esa estaba facil para ti! Paris, correcto.\n"
    "\n"
    "Ejemplo 2:\n"
    "Pregunta: Cuantas patas tiene una araña?\n"
    "Respuesta correcta: 8\n"
    "La persona dijo: seis creo\n"
    "(fallo)\n"
    "Romeo: Casi, pero son 8. Confundiste con los insectos."
)


def comentar(pregunta, esperada, respuesta_usuario, acerto):
    """Comentario del veredicto hecho a medida de lo que dijo el usuario.

    Devuelve None si no hay LLM o falla: entonces el caller usa las frases
    fijas de siempre. Antes SIEMPRE se usaban esas, asi que daba igual lo que
    hubiera contestado la persona -- si decia "3 por 8 es 28" oia "No, era
    24" y nada sobre su cuenta.
    """
    if not CORRECTOR_LLM:
        return None
    try:
        texto = llama_client.generar_respuesta(
            [{"role": "system", "content": _PROMPT_COMENTARIO},
             {"role": "user",
              "content": f"Pregunta: {pregunta}\n"
                         f"Respuesta correcta: {esperada}\n"
                         f"La persona dijo: {respuesta_usuario}\n"
                         f"({'acerto' if acerto else 'fallo'})"}],
            temperature=0.2, max_tokens=60, modelo=CORRECTOR_LLM).strip()
    except Exception as e:  # noqa: BLE001 - se cae a las frases fijas
        print(f"[trivia] no se pudo comentar ({e})")
        return None
    # El modelo copia el formato de los ejemplos y a veces antepone
    # "Romeo:" a su propia respuesta.
    if texto.lower().startswith("romeo:"):
        texto = texto.split(":", 1)[1].strip()
    texto = texto.strip('"').strip()

    # Una respuesta vacia o kilometrica no sirve: mejor la frase fija.
    if not texto or len(texto) > 200:
        return None
    # Si fallo, el comentario TIENE que decir cual era la correcta; es lo
    # unico que el usuario necesita saber. Si el modelo se la olvido, se
    # añade en vez de descartar todo el comentario.
    if not acerto and str(esperada).lower() not in texto.lower():
        texto = f"{texto} La respuesta era {esperada}."
    return texto


_PROMPT_CORRECTOR = (
    "Eres un corrector de trivia. Te doy la respuesta correcta y lo que dijo "
    "el usuario. Responde UNA sola palabra: SI si lo que dijo significa lo "
    "mismo que la respuesta correcta (aunque lo diga con otras palabras, le "
    "falten tildes o esté mal transcrito), o NO si no coincide. No expliques "
    "nada."
)

_ARTICULOS = {"el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al"}
_UMBRAL_SOLAPAMIENTO = 0.6
_UMBRAL_FUZZY = 0.8


def _normalizar(s):
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[¿?¡!.,;:]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _palabras_significativas(s):
    return [w for w in _normalizar(s).split() if w not in _ARTICULOS and len(w) > 1]


def _es_numerico(s):
    return bool(re.fullmatch(r"-?\d+([.,]\d+)?", s.strip()))


# Numeros escritos con letras. El STT transcribe lo que oye, asi que a la
# pregunta "cuantas patas tiene una arania" el usuario contesta "ocho" y
# llegaba como texto, no como 8. Hasta el 30 y las decenas cubre de sobra las
# respuestas del banco de preguntas.
_NUMEROS_TEXTO = {
    "cero": 0, "uno": 1, "una": 1, "un": 1, "dos": 2, "tres": 3, "cuatro": 4,
    "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10,
    "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15,
    "dieciseis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19,
    "veinte": 20, "veintiuno": 21, "veintidos": 22, "veintitres": 23,
    "veinticuatro": 24, "veinticinco": 25, "veintiseis": 26,
    "veintisiete": 27, "veintiocho": 28, "veintinueve": 29, "treinta": 30,
    "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70,
    "ochenta": 80, "noventa": 90, "cien": 100, "ciento": 100, "mil": 1000,
}


def _extraer_numeros(s):
    """Numeros de `s`, tanto en cifras como escritos con letras."""
    numeros = [float(n.replace(",", ".")) for n in re.findall(r"-?\d+(?:[.,]\d+)?", s)]

    palabras = _normalizar(s).split()
    for i, p in enumerate(palabras):
        if p not in _NUMEROS_TEXTO:
            continue
        valor = _NUMEROS_TEXTO[p]
        # "cuarenta y dos" = 40 + 2. Solo para decenas, que es como se dice
        # en español; "ciento cuarenta y dos" tambien sale bien porque el 100
        # se suma aparte.
        if valor >= 20 and valor % 10 == 0 and i + 2 < len(palabras) \
                and palabras[i + 1] == "y" and palabras[i + 2] in _NUMEROS_TEXTO:
            unidad = _NUMEROS_TEXTO[palabras[i + 2]]
            if unidad < 10:
                numeros.append(float(valor + unidad))
                continue
        numeros.append(float(valor))
    return numeros


def evaluar_respuesta(esperada, respuesta_usuario):
    """Compara la respuesta del usuario con la esperada. Devuelve True/False."""
    esperada = (esperada or "").strip()
    respuesta_usuario = (respuesta_usuario or "").strip()
    if not esperada or not respuesta_usuario:
        return False

    if _es_numerico(esperada):
        numero_esperado = float(esperada.replace(",", "."))
        if numero_esperado in _extraer_numeros(respuesta_usuario):
            return True
        # Sin return directo: si el numero no aparece tal cual todavia puede
        # estar dicho con letras ("cuarenta y dos") o mal transcrito por el
        # STT, y de eso se encarga el corrector de abajo.
        return _evaluar_con_llm(esperada, respuesta_usuario)

    esp_norm = _normalizar(esperada)
    resp_norm = _normalizar(respuesta_usuario)

    if esp_norm in resp_norm:
        return True

    palabras_esp = _palabras_significativas(esperada)
    palabras_resp = set(_palabras_significativas(respuesta_usuario))
    if palabras_esp:
        coincididas = sum(
            1 for p in palabras_esp
            if p in palabras_resp or any(
                difflib.SequenceMatcher(None, p, w).ratio() >= 0.84 for w in palabras_resp
            )
        )
        if coincididas / len(palabras_esp) >= _UMBRAL_SOLAPAMIENTO:
            return True

    if difflib.SequenceMatcher(None, esp_norm, resp_norm).ratio() >= _UMBRAL_FUZZY:
        return True

    # Ultimo recurso: preguntarle al LLM. La comparacion de arriba acierta
    # cuando el usuario dice la respuesta tal cual o parecida, pero no cuando
    # la dice de otra forma ("el pintor de la Gioconda" por "Leonardo da
    # Vinci") ni cuando el STT la transcribe mal. Solo se llega aca si la
    # comparacion ya dijo que NO, asi que el LLM nunca puede convertir un
    # acierto en fallo, solo rescatar un acierto que se habria perdido.
    #
    # Si no hay LLM configurado o no contesta, se queda en False: el juego
    # nunca se cuelga esperando a Ollama.
    return _evaluar_con_llm(esperada, respuesta_usuario)


def _evaluar_con_llm(esperada, respuesta_usuario):
    """True si el LLM considera que la respuesta es correcta. False ante
    cualquier duda o fallo."""
    if not CORRECTOR_LLM:
        return False
    try:
        veredicto = llama_client.generar_respuesta(
            [{"role": "system", "content": _PROMPT_CORRECTOR},
             {"role": "user",
              "content": f"Respuesta correcta: {esperada}\n"
                         f"Lo que dijo el usuario: {respuesta_usuario}\n"
                         f"¿SI o NO?"}],
            temperature=0.0, max_tokens=4, modelo=CORRECTOR_LLM).strip().upper()
    except Exception as e:  # noqa: BLE001 - el juego sigue sin el LLM
        print(f"[trivia] el corrector LLM falló ({e}), se queda en incorrecto")
        return False
    acierto = veredicto.startswith("SI") or veredicto.startswith("SÍ")
    # Se loguea siempre: sin esto no habia forma de saber si el corrector
    # llego a intervenir ni que decidio.
    print(f"[trivia] {CORRECTOR_LLM}: esperada={esperada!r} "
          f"usuario={respuesta_usuario!r} -> {veredicto!r} "
          f"({'rescatado' if acierto else 'sigue incorrecto'})")
    return acierto


# ─── Expresion: cara, musica, motores (venia de agent_behavior.py) ──
# El dataset usa nombres en español (columna 'cara'); FaceCommand.face_name
# espera las claves en inglés de faces/*.mp4|*.gif. "Neutral" mapea a
# "content" porque es la misma cara de reposo -- no hay un asset de
# "neutral" aparte, igual que en el original.
_CARA_A_DISPLAY = {
    "Feliz": "happy",
    "Triste": "sad",
    "Enojado": "angry",
    "Neutral": "content",
}

# Traduce el texto en español de la columna 'desplazamiento' del dataset al
# comando que entiende MotionCommand.msg (que a su vez es casi 1:1 con el
# protocolo del firmware -- ver romeo_interfaces/msg/MotionCommand.msg).
# Esta tabla vivía en Clients/Carrito_Client.py en el proyecto original;
# se mueve acá porque es lógica de NEGOCIO (qué significa "Izquierda" en
# el dataset), no de protocolo de hardware -- moving_control_node ya no sabe nada
# de español, solo de F/B/SL/SR/RL/RR/S/ROTATE360.
_DIRECCION_A_COMANDO = {
    "adelante": "F",
    "atrás": "B",
    "atras": "B",
    "izquierda": "SL",
    "derecha": "SR",
}


def elegir_cara_pregunta(pregunta, acerto):
    """Cara para el veredicto de una pregunta CON respuesta_esperada."""
    campo = "cara_respuesta_buena" if acerto else "cara_respuesta_mala"
    valor = (pregunta.get(campo) or "").strip()
    return _CARA_A_DISPLAY.get(valor)


def cara_para_emocion(nombre):
    """Traduce Feliz/Triste/Enojado/Neutral (columna 'cara') a la clave que
    entiende FaceCommand -- usado por el Juego de emociones/imitación para
    mostrar la cara ANTES de pedirle al usuario que la imite."""
    return _CARA_A_DISPLAY.get((nombre or "").strip())


def expresar_musica(pregunta, bridge, esperar=False, logger=None):
    """Si la pregunta trae algo en 'musical', le pide al `bridge` que la
    reproduzca (PlayMusic.srv vía communication_node). Vacío/ausente -> no
    hace nada, devuelve None. `esperar=True` para Reconocimiento Musical
    (la música es el enunciado, hay que escucharla completa antes de
    responder) -- ver la nota en romeo_interfaces/srv/PlayMusic.srv."""
    musica = (pregunta.get("musical") or "").strip()
    if not musica:
        return None
    if logger is not None:
        logger.info(f"[música] {musica}")
    bridge.reproducir_musica(musica, esperar=esperar)
    return musica


def expresar_desplazamiento(pregunta, bridge, logger=None):
    """Si la pregunta trae algo en 'desplazamiento', le pide al `bridge`
    que publique el MotionCommand correspondiente. Vacío/ausente -> no hace
    nada, devuelve None."""
    desplazamiento = (pregunta.get("desplazamiento") or "").strip()
    if not desplazamiento:
        return None
    if logger is not None:
        logger.info(f"[desplazamiento] {desplazamiento}")

    if desplazamiento.lower().startswith("girar"):
        comando = "ROTATE360"
    else:
        comando = _DIRECCION_A_COMANDO.get(desplazamiento.strip().lower())
    if not comando:
        if logger is not None:
            logger.warning(f"[desplazamiento] valor sin comando mapeado: {desplazamiento!r}")
        return None

    bridge.mover(comando)
    return desplazamiento
