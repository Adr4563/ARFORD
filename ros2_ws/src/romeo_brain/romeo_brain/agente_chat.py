# -*- coding: utf-8 -*-
"""agente_chat -- la conversacion libre.

Es el unico de los dos agentes que usa un LLM: le pasa el mensaje del
usuario a CHAT_MODEL (por Ollama) y devuelve lo que responda.

Sin system prompt a proposito: el comportamiento viene horneado en el
fine-tuning del modelo.

MANTIENE EL HILO. Antes se mandaba un unico mensaje por turno, sin nada de
lo anterior, asi que el modelo no podia resolver referencias:

    Usuario: me gustan los robots que hablan
    Romeo:   ¿Te gustan mas los que parecen humanos o los robóticos?
    Usuario: los robóticos
    Romeo:   (sin contexto, no sabia de que se hablaba)

Ahora se envian los ultimos TURNOS_HISTORIAL turnos. El limite existe
porque cada turno son tokens que el modelo tiene que leer: con un 1B en
una Pi, un historial largo cuesta latencia y acaba desbordando la ventana.

El historial es de la SESION: se borra con reiniciar(), que el orquestador
llama al empezar una sesion nueva. Para recuperar cosas de conversaciones
pasadas haria falta memoria episodica, que es otra cosa y no esta.
"""
import difflib
import os
import re

from . import llama_client, seguridad

# Cuantos turnos (par usuario+Romeo) se recuerdan. 6 son ~12 mensajes, que
# con respuestas cortas caben de sobra en la ventana de un modelo 1B.
TURNOS_HISTORIAL = int(os.environ.get("ROMEO_TURNOS_HISTORIAL", "6"))

# Maximo de frases en una respuesta de chat. El modelo ignora el prompt y
# responde con parrafos largos; esto lo corta por codigo (ver
# _limpiar_formato). Cambiable con ROMEO_MAX_FRASES sin recompilar.
# Creatividad del chat. 0.7 y no el 0.3 de llama_client: los tonos del AEB
# ya no traen frases para copiar, y con 0.3 Romeo contestaba casi igual a
# cada mensaje parecido.
TEMPERATURA = float(os.environ.get("ROMEO_TEMPERATURA_CHAT", "0.7"))

# 3 y no 2: los tonos del AEB piden reconocer lo que siente, animar y seguir
# la conversacion, y con 2 la respuesta se cortaba antes del animo.
MAX_FRASES = int(os.environ.get("ROMEO_MAX_FRASES", "3"))

# Cuando piden un CONSEJO, el prompt de base ("no ofrezcas ayuda, solo
# conversa") hacia que Romeo contestara con otra pregunta ("¿Que te hace
# sentir triste?"), y aunque los diera, MAX_FRASES los cortaba a la mitad.
# Ese turno se le pide que ayude de verdad y se le dejan mas frases.
MAX_FRASES_CONSEJO = int(os.environ.get("ROMEO_MAX_FRASES_CONSEJO", "4"))
_PIDE_CONSEJO = re.compile(
    r"consejo|aconsej|recomi[eé]nd|recomiendas|tips?\b|ideas? para|"
    r"qu[eé] (puedo|debo|deber[ií]a|tengo que) hacer|"
    r"c[oó]mo (puedo|hago|hacer|deber[ií]a)|ay[uú]dame a",
    re.IGNORECASE)
_PROMPT_CONSEJO = (
    "Te estan pidiendo un consejo: esta vez SI ayuda. Da dos o tres consejos "
    "concretos y faciles de hacer, dichos en frases cortas y seguidas, sin "
    "listas ni numeros, y no respondas con otra pregunta.")

# Cuando preguntan algo FACTUAL (una definicion, un dato, "que es X", "por
# que Y", "quien fue Z"), el prompt de base ("solo conversa, no ofrezcas
# ayuda") hacia que Romeo esquivara con otra pregunta en vez de responder.
# Ese turno se le deja contestar de verdad, breve y hablado, y se le pide que
# admita cuando NO sabe en vez de inventar (un modelo chico alucina en temas
# que no domina). Asi Romeo es versatil: conversa, pero tambien informa.
MAX_FRASES_FACTUAL = int(os.environ.get("ROMEO_MAX_FRASES_FACTUAL", "4"))
_PIDE_DATO = re.compile(
    r"\bqu[eé] (es|son|significa|quiere decir)\b|"
    r"\bqui[eé]n(es)? (es|son|fue|fueron|invent\w*|descubri\w*)|"
    r"\bpor qu[eé]\b|\bpara qu[eé] sirve\b|"
    r"\bcu[aá]l(es)? (es|son|fue)\b|\bcu[aá]nto(s|as)?\b|"
    r"\bcu[aá]ndo (fue|paso|ocurri|empez|naci)\b|\bd[oó]nde (esta|queda|naci)\b|"
    r"\bdefin(e|icion|ir|eme)\b|\bexplic(a|ame|ar|as)\b|"
    r"\bqu[eé] diferencia\b|\bde qu[eé] (trata|esta hecho)\b|"
    r"\bh[aá]blame (de|sobre)\b|\bqu[eé] sabes (de|sobre)\b",
    re.IGNORECASE)
_PROMPT_FACTUAL = (
    "Te estan preguntando por un dato o una definicion: esta vez RESPONDE de "
    "verdad, no contestes con otra pregunta. Da la respuesta clara en dos a "
    "cuatro frases cortas, dichas en voz alta, sin listas ni numeracion. Si "
    "no conoces el tema o la pregunta es ambigua, dilo con naturalidad y pide "
    "que te aclaren, en vez de inventar datos.")

# Instruccion de estilo, SOLO para modelos sin fine-tuning.
#
# El modulo no mandaba system prompt a proposito, porque con romeo (el
# qwen entrenado con TerapIA) el comportamiento viene horneado. Pero el
# modelo por defecto es llama3.2:1b, que es un asistente generico y
# responde como tal: parrafos de cinco lineas con listas de viñetas y un
# "¿en que puedo ayudarte hoy?" al final. Eso no es una conversacion, y
# ademas se oye eterno cuando lo lee el TTS en voz alta.
#
# Se manda solo si el modelo NO es uno de los entrenados, para no pisar su
# fine-tuning. ROMEO_PROMPT_CHAT="" lo desactiva del todo.
_PROMPT_POR_DEFECTO = (
    "Eres Romeo, un robot que conversa con una persona en voz alta. "
    "Responde en una o dos frases cortas, como en una charla hablada. "
    "Nunca uses listas ni viñetas. No ofrezcas ayuda ni preguntes en qué "
    "puedes ayudar: solo conversa, y si viene a cuento haz UNA pregunta "
    "corta sobre lo que te acaban de contar."
)

_MODELOS_ENTRENADOS = ("romeo", "arford")

# Base para los modelos entrenados. NO repite la personalidad -- esa viene
# del fine-tuning y del SYSTEM del Modelfile -- solo corrige lo que el
# dataset no le enseño: EmpatheticDialogues son turnos de listener, asi
# que el 68% de sus respuestas acaban en pregunta y el modelo cierra
# SIEMPRE igual, hasta repitiendo la que acaba de hacer.
#
# romeo-llama3b-v3 ya viene con eso corregido en el dataset (rebalanceado
# al 40%): sobre ocho frases acaba preguntando 2 veces, no 8. La linea se
# queda igual porque no estorba y sigue haciendo falta para los modelos
# entrenados con el v2.
#
# Va junto al tono del AEB (ver _system_prompt): sin esto el tono llegaria
# solo, sin nada que lo encuadre.
#
# LA IDENTIDAD VA AQUI, aunque el Modelfile ya la traiga. Ollama no suma
# los dos: el system que manda el cliente SUSTITUYE al del Modelfile, y
# sin ella el modelo contesta "Soy un modelo de lenguaje conocido como
# Llama" a "¿que eres tu?".
_PROMPT_ENTRENADO = (
    "Eres Romeo, un robot amigable que conversa en voz alta con niños y "
    "jóvenes. Responde en español, tuteando, con una o dos frases cortas, "
    "con calidez y empatía. No termines siempre preguntando: a veces basta "
    "con acompañar lo que te cuentan. Nunca repitas una pregunta que ya "
    "hiciste."
)


# Como trata a la persona, con cualquier modelo. Medido con gemma3:4b
# respondiendo a su juicio: decia "Oh, cariño" a todo, y "no tienes que
# pasar por esto solo" sin saber si habla con un hombre o una mujer.
_PROMPT_TRATO = (
    "Tutea siempre a la persona, nunca le hables de usted. "
    "No uses apodos como 'cariño', 'querida' o 'amigo'. No sabes si hablas "
    "con un hombre o con una mujer: usa frases sin genero (por ejemplo 'no "
    "tienes que pasarlo a solas' en vez de 'solo' o 'sola').")


def _prompt_para(modelo):
    """Vacio si el modelo ya trae el comportamiento horneado."""
    personalizado = os.environ.get("ROMEO_PROMPT_CHAT")
    if personalizado is not None:
        return personalizado
    nombre = (modelo or "").lower()
    if any(nombre.startswith(m) for m in _MODELOS_ENTRENADOS):
        return f"{_PROMPT_ENTRENADO} {_PROMPT_TRATO}"
    return f"{_PROMPT_POR_DEFECTO} {_PROMPT_TRATO}"


_PROMPT = _prompt_para(llama_client.CHAT_MODEL)

# Tono afectivo que manda el AEB (ver aeb.py): una frase de estilo, segun
# como se siente Romeo, que se añade al system prompt.
#
#   auto  (default) solo si ya hay system prompt, o sea con modelos SIN
#         fine-tuning. A los entrenados (romeo*, arford*) no se les manda
#         system prompt a proposito, y meterles uno solo por el tono podria
#         desviar el comportamiento horneado.
#   1     siempre, tambien con los entrenados (para probar si lo respetan)
#   0     nunca
TONO_AEB = os.environ.get("ROMEO_AEB_TONO", "auto").strip().lower()

_historial = []

# Parecido a partir del cual una respuesta se da por REPETIDA (ver
# _es_repeticion). Medido con romeo-qwen-v2: con historial devolvia la misma
# frase palabra por palabra ("¡Muy bien Adrián! Me encanta escuchar tus
# historias...") a cualquier mensaje, incluido "estoy muy molesto"; sin
# historial contestaba a lo que se le decia.
_UMBRAL_REPETICION = 0.8


def _es_repeticion(texto):
    """True si `texto` se parece demasiado a algo que Romeo ya dijo en esta
    sesion."""
    return any(
        difflib.SequenceMatcher(None, texto, m["content"]).ratio() >= _UMBRAL_REPETICION
        for m in _historial if m["role"] == "assistant")


def _system_prompt(tono):
    """El system prompt de este turno, o "" si no hay que mandar ninguno."""
    usar_tono = bool(tono) and (TONO_AEB in ("1", "true", "si")
                                or (TONO_AEB == "auto" and _PROMPT))
    if usar_tono:
        return f"{_PROMPT} {tono}".strip()
    return _PROMPT


def reiniciar():
    """Olvida la conversacion. Se llama al arrancar una sesion nueva: el
    usuario siguiente no deberia heredar el hilo del anterior."""
    _historial.clear()


def sembrar_turno_robot(texto):
    """Mete en el historial algo que Romeo dijo fuera del chat.

    El saludo de apertura lo dice el orquestador, no este agente, asi que
    el chat no se entera de que acaba de preguntar el nombre. Sin ese
    turno, al oir "Adrian" a secas el modelo no ve un nombre y responde
    "¿Que tal tu?"; con el, contesta "Hola Adrian"."""
    _historial.append({"role": "assistant", "content": texto})


# Emojis y pictogramas (caras, banderas, simbolos, el selector de variante
# y el unidor de secuencias). gemma2:2b los pone al final ("¿Como fue el
# partido? 🤩 ⚽ 🏆") y la voz no los puede leer.
_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U00002B00-\U00002BFF"
    "\U0000FE0F\U0000200D]")


def _limpiar_formato(texto, max_frases=None):
    """Quita el formato de texto escrito (listas, vinetas, numeracion,
    asteriscos, almohadillas, emojis) para que suene bien dicho EN VOZ ALTA.

    El modelo a veces ignora el system prompt y responde con listas Markdown.
    Esto no es un consejo al modelo: borra los simbolos de la salida pase lo
    que pase, y une las vinetas en prosa corrida."""
    texto = _EMOJI.sub("", texto)
    lineas = []
    for linea in texto.splitlines():
        s = linea.strip()
        if not s:
            continue
        # Vineta ("* ...", "- ...", "• ...") o numeracion ("1. ...", "2) ...").
        s = re.sub(r"^\s*([*\-•]|\d+[.)])\s+", "", s)
        # Enfasis/encabezados Markdown en cualquier posicion.
        s = s.replace("*", "").replace("#", "").replace("`", "")
        s = re.sub(r"\s{2,}", " ", s).strip()
        if s:
            lineas.append(s)
    # Une todo en prosa; si una linea no cierra con puntuacion, se le pone
    # punto para que al enlazarlas no queden frases pegadas.
    partes = []
    for s in lineas:
        if s and s[-1] not in ".!?:":
            s += "."
        partes.append(s)
    texto = " ".join(partes).strip()

    # Recorte a MAX_FRASES: el modelo ignora el prompt y suelta parrafos de
    # cinco consejos. Se corta por codigo tras las primeras frases para que
    # suene breve dicho en voz alta. Se parte por . ! ? conservando el signo.
    frases = re.findall(r"[^.!?]+[.!?]+|[^.!?]+$", texto)
    frases = [f.strip() for f in frases if f.strip()]
    tope = MAX_FRASES if max_frases is None else max_frases
    if len(frases) > tope:
        frases = frases[:tope]
    return " ".join(frases).strip()


def responder(mensaje_usuario, tono=None):
    """Devuelve la respuesta de Romeo, teniendo en cuenta los turnos
    anteriores de esta sesion.

    `tono` es la instruccion de estilo del AEB (o None). Va solo en el
    system prompt de ESTE turno: no se guarda en el historial, porque el
    animo cambia de un turno a otro."""
    # Si el mensaje es grave (hacerse daño, que lo lastimen, lastimar a
    # otro), la instruccion de seguridad SUSTITUYE al tono: el del AEB diria
    # "no te preocupes, esto va a pasar". Ver seguridad.py.
    riesgo, instruccion = seguridad.revisar(mensaje_usuario)
    if riesgo:
        sistema = f"{_PROMPT} {instruccion}".strip()
    else:
        sistema = _system_prompt(tono)
    consejo = bool(riesgo) or bool(_PIDE_CONSEJO.search(mensaje_usuario or ""))
    if consejo and not riesgo:
        sistema = f"{sistema} {_PROMPT_CONSEJO}".strip()
    # Pregunta factual (definicion/dato): deja que Romeo responda de verdad.
    # El riesgo y el consejo mandan sobre esto (una peticion de ayuda grave o
    # un consejo no se tratan como una consulta de datos).
    factual = (not riesgo and not consejo
               and bool(_PIDE_DATO.search(mensaje_usuario or "")))
    if factual:
        sistema = f"{sistema} {_PROMPT_FACTUAL}".strip()

    def _generar(con_historial):
        mensajes = ((list(_historial) if con_historial else [])
                    + [{"role": "user", "content": mensaje_usuario}])
        if sistema:
            mensajes.insert(0, {"role": "system", "content": sistema})
        # Sin recorte a dos frases: en el chat libre se quiere que Romeo se
        # exprese completo. La respuesta sale tal cual la da el modelo (solo
        # limpia de espacios). El recorte de primera_frase() sigue en la
        # trivia, donde las respuestas deben ser cortas.
        return llama_client.generar_respuesta(
            mensajes, temperature=TEMPERATURA).strip()

    texto = _generar(con_historial=True)
    # Un modelo pequeño se engancha a lo ultimo que dijo y lo copia. Si pasa,
    # se pide otra vez SIN historial: pierde el hilo en este turno, pero
    # contesta a lo que el usuario acaba de decir.
    if _es_repeticion(texto):
        texto = _generar(con_historial=False)

    if consejo:
        tope_frases = MAX_FRASES_CONSEJO
    elif factual:
        tope_frases = MAX_FRASES_FACTUAL
    else:
        tope_frases = None
    texto = _limpiar_formato(texto, tope_frases)
    _historial.append({"role": "user", "content": mensaje_usuario})
    _historial.append({"role": "assistant", "content": texto})
    # Se recorta por el principio, de dos en dos, para no dejar un turno a
    # medias (un 'user' sin su 'assistant' confunde al modelo).
    del _historial[:max(0, len(_historial) - TURNOS_HISTORIAL * 2)]
    return texto
