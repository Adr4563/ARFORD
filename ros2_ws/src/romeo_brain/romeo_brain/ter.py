# -*- coding: utf-8 -*-
"""TER -- que emocion EXPRESA el texto del usuario.

Es la tercera pieza de percepcion del framework de Rocha et al., junto al
FER (cara) y el SER (voz). Aca va por palabras clave y no por LLM: el
estado afectivo se actualiza en CADA turno, y una llamada mas al modelo por
turno costaria ~250 ms de latencia y dependeria de que la PC este
levantada. Esto no cuesta nada y funciona aunque no haya red.

QUE NO ES. Esto estima lo que el texto EXPRESA, no como se siente la
persona de verdad. Alguien puede escribir "estoy bien" y no estarlo, o ser
ironico, o el STT puede haber transcrito mal. Por eso:

  - la confianza nunca pasa de CONFIANZA_MAXIMA, por muchos terminos que
    casen: unas palabras clave no dan para mas;
  - devuelve "sin evidencia" (None, 0.0) sin ningun reparo, que es lo
    honesto cuando el texto no dice nada emocional;
  - ante señales contradictorias tampoco elige: devuelve sin evidencia.

El consumidor (ROSE) promedia varios turnos, asi que una lectura suelta
equivocada no mueve gran cosa. Lo que mueve el animo es la insistencia.
"""
import os
import re
import unicodedata

# Tope de confianza. El limite es conceptual, no un parametro a subir: por
# mucho que el texto diga "estoy tristisimo y destrozado", un emparejador
# de palabras no puede estar mas seguro que esto de lo que siente alguien.
CONFIANZA_MAXIMA = float(os.environ.get("ROMEO_TER_CONFIANZA_MAX", "0.6"))

# Cuanto tiene que ganar una emocion a la siguiente para que se la crea.
# Por debajo de esto el texto es ambiguo ("estoy contento pero cansado") y
# se prefiere no decir nada antes que elegir a cara o cruz.
_MARGEN_AMBIGUEDAD = 0.34

# Palabras de negacion. Si una aparece hasta _VENTANA_NEGACION palabras
# antes de un termino emocional, ese termino NO cuenta.
#
# Ojo: negar una emocion no afirma la contraria. "No estoy triste" no
# significa "estoy feliz", asi que la evidencia se descarta y ya.
_NEGACIONES = {"no", "ni", "nunca", "jamas", "tampoco", "sin", "nada"}
_VENTANA_NEGACION = 3

# Terminos por emocion, con su peso. Las claves son las de ANCLAJES en
# estado_afectivo, para poder mapear sin traducir.
#
# Van sin tildes a proposito: el texto se normaliza antes de buscar, asi
# que "esta" casa con "está" y con "ESTA". Es importante porque esto se
# alimenta de transcripciones del STT, que se come tildes a menudo.
_TERMINOS = {
    "tristeza": {
        "triste": 1.0, "tristeza": 1.0, "deprimido": 1.0, "deprimida": 1.0,
        # "solo" NO va aca: casi siempre es el adverbio ("solo quiero
        # jugar"), no la soledad. Se trata aparte, ver _SOLO_CONTEXTO.
        "sola": 0.7, "soledad": 0.9, "llorar": 1.0,
        "llorando": 1.0, "lloro": 0.9, "mal": 0.6, "fatal": 0.9,
        "horrible": 0.8, "terrible": 0.8, "destrozado": 1.0,
        "destrozada": 1.0, "vacio": 0.8, "vacia": 0.8, "perdido": 0.6,
        "perdida": 0.6, "desanimado": 1.0, "desanimada": 1.0,
        "nostalgia": 0.8, "extraño": 0.6, "duele": 0.9, "dolor": 0.8,
        "pena": 0.9, "sufro": 1.0, "sufriendo": 1.0, "decepcionado": 0.9,
        "decepcionada": 0.9, "desilusionado": 0.9, "rendirme": 0.9,
    },
    "felicidad": {
        "feliz": 1.0, "felicidad": 1.0, "contento": 1.0, "contenta": 1.0,
        "alegre": 1.0, "alegria": 1.0, "genial": 0.9, "estupendo": 0.9,
        "increible": 0.8, "maravilloso": 0.9, "encanta": 0.9,
        "encantado": 0.8, "encantada": 0.8, "divertido": 0.8,
        "gracioso": 0.7, "bien": 0.5, "perfecto": 0.7, "excelente": 0.8,
        "emocionado": 0.8, "emocionada": 0.8, "ilusion": 0.8,
        "orgulloso": 0.8, "orgullosa": 0.8, "gracias": 0.4,
        "jaja": 0.6, "jajaja": 0.7, "risa": 0.7, "reir": 0.7,
        "bonito": 0.6, "lindo": 0.6, "agradable": 0.6,
    },
    "enojo": {
        "enojado": 1.0, "enojada": 1.0, "enfadado": 1.0, "enfadada": 1.0,
        "molesto": 0.9, "molesta": 0.9, "furioso": 1.0, "furiosa": 1.0,
        "rabia": 1.0, "ira": 1.0, "odio": 1.0, "harto": 0.9, "harta": 0.9,
        "cansado": 0.5, "cansada": 0.5, "injusto": 0.8, "injusta": 0.8,
        # Sin "basta": es la palabra con la que se pide salir de la trivia
        # (frases/salir_trivia.txt), no una señal de enojo. Y "molesta"
        # estaba dos veces (0.9 y 0.8): queda solo la de arriba.
        "fastidio": 0.8, "indigna": 0.9, "frustrante": 0.8,
        "frustrado": 0.9, "frustrada": 0.9, "frustracion": 0.9,
        "estupido": 0.8, "idiota": 0.8,
    },
    "miedo": {
        "miedo": 1.0, "asustado": 1.0, "asustada": 1.0, "susto": 0.9,
        "terror": 1.0, "panico": 1.0, "nervioso": 0.8, "nerviosa": 0.8,
        "ansiedad": 0.9, "ansioso": 0.9, "ansiosa": 0.9,
        "preocupado": 0.8, "preocupada": 0.8, "angustia": 0.9,
        "inseguro": 0.7, "insegura": 0.7, "temor": 0.9,
    },
    "sorpresa": {
        "sorprendido": 1.0, "sorprendida": 1.0, "sorpresa": 0.9,
        "increible": 0.6, "inesperado": 0.9, "vaya": 0.5, "guau": 0.8,
        "wow": 0.8, "alucinante": 0.8,
    },
}

_PALABRA = re.compile(r"[a-zñ]+", re.IGNORECASE)

# "solo" cuenta como soledad unicamente detras de un verbo de estado o de
# "muy" ("me siento solo", "estoy muy solo"). En cualquier otro sitio es el
# adverbio ("solo quiero jugar", "tan solo una") y no dice nada del animo.
_SOLO_CONTEXTO = {"siento", "sentia", "senti", "sentirme", "estoy", "estaba",
                  "estuve", "quede", "quedo", "muy"}
_PESO_SOLO = 0.7


def _normalizar(texto):
    """Minusculas y sin tildes, para que 'está' y 'ESTA' casen igual.

    Las transcripciones del STT llegan con tildes inconsistentes, asi que
    comparar en crudo perderia la mitad de los emparejamientos.

    OJO: tambien convierte la ñ en n (NFD la separa en n + tilde
    combinante). Por eso los terminos pasan por esta misma funcion al
    cargar, ver abajo: antes "extraño" estaba escrito con ñ y no casaba
    nunca."""
    sin_tildes = unicodedata.normalize("NFD", (texto or "").lower())
    return "".join(c for c in sin_tildes if unicodedata.category(c) != "Mn")


# Los terminos se normalizan IGUAL que el texto: asi una clave escrita con
# tilde o ñ no queda inalcanzable sin que nadie lo note.
_TERMINOS = {
    emocion: {_normalizar(t): p for t, p in terminos.items()}
    for emocion, terminos in _TERMINOS.items()
}


def _esta_negado(palabras, indice):
    """True si hay una negacion en las _VENTANA_NEGACION palabras previas."""
    inicio = max(0, indice - _VENTANA_NEGACION)
    return any(p in _NEGACIONES for p in palabras[inicio:indice])


def analizar_por_palabras(texto):
    """Devuelve (emocion, confianza) de lo que el texto EXPRESA.

    (None, 0.0) cuando no hay evidencia suficiente -- texto sin carga
    emocional, o con señales contradictorias. Es un resultado valido y
    frecuente, no un fallo: la mayoria de los mensajes de una trivia
    ("Lima", "no se", "la siguiente") no dicen nada del animo."""
    palabras = _PALABRA.findall(_normalizar(texto))
    if not palabras:
        return None, 0.0

    evidencia = {}
    for i, palabra in enumerate(palabras):
        if palabra == "solo":
            if (i > 0 and palabras[i - 1] in _SOLO_CONTEXTO
                    and not _esta_negado(palabras, i)):
                evidencia["tristeza"] = evidencia.get("tristeza", 0.0) + _PESO_SOLO
            continue
        for emocion, terminos in _TERMINOS.items():
            peso = terminos.get(palabra)
            if peso is None:
                continue
            # Negada -> no cuenta. Y NO se apunta a la emocion contraria:
            # "no estoy triste" no dice que este feliz.
            if _esta_negado(palabras, i):
                continue
            evidencia[emocion] = evidencia.get(emocion, 0.0) + peso

    if not evidencia:
        return None, 0.0

    ordenadas = sorted(evidencia.items(), key=lambda x: x[1], reverse=True)
    emocion, peso = ordenadas[0]

    # Dos emociones casi empatadas: el texto es ambiguo y se calla.
    if len(ordenadas) > 1 and ordenadas[1][1] > peso * (1.0 - _MARGEN_AMBIGUEDAD):
        return None, 0.0

    # La confianza sube con la evidencia pero satura: un termino fuerte da
    # ~0.4, dos dan ~0.5, y de ahi no pasa de CONFIANZA_MAXIMA.
    confianza = CONFIANZA_MAXIMA * (1.0 - 1.0 / (1.0 + peso))
    return emocion, round(confianza, 3)


# ── Motor neuronal opcional: RoBERTuito ───────────────────────────────
#
# POR QUE HAY DOS MOTORES. Las palabras clave de arriba no cuestan nada y
# funcionan sin red, pero solo ven los terminos que estan en la tabla:
# "no me salio nada bien" no tiene ninguno y se va en blanco. RoBERTuito
# (pysentimiento/robertuito-emotion-analysis, entrenado con tuits en
# espanol) lee la frase entera. Es el equivalente en espanol del modelo
# que usa la tesis de Rocha, que solo existe en ingles.
#
# Se elige con ROMEO_TER_MOTOR=robertuito. El default sigue siendo
# "palabras" para no cambiar el comportamiento de nadie que no lo pida.
#
# MEDIDO EN LA PC (26 oct 2026), 18 ms por frase, los dos motores sobre las
# mismas entradas:
#
#     frase                          robertuito        palabras
#     "gane el partido de futbol"    felicidad 0.85    sin evidencia
#     "se murio mi perro"            tristeza  0.78    sin evidencia
#     "hoy me fue muy mal"           tristeza  0.61    tristeza 0.23
#     "no estoy triste"              tristeza  0.85    sin evidencia
#
# Gana en las tres primeras: lee frases que no tienen ningun termino de la
# tabla. PIERDE en la cuarta -- no procesa la negacion, ve "triste" y
# contesta tristeza. Las palabras clave si la manejan (_esta_negado).
# Por eso el default no cambia: son 435 MB y un fallo conocido a cambio de
# cubrir frases sin palabras clave. Conviene medirlo en la Pi antes de
# ponerlo por defecto.
#
# El modelo vive FUERA del repo, en ~/.romeo/modelos/ter/ (ter.onnx,
# ter.json y tokenizer.json), y se genera una vez en la PC con
# pruebas/ter/exportar_ter.py. Si falta, o si onnxruntime no esta, se
# avisa UNA vez y se sigue con las palabras clave: el robot nunca se queda
# sin TER por esto.
MOTOR = os.environ.get("ROMEO_TER_MOTOR", "palabras").strip().lower()
DIR_MODELO = os.environ.get("ROMEO_TER_DIR", os.path.expanduser("~/.romeo/modelos/ter"))

# Etiquetas del modelo -> claves de estado_afectivo.ANCLAJES.
_TRADUCCION = {"others": "neutral", "joy": "felicidad", "sadness": "tristeza",
               "anger": "enojo", "surprise": "sorpresa", "disgust": "asco",
               "fear": "miedo"}

# A diferencia de las palabras clave, aca la confianza SI puede llegar a 1:
# el modelo lee la frase entera, no terminos sueltos. Aun asi se deja un
# tope por la misma razon de siempre -- lo que el texto expresa no es lo
# que la persona siente -- y porque el STT puede haber transcrito mal.
CONFIANZA_MAXIMA_NEURONAL = float(os.environ.get("ROMEO_TER_CONFIANZA_MAX_NN", "0.85"))

# Por debajo de esto no hay emocion clara: el modelo reparte la
# probabilidad entre varias y ninguna manda.
_MINIMO_NEURONAL = float(os.environ.get("ROMEO_TER_MINIMO_NN", "0.40"))

_motor_nn = None          # la sesion, una vez cargada
_nn_fallido = False       # ya se intento y no se pudo: no reintentar


def _cargar_robertuito():
    """Carga la sesion ONNX. Devuelve None si no se puede (y avisa una vez)."""
    global _motor_nn, _nn_fallido
    if _motor_nn is not None or _nn_fallido:
        return _motor_nn
    try:
        import json as _json

        import numpy as _np
        import onnxruntime as _ort
        from tokenizers import Tokenizer

        with open(os.path.join(DIR_MODELO, "ter.json"), encoding="utf-8") as f:
            id2label = _json.load(f)["id2label"]
        etiquetas = [_TRADUCCION[id2label[str(i)]] for i in range(len(id2label))]
        tok = Tokenizer.from_file(os.path.join(DIR_MODELO, "tokenizer.json"))
        op = _ort.SessionOptions()
        op.intra_op_num_threads = int(os.environ.get("ROMEO_TER_HILOS", "2"))
        sesion = _ort.InferenceSession(os.path.join(DIR_MODELO, "ter.onnx"), op)
        _motor_nn = (sesion, tok, etiquetas, _np)
    except Exception as e:                                  # noqa: BLE001
        _nn_fallido = True
        print(f"[ter] RoBERTuito no disponible ({e}); sigo con palabras clave")
    return _motor_nn


def analizar_por_robertuito(texto):
    """(emocion, confianza) segun RoBERTuito, o None si no esta cargado.

    Devuelve (None, 0.0) -- como las palabras clave -- cuando ninguna
    emocion llega a _MINIMO_NEURONAL, y cuando gana "neutral": ninguna de
    las dos cosas dice nada del animo, y ROSE no se debe mover por eso."""
    motor = _cargar_robertuito()
    if motor is None:
        return None
    sesion, tok, etiquetas, np = motor

    enc = tok.encode(texto)
    ids = np.array([enc.ids], dtype=np.int64)
    masc = np.array([enc.attention_mask], dtype=np.int64)
    logits = sesion.run(None, {"input_ids": ids, "attention_mask": masc})[0][0]
    # softmax estable (se resta el maximo antes de exponenciar)
    e = np.exp(logits - logits.max())
    probs = e / e.sum()

    i = int(probs.argmax())
    emocion, p = etiquetas[i], float(probs[i])
    if emocion == "neutral" or p < _MINIMO_NEURONAL:
        return None, 0.0
    return emocion, round(min(p, CONFIANZA_MAXIMA_NEURONAL), 3)


def analizar(texto):
    """(emocion, confianza) de lo que el texto EXPRESA.

    Usa el motor de ROMEO_TER_MOTOR. Con "robertuito", si el modelo no esta
    disponible cae a las palabras clave en vez de quedarse sin TER."""
    if MOTOR in ("robertuito", "nn", "neuronal"):
        r = analizar_por_robertuito(texto)
        if r is not None:
            return r
    return analizar_por_palabras(texto)
