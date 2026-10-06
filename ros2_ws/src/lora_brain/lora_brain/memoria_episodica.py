# -*- coding: utf-8 -*-
"""Memoria episódica de Chat libre (BM25, sin embeddings) -- portado tal
cual de memoria_episodica.py del proyecto original, sin ningún cambio de
lógica. Sigue reusando registro_chat.REGISTRO (ver ese archivo), y el
mismo gate de relevancia (>=2 palabras de contenido en común, no "score >
0" -- ver el docstring original sobre por qué ese umbral viejo enganchaba
temas sin relación)."""
import json
import os
import re

from . import registro_chat

_MAXIMO_EPISODIOS = 300
_OVERLAP_MINIMO = 2
_MAX_CHARS_SNIPPET = 140

_STOPWORDS_ES = {
    "de", "la", "el", "los", "las", "un", "una", "unos", "unas", "y", "o",
    "a", "en", "que", "es", "por", "con", "no", "se", "su", "sus", "para",
    "como", "más", "mas", "pero", "le", "les", "ya", "este", "esta",
    "estos", "estas", "ese", "esa", "esos", "esas", "sí", "si", "porque",
    "entre", "cuando", "muy", "sin", "sobre", "también", "tambien", "me",
    "hasta", "hay", "donde", "quien", "quién", "desde", "todo", "toda",
    "todos", "todas", "nos", "durante", "uno", "ni", "contra", "otro",
    "otra", "otros", "otras", "ante", "ellos", "ellas", "e", "esto", "eso",
    "mi", "mis", "tu", "tus", "él", "ella", "nosotros", "nosotras",
    "vosotros", "vosotras", "ustedes", "usted", "yo", "tú", "qué", "que",
    "cuál", "cual", "cómo", "como", "cuándo", "cuando", "dónde", "donde",
    "quién", "quien", "algo", "nada", "algunos", "algunas", "mucho",
    "muchos", "mucha", "muchas", "poco", "pocos", "está", "esta", "están",
    "estar", "ser", "soy", "eres", "somos", "son", "fue", "fui", "era",
    "han", "he", "has", "hemos", "vos", "te", "tal",
}

_TOKEN_RE = re.compile(r"[a-záéíóúñü0-9]+", re.IGNORECASE)

_cache = {"mtime": None, "bm25": None, "episodios": None, "corpus_tok": None}


def _tokenizar(texto):
    crudo = _TOKEN_RE.findall(texto.lower())
    return [t for t in crudo if t not in _STOPWORDS_ES and len(t) > 2]


def _cargar_episodios():
    ruta = registro_chat.REGISTRO
    if not os.path.exists(ruta):
        return []
    episodios = []
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if not linea:
                    continue
                try:
                    fila = json.loads(linea)
                except ValueError:
                    continue
                usuario = (fila.get("usuario") or "").strip()
                respuesta = (fila.get("respuesta") or "").strip()
                if usuario and respuesta:
                    episodios.append({"usuario": usuario, "respuesta": respuesta})
    except OSError:
        return []
    return episodios[-_MAXIMO_EPISODIOS:]


def _obtener_indice():
    from rank_bm25 import BM25Okapi

    ruta = registro_chat.REGISTRO
    mtime = os.path.getmtime(ruta) if os.path.exists(ruta) else None
    if mtime is not None and mtime == _cache["mtime"] and _cache["bm25"] is not None:
        return _cache["bm25"], _cache["episodios"], _cache["corpus_tok"]

    episodios = _cargar_episodios()
    corpus_tok = [
        _tokenizar(f"{ep['usuario']} {ep['respuesta']}") for ep in episodios
    ]
    bm25 = BM25Okapi(corpus_tok) if corpus_tok else None

    _cache["mtime"] = mtime
    _cache["bm25"] = bm25
    _cache["episodios"] = episodios
    _cache["corpus_tok"] = corpus_tok
    return bm25, episodios, corpus_tok


def buscar_relevante(mensaje_usuario):
    query_tok = set(_tokenizar(mensaje_usuario))
    if not query_tok:
        return None

    bm25, episodios, corpus_tok = _obtener_indice()
    if bm25 is None or not episodios:
        return None

    scores = bm25.get_scores(list(query_tok))
    mejor_idx, mejor_overlap, mejor_score = None, 0, None
    for i, tokens in enumerate(corpus_tok):
        overlap = len(query_tok & set(tokens))
        if overlap > mejor_overlap or (overlap == mejor_overlap and overlap > 0 and scores[i] > mejor_score):
            mejor_idx, mejor_overlap, mejor_score = i, overlap, scores[i]

    if mejor_idx is None or mejor_overlap < _OVERLAP_MINIMO:
        return None

    respuesta = episodios[mejor_idx]["respuesta"]
    if len(respuesta) > _MAX_CHARS_SNIPPET:
        respuesta = respuesta[:_MAX_CHARS_SNIPPET].rsplit(" ", 1)[0] + "..."
    return f"(Ya hablamos de esto antes: {respuesta})"
