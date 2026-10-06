"""Acceso a las preguntas de Trivia -- portado tal cual de preguntas.py del
proyecto original: preguntas.jsonl cargado en memoria (dict simple, no
BM25 -- el BM25 viejo era para el RAG de Chat libre, que ya no existe desde
el proyecto original), sin HTTP, sin servidor aparte. Sigue siendo así en
ROS2: orchestrator_node importa este módulo directo, no hay ningún motivo
para exponerlo por tópico/servicio (es memoria local del mismo proceso,
igual que antes era memoria local del mismo proceso Python).

Único cambio: PREGUNTAS_FILE ahora sale de ament_index_python (asset
instalado del paquete) en vez de una ruta relativa al archivo."""

import json
import os
import random

from ament_index_python.packages import get_package_share_directory

PREGUNTAS_FILE = os.path.join(
    get_package_share_directory('lora_brain'), 'data', 'preguntas.jsonl')

_preguntas_cache = {}


def _cargar_preguntas():
    global _preguntas_cache
    with open(PREGUNTAS_FILE, encoding="utf-8") as f:
        preguntas = [json.loads(l) for l in f if l.strip()]
    if not preguntas:
        print("preguntas.jsonl está vacío, nada que cargar.")
        return
    _preguntas_cache = {p["id"]: p for p in preguntas}
    print(f"[preguntas.py] {len(preguntas)} preguntas cargadas en memoria.")


def _formatear(p):
    return {
        "id": p["id"], "pregunta": p["pregunta"],
        "respuesta_esperada": p.get("respuesta_esperada", ""), "cara": p.get("cara", "Neutral"),
        "cara_respuesta_buena": p.get("cara_respuesta_buena", ""),
        "cara_respuesta_mala": p.get("cara_respuesta_mala", ""),
        "musical": p.get("musical", ""),
        "desplazamiento": p.get("desplazamiento", ""),
    }


def pregunta_aleatoria(excluir=()):
    ya_usados = set(excluir)
    disponibles = [
        p for pid, p in _preguntas_cache.items()
        if pid not in ya_usados and p.get("respuesta_esperada", "").strip()
    ]
    if not disponibles:
        return None
    return _formatear(random.choice(disponibles))


def pregunta_por_tema(tema, excluir=()):
    tema = (tema or "").strip()
    if not tema:
        return None
    ya_usados = set(excluir)
    disponibles = [
        p for pid, p in _preguntas_cache.items()
        if pid not in ya_usados
        and tema in [t.strip() for t in p.get("tema", "").split("/")]
    ]
    if not disponibles:
        return None
    return _formatear(random.choice(disponibles))


def preguntas_por_tema(tema, excluir=(), cantidad=5):
    tema = (tema or "").strip()
    if not tema:
        return []
    ya_usados = set(excluir)
    disponibles = [
        p for pid, p in _preguntas_cache.items()
        if pid not in ya_usados
        and tema in [t.strip() for t in p.get("tema", "").split("/")]
    ]
    elegidas = random.sample(disponibles, min(cantidad, len(disponibles)))
    return [_formatear(p) for p in elegidas]


_cargar_preguntas()  # se arma una sola vez, al importar el módulo
