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
    get_package_share_directory('romeo_brain'), 'data', 'preguntas.jsonl')

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


def _temas_de(p):
    """El tema de una pregunta, como lista de un solo elemento.

    El original partía la columna por "/" para admitir una pregunta en
    varios temas. Eso estaba ROTO y no servía a nadie: el único valor del
    dataset con una barra es el NOMBRE de un tema -- "Socialización /
    presentación - Nivel 1" -- y el split lo rompía en dos trozos
    ("Socialización", "presentación - Nivel 1") que no calzaban con ningún
    tema real, así que ese tema era inalcanzable: nunca salía en una tanda.
    No se puede arreglar exigiendo " / " con espacios, porque el nombre usa
    exactamente ese separador; y ninguna pregunta del dataset declara dos
    temas, así que el split entero sobra.

    Si algún día una pregunta tiene que servir a varios temas, el dataset
    necesita un separador que no aparezca en los nombres (o una lista en el
    JSON), no esta heurística.

    temas_disponibles() usa esta misma función a propósito: así el catálogo
    que se le ofrece al usuario nunca promete un tema que el filtro de acá
    no sepa encontrar."""
    tema = (p.get("tema") or "").strip()
    return [tema] if tema else []


def temas_disponibles():
    """Todos los temas con al menos una pregunta, ordenados.

    orchestrator_node lo usa para armar su catálogo en vez de tener las
    cadenas duplicadas a mano: esa lista ya había divergido del dataset
    (ofrecía 3 temas sin ninguna pregunta)."""
    return sorted({t for p in _preguntas_cache.values() for t in _temas_de(p)})


def preguntas_por_tema(tema, excluir=(), cantidad=5):
    tema = (tema or "").strip()
    if not tema:
        return []
    ya_usados = set(excluir)
    disponibles = [
        p for pid, p in _preguntas_cache.items()
        if pid not in ya_usados and tema in _temas_de(p)
    ]
    elegidas = random.sample(disponibles, min(cantidad, len(disponibles)))
    return [_formatear(p) for p in elegidas]


_cargar_preguntas()  # se arma una sola vez, al importar el módulo
