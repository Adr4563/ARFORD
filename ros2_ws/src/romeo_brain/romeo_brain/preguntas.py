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
import re

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


# ══════════════════════════════════════════════════════════════════════
# Categorías de Trivia -- lo que el robot le OFRECE al usuario.
#
# El dataset tiene 13 familias de tema repartidas en 49 "<familia> - Nivel
# N", y antes el robot recitaba 5 de esos 49 al azar: le pedía al niño que
# eligiera entre cosas como "Geografía - Nivel 7, Razonamiento matemático -
# Nivel 5, ...". Imposible de seguir en voz alta, y cambiaba en cada turno.
#
# Acá se agrupan en 5 categorías con nombre corto. El usuario elige una
# ("ciencia") y el robot saca las preguntas de CUALQUIER nivel de las
# familias que la componen, así que los niveles siguen usándose pero no se
# le mencionan a nadie.
#
# Quedan fuera a propósito tres familias: "Juego de emociones" y "Juego de
# imitación" (se juegan con la cámara, no con texto -- ver el juego de
# emociones en orchestrator_node) y "Dilema del coche autónomo" y
# "Socialización / presentación" (no tienen respuesta correcta, así que no
# puntúan en una tanda de trivia).
# ══════════════════════════════════════════════════════════════════════
CATEGORIAS = {
    "ciencia": ["Ciencia y naturaleza"],
    "historia": ["Historia", "Cultura general"],
    "música": ["Arte, música y cultura", "música", "musica", "Reconocimiento musical"],
    "matemática": ["Matemática: multiplicación", "Razonamiento matemático"],
    "geografía": ["Geografía"],
}

# Lo que el usuario puede escribir/decir para elegir cada categoría. El
# router ya le quita acentos y mayúsculas (ver resolver_categoria), así que
# acá van las variantes de PALABRA, no de tildes.
_SINONIMOS = {
    "ciencia": ("ciencia", "ciencias", "naturaleza", "animales", "biologia"),
    "historia": ("historia", "historias", "cultura", "cultura general", "general"),
    "música": ("musica", "musical", "canciones", "cancion", "arte", "canta"),
    "matemática": ("matematica", "matematicas", "mate", "numeros", "cuentas",
                    "multiplicacion", "multiplicar", "sumas", "razonamiento"),
    "geografía": ("geografia", "paises", "pais", "capitales", "mapa", "mundo"),
}


def _sin_tildes(s):
    import unicodedata
    s = unicodedata.normalize("NFKD", (s or "").lower())
    return "".join(c for c in s if not unicodedata.combining(c)).strip()


def categorias_disponibles():
    """Las categorías que tienen al menos una pregunta sin usar, en el orden
    en que se le dicen al usuario."""
    return [c for c in CATEGORIAS if preguntas_por_categoria(c, cantidad=1)]


def resolver_categoria(eleccion):
    """Qué categoría eligió el usuario, o None si no se reconoce.

    Compara sin tildes ni mayúsculas y por substring, para que "quiero
    ciencia", "CIENCIAS" o "de ciencia porfa" caigan todos en "ciencia"."""
    texto = _sin_tildes(eleccion)
    if not texto:
        return None
    for cat, palabras in _SINONIMOS.items():
        if any(p in texto for p in palabras):
            return cat
    return None


def preguntas_por_categoria(categoria, excluir=(), cantidad=5):
    """Preguntas al azar de una categoría, de cualquier nivel de sus
    familias. Solo devuelve las que TIENEN respuesta esperada: una tanda de
    trivia necesita poder dar un veredicto."""
    familias = CATEGORIAS.get(categoria)
    if not familias:
        return []
    ya = set(excluir)
    disponibles = [
        p for pid, p in _preguntas_cache.items()
        if pid not in ya
        and (p.get("respuesta_esperada") or "").strip()
        and any(_familia_de(t) in familias for t in _temas_de(p))
    ]
    elegidas = random.sample(disponibles, min(cantidad, len(disponibles)))
    return [_formatear(p) for p in elegidas]


def _familia_de(tema):
    """"Geografía - Nivel 7" -> "Geografía". El nivel se usa para elegir
    preguntas pero nunca se le menciona al usuario."""
    return re.sub(r"\s*-\s*Nivel\s*\d+\s*$", "", tema).strip()


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
