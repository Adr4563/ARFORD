"""Gramatica JSGF con las frases de comandos.txt, para el modo comandos.

Con gramatica PocketSphinx solo puede devolver una de esas frases (o nada),
en vez de buscar entre 20 000-285 000 palabras. Es el mismo caso de uso que
tendria el menu de Romeo.

Cada palabra TIENE que estar en el diccionario del modelo, o pocketsphinx no
arranca. Las que faltan se avisan y se quitan.
"""

import os

from modelos import MODELOS

AQUI = os.path.dirname(os.path.abspath(__file__))


def leer_comandos(ruta=os.path.join(AQUI, "comandos.txt")):
    with open(ruta, encoding="utf-8") as f:
        return [l.strip().lower() for l in f if l.strip() and not l.startswith("#")]


def _palabras_del_diccionario(ruta):
    with open(ruta, encoding="utf-8", errors="replace") as f:
        return {l.split(None, 1)[0] for l in f if l.strip()}


def crear_gramatica(modelo, destino=None):
    """Escribe el .gram y devuelve su ruta."""
    destino = destino or os.path.join(AQUI, f"comandos_{modelo}.gram")
    dicc = _palabras_del_diccionario(MODELOS[modelo]["dict"])
    frases = []
    for frase in leer_comandos():
        faltan = [p for p in frase.split() if p not in dicc]
        if faltan:
            print(f"[gramatica] {modelo}: {frase!r} se quita, no estan en el "
                  f"diccionario: {faltan}")
            continue
        frases.append(frase)
    with open(destino, "w", encoding="utf-8") as f:
        f.write("#JSGF V1.0;\ngrammar comandos;\n")
        f.write("public <comando> = " + " | ".join(frases) + ";\n")
    return destino
