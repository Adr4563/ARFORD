# -*- coding: utf-8 -*-
"""Frases que dice Romeo, cargadas de data/frases/ en vez de estar en el
codigo.

Cada archivo .txt es una lista: una frase por linea, las vacias y las que
empiezan por # se ignoran. Para añadir una variante basta escribirla ahi y
reiniciar el nodo -- no hace falta tocar Python ni recompilar.

    presentaciones.txt       lo que dice al arrancar ("soy Romeo, puedo...")
    saludos_apertura.txt     como pide el nombre
    bienvenidas.txt          saludo con el nombre + que propone hacer
    despedidas.txt           al cerrar la sesion
    sin_cara.txt             cuando la camara no ve a nadie
    respuestas_trivia_*.txt  veredictos de acierto y error
    salir_trivia.txt         frases con las que el usuario pide dejar la trivia
    tema_personal.txt        frases que indican que quiere hablar de lo suyo
    pedidos_emocion.json     como pide cada cara del juego de emociones

Las que llevan {nombre} o {esperada} son plantillas: se rellenan al usarlas.

Si un archivo falta o esta vacio se usa el respaldo que hay aca abajo, para
que el robot siga hablando aunque se borre algo por error.
"""
import json
import os
import random

from ament_index_python.packages import get_package_share_directory

_DIR = os.path.join(
    get_package_share_directory('romeo_brain'), 'data', 'frases')

# Minimo imprescindible si no se puede leer el archivo.
_RESPALDO = {
    'presentaciones': ["Hola, soy Romeo, un robot."],
    'saludos_apertura': ["¿Cómo te llamas?"],
    'bienvenidas': ["Mucho gusto, {nombre}. ¿Jugamos una trivia o charlamos?"],
    'despedidas': ["Hasta luego, {nombre}."],
    'sin_cara': ["No te vi bien, ¿te acercas?"],
    'respuestas_trivia_acierto': ["¡Correcto!"],
    'respuestas_trivia_error': ["No, era {esperada}."],
    'salir_trivia': ["salir de la trivia", "basta"],
    'tema_personal': ["estoy triste"],
}


def _leer(nombre):
    ruta = os.path.join(_DIR, f"{nombre}.txt")
    try:
        with open(ruta, encoding="utf-8") as f:
            lineas = [l.strip() for l in f]
        frases = [l for l in lineas if l and not l.startswith("#")]
        if frases:
            return frases
        print(f"[frases] {nombre}.txt está vacío, se usa el respaldo")
    except OSError as e:
        print(f"[frases] no se pudo leer {nombre}.txt ({e}), se usa el respaldo")
    return _RESPALDO.get(nombre, [])


def _leer_json(nombre, respaldo):
    ruta = os.path.join(_DIR, f"{nombre}.json")
    try:
        with open(ruta, encoding="utf-8") as f:
            datos = json.load(f)
        if datos:
            return datos
    except (OSError, ValueError) as e:
        print(f"[frases] no se pudo leer {nombre}.json ({e}), se usa el respaldo")
    return respaldo


# Se leen una vez al importar: son unos pocos KB y evita tocar el disco en
# cada turno.
PRESENTACIONES = _leer("presentaciones")
SALUDOS_APERTURA = _leer("saludos_apertura")
BIENVENIDAS = _leer("bienvenidas")
DESPEDIDAS = _leer("despedidas")
SIN_CARA = _leer("sin_cara")
RESPUESTAS_TRIVIA_ACIERTO = _leer("respuestas_trivia_acierto")
RESPUESTAS_TRIVIA_ERROR = _leer("respuestas_trivia_error")
SALIR_TRIVIA = _leer("salir_trivia")
TEMA_PERSONAL = _leer("tema_personal")
PEDIDOS_EMOCION = _leer_json(
    "pedidos_emocion",
    {"Feliz": ["¡Hazme una cara de felicidad!"],
     "Triste": ["Ahora una cara triste."],
     "Enojado": ["Ahora una cara de enojado."],
     "Neutral": ["Y por último, una cara neutra."]})


def una(lista, **kwargs):
    """Devuelve una frase al azar de `lista`, con las plantillas rellenadas."""
    frase = random.choice(lista)
    return frase.format(**kwargs) if kwargs else frase


print(f"[frases.py] {sum(len(l) for l in (PRESENTACIONES, SALUDOS_APERTURA, BIENVENIDAS, DESPEDIDAS, SIN_CARA, RESPUESTAS_TRIVIA_ACIERTO, RESPUESTAS_TRIVIA_ERROR, SALIR_TRIVIA, TEMA_PERSONAL))} frases cargadas de data/frases/.")
