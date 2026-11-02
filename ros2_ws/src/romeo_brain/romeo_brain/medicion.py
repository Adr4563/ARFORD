# -*- coding: utf-8 -*-
"""Cronometro para medir ROMEO modulo a modulo.

APAGADO POR DEFECTO. Sin ROMEO_MEDIR=1 el decorador devuelve la funcion
TAL CUAL, sin envolverla: en produccion no cuesta ni una llamada extra.
Esto no es un profiler -- cProfile mide cada funcion interna y su
sobrecarga falsearia justo lo que se quiere medir en un robot que corre
en tiempo real. Aqui se miden NUEVE puntos elegidos a mano.

COMO SE USA

    from romeo_brain.medicion import medir, turno, tiempos

    @medir("stt")
    def transcribir(self, samples, sr):
        ...

y al cerrar el turno:

    with turno():          # vacia lo anterior
        ...                # el turno entero
    fila = tiempos()       # {"t_stt": 0.83, "t_ser": 0.12, ...}

LOS TIEMPOS SON POR HILO. Cada turno del orquestador corre en su hilo,
y el servidor STT atiende peticiones en paralelo: un acumulador global
mezclaria turnos distintos. threading.local() los separa.

QUE NO MIDE. La latencia L cruza dos procesos -- t_fin nace en el VAD de
communication_node y t_inicio_audio en _reproducir(), con un servicio
ROS en medio -- asi que no sale de un decorador. Se mide pasando los dos
instantes al CSV y restandolos ahi.
"""
import functools
import os
import threading
import time
from contextlib import contextmanager

ACTIVO = os.environ.get("ROMEO_MEDIR", "") not in ("", "0", "no", "false")

# Un acumulador por hilo: ver el docstring.
_local = threading.local()


def activa():
    """True si la medicion esta encendida. Para que quien llama no tenga
    que leer la constante ni la variable de entorno."""
    return ACTIVO


def _acumulador():
    if not hasattr(_local, "tiempos"):
        _local.tiempos = {}
    return _local.tiempos


def medir(nombre):
    """Decorador: suma al acumulador del hilo lo que tarda la funcion.

    `nombre` es la columna del CSV sin el prefijo: "stt" -> "t_stt".

    Si la misma etiqueta se mide varias veces en un turno (el LLM en
    trivia: una para el veredicto y otra para el comentario), los tiempos
    se SUMAN. Es lo que se quiere para "cuanto tardo el LLM este turno"."""
    def envolver(fn):
        if not ACTIVO:
            return fn

        @functools.wraps(fn)
        def dentro(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                acc = _acumulador()
                acc[nombre] = acc.get(nombre, 0.0) + (time.perf_counter() - t0)
        return dentro
    return envolver


@contextmanager
def medir_bloque(nombre):
    """Lo mismo que @medir pero para un trozo de codigo suelto.

    Para lo que no es una funcion propia: la llamada a urlopen del TTS,
    o _aplicar_afecto() mas _comportamiento() juntos como "capa
    afectiva"."""
    if not ACTIVO:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        acc = _acumulador()
        acc[nombre] = acc.get(nombre, 0.0) + (time.perf_counter() - t0)


@contextmanager
def turno():
    """Delimita un turno: vacia el acumulador al entrar.

    Se vacia al ENTRAR y no al salir para que tiempos() siga sirviendo
    despues del bloque, que es donde se escribe la fila del CSV."""
    _local.tiempos = {}
    yield


def anotar(nombre, valor):
    """Mete un valor medido por fuera (tokens del LLM, duracion del
    audio) en la misma fila."""
    if ACTIVO:
        _acumulador()[nombre] = valor


def tiempos(prefijo="t_"):
    """Lo acumulado en este hilo, listo para el CSV.

    Los tiempos van a milisegundos redondeados a 1 decimal; lo que se
    metio con anotar() se devuelve tal cual y sin prefijo."""
    salida = {}
    for clave, valor in _acumulador().items():
        if isinstance(valor, float):
            salida[f"{prefijo}{clave}"] = round(valor * 1000, 1)
        else:
            salida[clave] = valor
    return salida
