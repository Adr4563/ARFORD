# -*- coding: utf-8 -*-
"""Paso 0 de PULSE en el FER: media ponderada por la calidad de YuNet,
con media recortada de respaldo.

Corre sin ROS ni camara: se simula ament_index_python, que el modulo
importa al cargarse.

    python -m pytest ros2_ws/src/romeo_drivers/test -q
"""
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if "ament_index_python" not in sys.modules:
    try:
        import ament_index_python.packages  # noqa: F401
    except ImportError:
        pkg = types.ModuleType("ament_index_python")
        sub = types.ModuleType("ament_index_python.packages")
        sub.get_package_share_directory = lambda _n: "."
        pkg.packages = sub
        sys.modules["ament_index_python"] = pkg
        sys.modules["ament_index_python.packages"] = sub

from romeo_drivers._emotion_detector import (  # noqa: E402
    DetectorEmocion, _media_recortada)

resumir = DetectorEmocion.resumir


def test_media_ponderada_por_calidad():
    buena = [0.0, 1.0, 0.0, 0.0]          # cara frontal, feliz
    mala = [0.0, 0.0, 0.0, 1.0]           # cara de perfil, "triste"
    r = resumir([buena, mala], [0.9, 0.1])
    assert np.allclose(r, [0.0, 0.9, 0.0, 0.1])
    assert abs(sum(r) - 1.0) < 1e-12


def test_normaliza_tras_ponderar():
    r = resumir([[2.0, 2.0], [1.0, 3.0]], [1.0, 1.0])
    assert np.allclose(r, [3 / 8, 5 / 8])


def test_respaldo_si_calidades_suman_cero_o_faltan():
    cuadros = [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]]
    esperado = np.mean(cuadros, axis=0)   # N=3, 15%: no recorta nada
    for cal in ([0.0, 0.0, 0.0], None, [0.9, None, 0.8], [0.5]):
        assert np.allclose(resumir(cuadros, cal), esperado / esperado.sum())


def test_media_recortada_descarta_extremos():
    p = np.array([[0.5]] * 9 + [[100.0]])            # N=10 -> quita 1 por lado
    assert np.allclose(_media_recortada(p, 0.15), [0.5])


def test_casos_vacios():
    assert resumir([]) is None
    assert resumir([[0.0, 0.0]], [1.0]) is None


if __name__ == "__main__":
    for nombre, f in sorted(globals().items()):
        if nombre.startswith("test_") and callable(f):
            f()
            print(f"ok    {nombre}")
