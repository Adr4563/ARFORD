# -*- coding: utf-8 -*-
"""Pruebas de PULSE optimizado: confianza por margen top-1/top-2 con
fiabilidad de modalidad (paso 1) y fusion con consenso intermodal (paso 3).

Corre sin ROS: estado_afectivo es Python puro.

    python -m pytest ros2_ws/src/romeo_brain/test -q
    python ros2_ws/src/romeo_brain/test/test_pulse_optimized.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from romeo_brain import estado_afectivo as ea  # noqa: E402

ETIQ = ["enojo", "felicidad", "neutral", "tristeza"]


def _cerca(a, b, tol=1e-9):
    return abs(a - b) <= tol


# ── Paso 1: confianza ───────────────────────────────────────────────

def test_confianza_margen_por_peso():
    p = [0.05, 0.70, 0.15, 0.10]                       # M = 0.70 - 0.15
    assert _cerca(ea.confianza(p, "TER"), 1.00 * 0.55)
    assert _cerca(ea.confianza(p, "FER"), 0.85 * 0.55)
    assert _cerca(ea.confianza(p, "SER"), 0.70 * 0.55)
    assert _cerca(ea.confianza(p), 0.55)              # sin fuente: peso 1


def test_confianza_casos_borde():
    assert ea.confianza([]) == 0.0
    assert ea.confianza(None) == 0.0
    assert ea.confianza([0.25, 0.25, 0.25, 0.25], "TER") == 0.0   # empate
    assert _cerca(ea.confianza([1.0], "FER"), 0.85)               # una clase
    assert 0.0 <= ea.confianza([1.0, 0.0, 0.0, 0.0], "TER") <= 1.0


def test_percibir_categorias_usa_fuente():
    p = [0.05, 0.80, 0.10, 0.05]
    _, c_ter, dom = ea.percibir_categorias(p, ETIQ, "TER")
    _, c_ser, _ = ea.percibir_categorias(p, ETIQ, "SER")
    assert dom == "felicidad"
    assert _cerca(c_ser / c_ter, 0.70)


# ── Paso 3: fusion con consenso ─────────────────────────────────────

def _escenario_feliz_con_ser_triste():
    fer_p = [0.05, 0.80, 0.10, 0.05]                  # cara: feliz
    ter_p = [0.03, 0.85, 0.08, 0.04]                  # texto: feliz
    ser_p = [0.05, 0.05, 0.15, 0.75]                  # voz: triste (ruido)
    lecturas = []
    for fuente, p in (("FER", fer_p), ("TER", ter_p), ("SER", ser_p)):
        v, c, dom = ea.percibir_categorias(p, ETIQ, fuente)
        lecturas.append((fuente, v, c, dom))
    return lecturas


def test_ser_discordante_pierde_confianza_y_su_queda_positivo():
    lecturas = _escenario_feliz_con_ser_triste()
    assert [d for *_, d in lecturas] == ["felicidad", "felicidad", "tristeza"]

    pares = [(v, c) for _f, v, c, _d in lecturas]
    atenuadas = ea.confianzas_por_consenso(pares)
    originales = [c for _v, c in pares]
    alfa = [a / c for a, c in zip(atenuadas, originales)]

    # Todas se atenuan algo, pero SER (la discordante) es la que mas.
    assert alfa[2] < alfa[0] and alfa[2] < alfa[1]
    assert alfa[2] < 0.25
    # Su peso relativo en la fusion baja.
    rel_antes = originales[2] / sum(originales)
    rel_despues = atenuadas[2] / sum(atenuadas)
    assert rel_despues < rel_antes

    s_u = ea.fusionar([(f, v, c) for f, v, c, _d in lecturas])
    s_pre = ea._media_ponderada(pares)
    assert s_u is not None and s_u[0] > 0.0
    # El consenso empuja hacia la mayoria feliz: mas valencia que sin el.
    assert s_u[0] > s_pre[0]


def test_alpha_es_exp_menos_gamma_d():
    s1, s2, s3 = (0.5, 0.0, 0.0), (0.5, 0.0, 0.0), (-0.5, 0.0, 0.0)
    at = ea.confianzas_por_consenso([(s1, 1.0), (s2, 1.0), (s3, 1.0)], gamma=1.5)
    assert _cerca(at[2], math.exp(-1.5 * 1.0))        # d = 1 al consenso
    assert _cerca(at[0], math.exp(-1.5 * 0.5))        # consenso de s2, s3 = 0


def test_una_sola_fuente_no_se_atenua():
    s = ea.ANCLAJES["felicidad"]
    r = ea.fusionar([(s, 0.4)])
    assert all(_cerca(r[i], s[i]) for i in range(3))
    assert ea.confianzas_por_consenso([(s, 0.4)]) == [0.4]


def test_dos_fuentes_simetricas():
    a, b = ea.ANCLAJES["felicidad"], ea.ANCLAJES["tristeza"]
    s_u = ea.fusionar([(a, 0.5), (b, 0.5)])
    esperado = tuple((a[i] + b[i]) / 2 for i in range(3))
    assert all(_cerca(s_u[i], esperado[i]) for i in range(3))


def test_fusionar_sin_divisiones_por_cero():
    s = ea.ANCLAJES["felicidad"]
    assert ea.fusionar([]) is None
    assert ea.fusionar(None) is None
    assert ea.fusionar([(None, 0.9)]) is None
    assert ea.fusionar([(s, 0.0), (s, -1.0)]) is None
    assert ea.fusionar([(s, float("nan"))]) is None
    # Una fuente con c=0 no rompe a las demas.
    assert ea.fusionar([("FER", s, 0.0), ("TER", s, 0.5)]) == s
    # Distancias enormes: underflow de exp -> vuelve a la media simple.
    lejos = (1e6, 0.0, 0.0)
    r = ea.fusionar([(s, 1.0), (lejos, 1.0), ((-1e6, 0.0, 0.0), 1.0)])
    assert r is not None and all(math.isfinite(x) for x in r)


def test_formato_de_lecturas_mixto():
    s = ea.ANCLAJES["enojo"]
    assert ea.fusionar([("TER", s, 0.3)]) == ea.fusionar([(s, 0.3)])


# ── ROSE sigue funcionando con la salida de la fusion ──────────────

def test_rose_recibe_fusion_positiva():
    lecturas = _escenario_feliz_con_ser_triste()
    s_u = ea.fusionar([(f, v, c) for f, v, c, _d in lecturas])
    est = ea.EstadoAfectivo(perfil="OPTIMISTIC")
    est.empatizar(s_u, max(c for _f, _v, c, _d in lecturas))
    assert est.vector[0] > 0.0


if __name__ == "__main__":
    fallos = 0
    for nombre, f in sorted(globals().items()):
        if nombre.startswith("test_") and callable(f):
            try:
                f()
                print(f"ok    {nombre}")
            except AssertionError as e:
                fallos += 1
                print(f"FALLA {nombre}: {e}")
    sys.exit(1 if fallos else 0)
