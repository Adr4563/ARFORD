# -*- coding: utf-8 -*-
"""traza_afectiva -- cada paso del calculo de valencia/arousal, por turno.

El log solo dice el resultado de cada etapa ("la voz suena a tristeza
(confianza 0.42)"). Para ver COMO se llego ahi, el orquestador arma con
esto un registro por turno con los valores intermedios -- probabilidades,
entropia, confianza, CAT -> VAD, fusion, empatia, decaimiento y AEB -- y lo
escribe como una linea JSON en ROMEO_TRAZA_AFECTO (por defecto
~/romeo_afecto.jsonl). La pagina /calculo del panel (pruebas/panel_robot)
lo muestra en vivo.

Todo se calcula con las MISMAS funciones de estado_afectivo y aeb que usa
el robot, asi que los numeros de la pagina son los del turno real.

Escribir nunca puede tumbar un turno: los errores se tragan.
"""
import json
import math
import os
import time

from . import aeb, estado_afectivo

RUTA = os.path.expanduser(os.environ.get("ROMEO_TRAZA_AFECTO", "~/romeo_afecto.jsonl"))


def _r(x, n=4):
    return round(float(x), n)


def _vad(v):
    return None if v is None else [_r(c) for c in v]


def entropia_normalizada(probabilidades):
    """H(p) / ln K, la que usa estado_afectivo.confianza (c = 1 - H)."""
    ps = [p for p in probabilidades if p > 0.0]
    if len(ps) < 2:
        return 0.0
    return -sum(p * math.log(p) for p in ps) / math.log(len(probabilidades))


def lectura(fuente, probabilidades, etiquetas, **extra):
    """Detalle de PULSE para una modalidad: de la distribucion cruda a
    (vector VAD, confianza), paso por paso. None si no hay distribucion."""
    probs = [float(p) for p in (probabilidades or [])]
    if not probs or len(probs) != len(etiquetas) or sum(probs) <= 0.0:
        return None
    agrupadas, pg = estado_afectivo.agrupar_en_cuatro(probs, etiquetas)
    total = sum(pg)
    pg = [p / total for p in pg]
    vector, c = estado_afectivo.percibir(pg, agrupadas)
    filas = []
    for etiqueta, p in zip(agrupadas, pg):
        ancla = estado_afectivo.ANCLAJES.get(etiqueta, (0.0, 0.0, 0.0))
        filas.append({"emocion": etiqueta, "p": _r(p), "ancla": _vad(ancla),
                      "aporte": _vad([p * x for x in ancla])})
    return {
        "fuente": fuente,
        "entrada": {e: _r(p) for e, p in zip(etiquetas, probs)},
        "filas": filas,
        "k": len(pg),
        "entropia": _r(entropia_normalizada(pg)),
        "confianza": _r(c),
        "vad": _vad(vector),
        "dominante": agrupadas[max(range(len(pg)), key=pg.__getitem__)],
        **extra,
    }


def comportamiento(estado):
    """Detalle del AEB: desviacion respecto a la base, direccion por
    coseno, zona neutra y nivel 1-4, igual que aeb.mapear()."""
    base = estado.animo_base
    dv = [estado.vector[i] - base[i] for i in range(3)]
    mag = math.sqrt(sum(x * x for x in dv))
    cosenos = {}
    if mag > 0.0:
        for cara, emocion in (("happy", "felicidad"), ("sad", "tristeza"), ("angry", "enojo")):
            ancla = estado_afectivo.ANCLAJES[emocion]
            na = math.sqrt(sum(x * x for x in ancla))
            cosenos[emocion] = _r(sum(dv[i] / mag * ancla[i] / na for i in range(3)))
    direccion, _ = estado.desviacion()
    emocion = {"happy": "felicidad", "sad": "tristeza", "angry": "enojo"}.get(direccion)
    radio = estado_afectivo.radio_neutral(emocion, base) if emocion else 0.0
    recorrido = (math.dist(estado_afectivo.ANCLAJES[emocion], base) - radio) if emocion else 0.0
    comp = aeb.mapear(estado)
    return {
        "base": _vad(base), "vector": _vad(estado.vector), "desviacion": _vad(dv),
        "magnitud": _r(mag), "cosenos": cosenos, "emocion": emocion,
        "fraccion_neutral": estado_afectivo._FRACCION_NEUTRAL,
        "radio": _r(radio), "recorrido": _r(recorrido),
        "nivel": comp["nivel"], "estado": comp["estado"], "cara": comp["cara"],
    }


def nueva(texto, estado, antes_decaer):
    """Arranca la traza de un turno, con el decaimiento que se acaba de
    aplicar (antes_decaer -> estado.vector)."""
    return {
        "t": time.time(), "texto": texto,
        "perfil": {"nombre": getattr(estado, "perfil", None), "empatia": estado.empatia,
                   "base": _vad(estado.animo_base), "decaimiento": estado.decaimiento,
                   "latencia": estado.latencia, "paso": estado.paso},
        "decaer": {"antes": _vad(antes_decaer), "despues": _vad(estado.vector)},
        "lecturas": [], "fusion": None, "empatia": None, "aeb": None,
    }


def reiniciar():
    """Vacia el archivo al arrancar: cada corrida empieza limpia."""
    try:
        with open(RUTA, "w", encoding="utf-8"):
            pass
    except OSError:
        pass


def escribir(traza):
    try:
        with open(RUTA, "a", encoding="utf-8") as f:
            f.write(json.dumps(traza, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 - la traza nunca puede tumbar un turno
        pass
