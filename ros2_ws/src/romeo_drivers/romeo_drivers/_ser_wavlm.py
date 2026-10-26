# -*- coding: utf-8 -*-
"""SER con WavLM: 3loi/SER-Odyssey-Baseline-WavLM-Categorical.

Alternativa al modelo por defecto de _ser_engine.py. Se elige con
ROMEO_SER_MODELO=3loi/SER-Odyssey-Baseline-WavLM-Categorical y da la misma
tupla (etiquetas, probabilidades), asi que el resto del sistema no cambia.

POR QUE ESTE. El de serie (wav2vec2 + MESD) esta entrenado con frases
ACTUADAS en estudio, y con el microfono de la placa se iba a "Neutral"
casi siempre. Medido sobre los mismos cuatro audios:

    audio      este modelo      el de MESD
    triste     Sad   0.93       Neutral 0.87
    feliz      Happy 0.84       Neutral
    enojado    Fear  0.39       Neutral 0.87

Este sale del baseline del Odyssey SER Challenge, entrenado con
MSP-Podcast (habla espontanea, con ruido) y sobre WavLM, que va mejor que
wav2vec2 en tareas paralinguisticas. Ademas da las 8 emociones que ya
mapea _ser_engine._ETIQUETAS, incluidas sorpresa y desprecio.

LA ARQUITECTURA VA REIMPLEMENTADA AQUI a proposito. El repo la trae en un
pipeline_utils.py propio que obligaria a trust_remote_code=True, o sea a
ejecutar codigo bajado de internet dentro del robot. Son tres piezas
(WavLM + pooling atencional + un MLP), asi que se escriben y se cargan los
pesos del checkpoint, que es lo unico que se baja.
"""
import json
import os

import numpy as np

REPO = "3loi/SER-Odyssey-Baseline-WavLM-Categorical"
SAMPLE_RATE = 16000


def _construir(repo, hilos):
    """(ssl, pool, clf, etiquetas, media, desv). Importa torch aqui dentro
    para que el modulo se pueda importar sin tenerlo."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoModel

    torch.set_num_threads(hilos)

    class PoolingAtencional(nn.Module):
        """Media y desviacion de los cuadros, ponderadas por atencion."""

        def __init__(self, dim):
            super().__init__()
            self.sap_linear = nn.Linear(dim, dim)
            self.attention = nn.Parameter(torch.zeros(dim, 1))

        def forward(self, x):                   # (1, T, dim) -> (1, dim*2)
            h = torch.tanh(self.sap_linear(x))
            w = F.softmax(torch.matmul(h, self.attention).squeeze(2), dim=1).unsqueeze(2)
            mu = torch.sum(x * w, dim=1)
            rh = torch.sqrt((torch.sum((x ** 2) * w, dim=1) - mu ** 2).clamp(min=1e-5))
            return torch.cat((mu, rh), 1)

    class Clasificador(nn.Module):
        """Linear + LayerNorm + ReLU, y la capa de salida."""

        def __init__(self, entrada, oculto, salidas):
            super().__init__()
            self.fc = nn.ModuleList([nn.Sequential(
                nn.Linear(entrada, oculto), nn.LayerNorm(oculto),
                nn.ReLU(), nn.Dropout(0.0))])
            self.out = nn.Sequential(nn.Linear(oculto, salidas))
            self.inp_drop = nn.Dropout(0.0)

        def forward(self, x):
            h = self.inp_drop(x)
            for capa in self.fc:
                h = capa(h)
            return self.out(h)

    cfg = json.load(open(hf_hub_download(repo, "config.json"), encoding="utf-8"))
    etiquetas = [cfg["id2label"][str(i)] for i in range(cfg["num_classes"])]
    pesos = load_file(hf_hub_download(repo, "model.safetensors"))

    def trozo(prefijo):
        n = len(prefijo)
        return {k[n:]: v for k, v in pesos.items() if k.startswith(prefijo)}

    ssl = AutoModel.from_pretrained(cfg["ssl_type"])
    ssl.load_state_dict(trozo("ssl_model."), strict=False)
    ssl.eval()

    pool = PoolingAtencional(cfg["hidden_size"])
    pool.load_state_dict(trozo("pool_model."))
    pool.eval()

    clf = Clasificador(cfg["hidden_size"] * 2, cfg["hidden_size"], cfg["num_classes"])
    clf.load_state_dict(trozo("ser_model."))
    clf.eval()

    return ssl, pool, clf, etiquetas, cfg["mean"], cfg["std"]


class ClasificadorWavLM:
    """Misma interfaz que _ser_engine.ClasificadorVoz."""

    def __init__(self, modelo=None, hilos=None):
        self.modelo = modelo or REPO
        self._hilos = int(hilos or os.environ.get("ROMEO_SER_HILOS", "2"))
        self._piezas = None
        self.error = None

    @property
    def listo(self):
        return self._piezas is not None

    def cargar(self):
        """True si quedo listo. Nunca lanza: sin SER el STT sigue."""
        if self._piezas is not None:
            return True
        try:
            self._piezas = _construir(self.modelo, self._hilos)
            return True
        except Exception as e:                  # noqa: BLE001
            self.error = str(e)[:200]
            self._piezas = None
            return False

    def clasificar(self, samples, sample_rate=SAMPLE_RATE):
        """(etiquetas, probabilidades) de la frase, o None.

        Las etiquetas salen en ingles ("Angry", "Sad"...), que es lo que
        _ser_engine.agrupar() espera: ya las traduce a las claves de
        ANCLAJES."""
        if not self.cargar():
            return None
        import torch

        arr = np.asarray(samples, dtype=np.float32).reshape(-1)
        if arr.size < SAMPLE_RATE // 2:          # menos de medio segundo
            return None
        if sample_rate != SAMPLE_RATE:
            n = int(arr.size * SAMPLE_RATE / sample_rate)
            arr = np.interp(np.linspace(0, arr.size - 1, n),
                            np.arange(arr.size), arr).astype(np.float32)
        try:
            ssl, pool, clf, etiquetas, media, desv = self._piezas
            x = torch.from_numpy(((arr - media) / (desv + 1e-9)).astype(np.float32))
            with torch.no_grad():
                logits = clf(pool(ssl(x.unsqueeze(0)).last_hidden_state))[0].numpy()
            e = np.exp(logits - logits.max())
            p = e / e.sum()
        except Exception as ex:                  # noqa: BLE001
            print(f"[SER] falló al clasificar: {ex}")
            return None

        from ._ser_engine import agrupar
        return agrupar([{"label": n, "score": float(v)}
                        for n, v in zip(etiquetas, p)])
