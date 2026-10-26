# -*- coding: utf-8 -*-
"""TER neuronal -- que emocion EXPRESA el texto, con RoBERTuito.

Corre DONDE CORRE EL STT (la PC), sobre el texto que acaba de transcribir:
igual que el SER, que ya va sobre el mismo audio. Asi la Pi no carga 418 MB
ni paga la inferencia, y el texto no vuelve a viajar por la red -- ya esta
en el servidor.

Modelo: pysentimiento/robertuito-emotion-analysis, un RoBERTa entrenado con
tuits en espanol. Es el equivalente en espanol del de la tesis de Rocha
(michellejieli/emotion_text_classifier, solo ingles). 7 clases, con
"others" haciendo de neutral.

Se exporta una vez con pruebas/ter/exportar_ter.py y vive fuera del repo,
en ~/.romeo/modelos/ter/ (ter.onnx, ter.json, tokenizer.json). Si no esta,
`listo` queda en False y el servidor simplemente no manda TER: el
orquestador cae a las palabras clave de romeo_brain/ter.py.

LO QUE NO SABE HACER. No procesa la negacion: "no estoy triste" lo lee como
tristeza (medido). Las palabras clave si la manejan. Por eso el campo que
manda este motor es una sugerencia, no un reemplazo -- ver como lo combina
orchestrator_node._percibir_texto().
"""
import json
import os

import numpy as np

DIR = os.environ.get("ROMEO_TER_DIR", os.path.expanduser("~/.romeo/modelos/ter"))
HILOS = int(os.environ.get("ROMEO_TER_HILOS", "2"))

# Etiquetas del modelo -> las CUATRO del robot (neutral, tristeza, enojo,
# felicidad). Mismo criterio que _ser_engine._ETIQUETAS: las que sobran se
# agrupan en la mas cercana en vez de descartarse, para no perder la señal.
# Varias claves pueden caer en la misma emocion; agrupar() las suma.
TRADUCCION = {"others": "neutral", "joy": "felicidad", "sadness": "tristeza",
              "anger": "enojo", "surprise": "felicidad", "disgust": "enojo",
              "fear": "tristeza"}


class ClasificadorTexto:
    """RoBERTuito en ONNX. Carga perezosa: el primer uso paga ~0,8 s."""

    def __init__(self, directorio=DIR, hilos=HILOS):
        self._dir = directorio
        self._hilos = hilos
        self._sesion = None
        self._tok = None
        self.etiquetas = []
        self.modelo = "robertuito-emotion-analysis"
        self.error = None
        self.listo = False

    def cargar(self):
        """True si quedo utilizable. No lanza: el servidor sigue sin TER."""
        if self.listo:
            return True
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            with open(os.path.join(self._dir, "ter.json"), encoding="utf-8") as f:
                id2label = json.load(f)["id2label"]
            self.etiquetas = [TRADUCCION[id2label[str(i)]] for i in range(len(id2label))]
            self._tok = Tokenizer.from_file(os.path.join(self._dir, "tokenizer.json"))
            op = ort.SessionOptions()
            op.intra_op_num_threads = self._hilos
            self._sesion = ort.InferenceSession(os.path.join(self._dir, "ter.onnx"), op)
            self.listo = True
        except Exception as e:                              # noqa: BLE001
            self.error = str(e)
        return self.listo

    def clasificar(self, texto):
        """(etiquetas, probabilidades) para PULSE, o None.

        Devuelve la distribucion COMPLETA, no la ganadora: estado_afectivo
        la necesita entera para la confianza por entropia (ec. 4.1-4.2) y
        para el paso a VAD (ec. 4.3). Es el mismo formato que manda el SER.
        """
        if not texto or not texto.strip():
            return None
        if not self.cargar():
            return None
        try:
            enc = self._tok.encode(texto)
            ids = np.array([enc.ids], dtype=np.int64)
            masc = np.array([enc.attention_mask], dtype=np.int64)
            logits = self._sesion.run(
                None, {"input_ids": ids, "attention_mask": masc})[0][0]
            # softmax estable: se resta el maximo antes de exponenciar
            e = np.exp(logits - logits.max())
            probs = e / e.sum()
            # Con cuatro emociones hay etiquetas repetidas (p.ej. fear y
            # sadness -> tristeza): se suman, como hace agrupar() con el SER.
            acum = {}
            for nombre, valor in zip(self.etiquetas, probs):
                acum[nombre] = acum.get(nombre, 0.0) + float(valor)
            claves = sorted(acum)
            return claves, [round(acum[k], 4) for k in claves]
        except Exception as e:                              # noqa: BLE001
            self.error = str(e)
            return None
