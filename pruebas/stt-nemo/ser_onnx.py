"""SER (emocion de la voz) en la Pi con onnxruntime, sin torch ni transformers.

Es el MISMO modelo que usa stt_server.py en la PC
(somosnlp-hackathon-2022/wav2vec2-base-finetuned-sentiment-mesd, ver
romeo_drivers/_ser_engine.py), exportado a ONNX con exportar_ser.py. En la PC
la salida ONNX y la de torch difieren como mucho en 2e-5.

Por que ONNX: en la Pi no hay transformers por apt y no se quiere venv;
onnxruntime ya esta instalado (lo usa el detector de caras).

El modelo vive fuera del repo (378 MB), en ~/.romeo/modelos/ser/:
    ser.onnx   el modelo
    ser.json   etiquetas y si el extractor normaliza
"""

import json
import os

import numpy as np

DIR = os.environ.get("ROMEO_SER_DIR", os.path.expanduser("~/.romeo/modelos/ser"))
DURACION_MINIMA = 0.5   # igual que _ser_engine.py: "si"/"no" no se juzgan

# Mismas claves que _ser_engine._ETIQUETAS / estado_afectivo.ANCLAJES.
_TRADUCCION = {"Anger": "enojo", "Disgust": "asco", "Fear": "miedo",
               "Happiness": "felicidad", "Neutral": "neutral", "Sadness": "tristeza"}


class ClasificadorVozOnnx:

    def __init__(self, directorio=DIR, hilos=2):
        import onnxruntime as ort
        with open(os.path.join(directorio, "ser.json"), encoding="utf-8") as f:
            meta = json.load(f)
        self._etiquetas = [_TRADUCCION.get(meta["id2label"][str(i)], meta["id2label"][str(i)])
                           for i in range(len(meta["id2label"]))]
        self._normalizar = meta.get("do_normalize", True)
        opciones = ort.SessionOptions()
        opciones.intra_op_num_threads = hilos
        self._sesion = ort.InferenceSession(os.path.join(directorio, "ser.onnx"), opciones,
                                            providers=["CPUExecutionProvider"])

    def clasificar(self, samples, sample_rate=16000):
        """(etiquetas, probabilidades) como _ser_engine.ClasificadorVoz, o None
        si la frase es muy corta."""
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if x.size < DURACION_MINIMA * sample_rate:
            return None
        if self._normalizar:
            # Lo mismo que hace Wav2Vec2FeatureExtractor: media 0, varianza 1.
            # Tambien hace que el nivel bajo de la placa no importe.
            x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
        logits = self._sesion.run(None, {"input_values": x[None, :]})[0][0]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        return list(self._etiquetas), [float(v) for v in p]
