# -*- coding: utf-8 -*-
"""SER -- reconocimiento de la emoción en la VOZ (Speech Emotion Recognition).

Corre en la PC, dentro de stt_server.py, sobre el MISMO audio de la frase
que se transcribe: la Pi no manda nada extra ni carga ningún modelo.

Es una fuente de la ventana de percepción afectiva (tesis de Rocha et al.,
sec. 3.4.1): entrega la emoción como CATEGORÍAS con probabilidades, igual
que FER y TER, y PULSE aplica la confianza por entropía (ec. 4.1-4.2), la
conversión CAT -> VAD (ec. 4.3) y la fusión (ec. 4.4).

EL MODELO ES INTERCAMBIABLE (variable ROMEO_SER_MODELO). Cualquier modelo de
`audio-classification` de HuggingFace sirve, siempre que sus etiquetas
estén en _ETIQUETAS (inglés o español, se traducen solas). Por defecto:

    somosnlp-hackathon-2022/wav2vec2-base-finetuned-sentiment-mesd
        wav2vec2-base afinado con MESD (Mexican Emotional Speech Database),
        o sea entrenado en ESPAÑOL. 6 emociones: Anger, Disgust, Fear,
        Happiness, Neutral, Sadness. 83 % de precisión en MESD.
        Limitación: MESD son frases cortas actuadas; con habla espontánea
        acierta menos, y por eso la confianza que calcula PULSE importa.

Variables de entorno:
    ROMEO_SER          0 lo desactiva (default 1)
    ROMEO_SER_MODELO   id del modelo en HuggingFace (default: el de arriba)

Necesita `pip install transformers torch` en la PC. Si falta o el modelo no
carga, el SER queda desactivado y el STT sigue funcionando igual.
"""
import os

import numpy as np

MODELO = os.environ.get(
    "ROMEO_SER_MODELO",
    "somosnlp-hackathon-2022/wav2vec2-base-finetuned-sentiment-mesd")
ACTIVO = os.environ.get("ROMEO_SER", "1") not in ("0", "false", "False")
SAMPLE_RATE = 16000

# Por debajo de esto no hay voz suficiente para juzgar el tono ("sí", "no").
DURACION_MINIMA = 0.5

# Etiqueta del modelo -> clave de ANCLAJES (romeo_brain/estado_afectivo.py).
# Cubre los nombres más comunes en inglés y español, para que cambiar de
# modelo no obligue a tocar código. Lo que no está aquí ("other",
# "unknown"...) se descarta y el resto se renormaliza.
# CUATRO EMOCIONES, NO OCHO. El robot trabaja con neutral, tristeza, enojo
# y felicidad, que son las unicas que la camara sabe distinguir
# (_emotion_detector.ETIQUETAS_RELEVANTES) y las unicas que tienen cara en
# estado_afectivo._CARAS. Las demas se AGRUPAN en la mas cercana en vez de
# descartarse, para no tirar la señal:
#
#     miedo     -> tristeza   negativa; descartarlo dejaba la activacion
#                             en -0.14 cuando el modelo veia miedo claro,
#                             o sea pasaba de "alerta" a "apagado"
#     asco      -> enojo      negativa y activada, como el enojo
#     sorpresa  -> felicidad  positiva y activada
#     desprecio -> enojo      negativa, dominante
#
# Medido sobre las distribuciones tipicas: en las cuatro que se conservan
# el VAD se mueve 0.04-0.09, por debajo del ruido del propio sistema.
_ETIQUETAS = {
    "anger": "enojo", "angry": "enojo", "ang": "enojo", "enojo": "enojo",
    "ira": "enojo",
    "disgust": "enojo", "disgusted": "enojo", "asco": "enojo",
    "contempt": "enojo", "desprecio": "enojo",
    "fear": "tristeza", "fearful": "tristeza", "miedo": "tristeza",
    "happiness": "felicidad", "happy": "felicidad", "hap": "felicidad",
    "joy": "felicidad", "alegria": "felicidad", "felicidad": "felicidad",
    "surprise": "felicidad", "surprised": "felicidad", "sorpresa": "felicidad",
    "neutral": "neutral", "neu": "neutral", "calm": "neutral",
    "sadness": "tristeza", "sad": "tristeza", "tristeza": "tristeza",
}


def traducir_etiqueta(etiqueta):
    """'Anger' -> 'enojo'. Acepta formatos como '生气/angry'. None si no
    es una emoción conocida."""
    e = str(etiqueta).strip().lower().split("/")[-1]
    return _ETIQUETAS.get(e)


def agrupar(salida):
    """De la salida del pipeline ([{label, score}, ...]) a (etiquetas,
    probabilidades) en claves de ANCLAJES, sumando las que caen en la misma
    clave y renormalizando. None si no queda ninguna emoción conocida."""
    acumulado = {}
    for item in salida or []:
        clave = traducir_etiqueta(item.get("label", ""))
        if clave is not None:
            acumulado[clave] = acumulado.get(clave, 0.0) + float(item.get("score", 0.0))
    total = sum(acumulado.values())
    if total <= 0.0:
        return None
    etiquetas = sorted(acumulado)
    return etiquetas, [acumulado[e] / total for e in etiquetas]


class ClasificadorVoz:
    """Envuelve el modelo. Se carga una vez, al arrancar el servidor."""

    def __init__(self, modelo=None):
        self.modelo = modelo or MODELO
        self._pipe = None
        self.error = None

    @property
    def listo(self):
        return self._pipe is not None

    def cargar(self):
        """Devuelve True si quedó listo. Nunca lanza."""
        if not ACTIVO:
            self.error = "desactivado (ROMEO_SER=0)"
            return False
        if self._pipe is not None:
            return True
        try:
            import torch
            from transformers import pipeline
            dispositivo = 0 if torch.cuda.is_available() else -1
            self._pipe = pipeline("audio-classification", model=self.modelo,
                                  device=dispositivo)
            return True
        except Exception as e:  # noqa: BLE001 - sin SER, el STT sigue
            self.error = str(e)[:200]
            self._pipe = None
            return False

    def clasificar(self, samples, sample_rate=SAMPLE_RATE):
        """(etiquetas, probabilidades) de la frase, o None si no hay modelo,
        la frase es muy corta o falla. Nunca lanza."""
        if self._pipe is None:
            return None
        arr = np.asarray(samples, dtype=np.float32).reshape(-1)
        if arr.size < DURACION_MINIMA * sample_rate:
            return None
        if sample_rate != SAMPLE_RATE:   # el modelo espera 16 kHz
            n = int(arr.size * SAMPLE_RATE / sample_rate)
            arr = np.interp(np.linspace(0, arr.size - 1, n),
                            np.arange(arr.size), arr).astype(np.float32)
        try:
            salida = self._pipe({"raw": arr, "sampling_rate": SAMPLE_RATE}, top_k=None)
        except Exception as e:  # noqa: BLE001
            print(f"[SER] falló al clasificar: {e}")
            return None
        return agrupar(salida)
