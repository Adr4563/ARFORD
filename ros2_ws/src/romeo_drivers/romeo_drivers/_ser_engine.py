# -*- coding: utf-8 -*-
"""SER -- reconocimiento de la emoción en la VOZ (Speech Emotion Recognition).

Corre en la PC, dentro de stt_server.py, sobre el MISMO audio de la frase
que se transcribe: la Pi no manda nada extra ni carga ningún modelo.

Es una fuente de la ventana de percepción afectiva (tesis de Rocha et al.,
sec. 3.4.1): entrega la emoción como CATEGORÍAS con probabilidades, igual
que FER y TER, y PULSE aplica la confianza por entropía (ec. 4.1-4.2), la
conversión CAT -> VAD (ec. 4.3) y la fusión (ec. 4.4).

MODELO: emotion2vec+ (emotion2vec/emotion2vec_plus_base), cargado con FunASR.
Se eligió tras comparar candidatos sobre los mismos audios (ver
pruebas/voz_ser/INFORME_SER.md): acertó 13/24 frente a 7/24 del HuBERT-Dusha
anterior, con menor latencia. Devuelve etiquetas en la forma "开心/happy"
(chino/inglés); traducir_etiqueta() toma la parte en inglés, que ya está en
_ETIQUETAS. Emociones del modelo: angry, happy, sad, neutral, disgusted,
fearful, surprised, other, unknown.

Variables de entorno:
    ROMEO_SER          0 lo desactiva (default 1)
    ROMEO_SER_MODELO   id del modelo emotion2vec (default: el de arriba)
    ROMEO_SER_HILOS    hilos de CPU para FunASR (default 2)

Necesita `pip install funasr torch soundfile` en la PC. Si falta o el modelo
no carga, el SER queda desactivado y el STT sigue funcionando igual.
"""
import os
import tempfile

import numpy as np

MODELO = os.environ.get("ROMEO_SER_MODELO", "emotion2vec/emotion2vec_plus_base")
ACTIVO = os.environ.get("ROMEO_SER", "1") not in ("0", "false", "False")
HILOS = int(os.environ.get("ROMEO_SER_HILOS", "2"))
SAMPLE_RATE = 16000

# Por debajo de esto no hay voz suficiente para juzgar el tono ("sí", "no").
DURACION_MINIMA = 0.5

# Etiqueta del modelo -> clave de ANCLAJES (romeo_brain/estado_afectivo.py).
# Cubre los nombres más comunes en inglés y español, para que cambiar de
# modelo no obligue a tocar código. Lo que no está aquí ("unknown"...) se
# descarta y el resto se renormaliza.
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
    "positive": "felicidad", "joy": "felicidad", "alegria": "felicidad", "felicidad": "felicidad",
    "surprise": "felicidad", "surprised": "felicidad", "sorpresa": "felicidad",
    "neutral": "neutral", "neu": "neutral", "calm": "neutral", "other": "neutral",
    "sadness": "tristeza", "sad": "tristeza", "tristeza": "tristeza",
}


def traducir_etiqueta(etiqueta):
    """'Anger' -> 'enojo'. Acepta formatos como '生气/angry'. None si no
    es una emoción conocida."""
    e = str(etiqueta).strip().lower().split("/")[-1]
    return _ETIQUETAS.get(e)


def agrupar(salida):
    """De la salida del modelo ([{label, score}, ...]) a (etiquetas,
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
    """Envuelve emotion2vec+ (FunASR). Se carga una vez, al arrancar el
    servidor. Misma interfaz que antes: .modelo, .listo, .error, .cargar(),
    .clasificar()."""

    def __init__(self, modelo=None, hilos=None):
        self.modelo = modelo or MODELO
        self.hilos = hilos if hilos is not None else HILOS
        self._m = None
        self.error = None

    @property
    def listo(self):
        return self._m is not None

    def cargar(self):
        """Devuelve True si quedó listo. Nunca lanza."""
        if not ACTIVO:
            self.error = "desactivado (ROMEO_SER=0)"
            return False
        if self._m is not None:
            return True
        try:
            import torch  # noqa: F401  (FunASR lo necesita; aviso claro si falta)
            from funasr import AutoModel
            dispositivo = "cuda" if _hay_cuda() else "cpu"
            kw = dict(disable_update=True, ncpu=self.hilos, device=dispositivo)
            # emotion2vec vive en HuggingFace ("emotion2vec/...") y, de respaldo,
            # en ModelScope ("iic/..."), por si HF no responde.
            try:
                self._m = AutoModel(model=self.modelo, hub="hf", **kw)
            except Exception:
                self._m = AutoModel(model=self.modelo.replace("emotion2vec/", "iic/"),
                                    hub="ms", **kw)
            return True
        except Exception as e:  # noqa: BLE001 - sin SER, el STT sigue
            self.error = str(e)[:200]
            self._m = None
            return False

    def clasificar(self, samples, sample_rate=SAMPLE_RATE):
        """(etiquetas, probabilidades) de la frase, o None si no hay modelo,
        la frase es muy corta o falla. Nunca lanza."""
        if self._m is None:
            return None
        arr = np.asarray(samples, dtype=np.float32).reshape(-1)
        if arr.size < DURACION_MINIMA * sample_rate:
            return None
        if sample_rate != SAMPLE_RATE:   # el modelo espera 16 kHz
            n = int(arr.size * SAMPLE_RATE / sample_rate)
            arr = np.interp(np.linspace(0, arr.size - 1, n),
                            np.arange(arr.size), arr).astype(np.float32)
        try:
            import soundfile as sf
            # FunASR lee desde archivo; se escribe un WAV temporal de 16 kHz.
            with tempfile.TemporaryDirectory() as d:
                ruta = os.path.join(d, "a.wav")
                sf.write(ruta, arr, SAMPLE_RATE)
                res = self._m.generate(ruta, granularity="utterance",
                                       extract_embedding=False)
        except Exception as e:  # noqa: BLE001
            print(f"[SER] falló al clasificar: {e}")
            return None
        if not res:
            return None
        r = res[0]
        return agrupar([{"label": l, "score": float(s)}
                        for l, s in zip(r.get("labels", []), r.get("scores", []))])


def _hay_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False
