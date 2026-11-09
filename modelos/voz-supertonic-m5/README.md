# Voz Supertonic M5 (infantil)

Voz opcional para Romeo. **No está activada:** Romeo sigue hablando con Piper
`es_MX-claude-high`, y `instalar_modelos.sh` no copia esta carpeta.

Se eligió escuchando varias voces locales en español. Es la voz adulta M5 de
Supertonic 3 con el tono subido; no es un modelo entrenado con voces de niños.

## Ajustes elegidos

| Parámetro | Valor |
|---|---|
| Modelo | Supertonic 3, cuantizado `int8` |
| Voz | M5 (`sid = 4`) |
| Idioma | `es` |
| Pasos (`num_steps`) | 12 |
| Tono | +5 semitonos (preset «Infantil») |
| Velocidad | 1.0 |
| Efecto de robot | Ninguno |

`voice.bin` guarda las 10 voces de Supertonic en un solo archivo, por eso viene
completo aunque solo se use M5.

## Cómo generar la voz

Necesita `sherpa-onnx` y `numpy`. El tono sube sintetizando más lento y
remuestreando después, así la duración no cambia:

```python
import wave

import numpy as np
import sherpa_onnx as so

D = "modelos/voz-supertonic-m5/"
SEMITONOS = 5

tts = so.OfflineTts(so.OfflineTtsConfig(model=so.OfflineTtsModelConfig(
    supertonic=so.OfflineTtsSupertonicModelConfig(
        duration_predictor=D + "duration_predictor.int8.onnx",
        text_encoder=D + "text_encoder.int8.onnx",
        vector_estimator=D + "vector_estimator.int8.onnx",
        vocoder=D + "vocoder.int8.onnx",
        tts_json=D + "tts.json",
        unicode_indexer=D + "unicode_indexer.bin",
        voice_style=D + "voice.bin"),
    num_threads=3)))

f = 2 ** (SEMITONOS / 12)
g = so.GenerationConfig()
g.sid, g.speed, g.num_steps, g.extra = 4, 1.0 / f, 12, {"lang": "es"}
a = tts.generate("Hola, soy Romeo. ¿Cómo te llamas?", g)

x = np.asarray(a.samples, dtype=np.float32)
n = int(x.size / f)
x = np.fft.irfft(np.fft.rfft(x)[:n // 2 + 1], n) * (n / x.size)
x *= 0.85 / max(1e-6, float(np.abs(x).max()))

with wave.open("voz_m5.wav", "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(a.sample_rate)
    w.writeframes((x * 32767).astype(np.int16).tobytes())
```

## Para activarla más adelante

Todavía no se puede activar con un parámetro. `stt_server.py` y `tts_server.py`
corren en la PC, y `tts_server.py` solo carga voces VITS, así que habría que
agregarle Supertonic y el cambio de tono de arriba.

## Origen y licencia

- Modelo: [`sherpa-onnx-supertonic-3-tts-int8-2026-05-11`](https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models),
  conversión de [Supertonic](https://huggingface.co/Supertone/supertonic-3).
- Licencia: los pesos usan **OpenRAIL-M**, y el código de ejemplo de Supertone
  usa MIT (`LICENSE`). Revisa las restricciones de uso de OpenRAIL-M antes de
  distribuirla.
