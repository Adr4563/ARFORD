# Modelos

Modelos que el código carga pero que no se generan al compilar. Van por
**Git LFS** porque varios pasan los 100 MB que GitHub acepta por archivo.

```bash
sudo apt install git-lfs && git lfs install   # una sola vez por máquina
git pull                                      # trae los modelos enteros
bash modelos/instalar_modelos.sh              # los copia a donde los busca Romeo
```

| Carpeta | Modelo | Origen | Se instala en |
|---|---|---|---|
| `ter/` | RoBERTuito (emoción en texto) | [pysentimiento/robertuito-emotion-analysis](https://huggingface.co/pysentimiento/robertuito-emotion-analysis), exportado con `pruebas/ter/exportar_ter.py` | `~/.romeo/modelos/ter/` |
| `ser/` | wav2vec2 MESD (emoción en voz) | [somosnlp-hackathon-2022/wav2vec2-base-finetuned-sentiment-mesd](https://huggingface.co/somosnlp-hackathon-2022/wav2vec2-base-finetuned-sentiment-mesd), exportado con `pruebas/stt-nemo/exportar_ser.py` | `~/.romeo/modelos/ser/` |
| `asr/` | whisper-tiny y Silero VAD | [sherpa-onnx asr-models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models) | `~/.romeo/modelos/` |
| `voz/` | Piper `es_MX-claude-high` | [sherpa-onnx tts-models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models) y [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices/tree/main/es/es_MX/claude/high) | `~/.romeo/modelos/` y `~/piper-voces/` |

El FER (`emotion-ferplus-8.onnx` y YuNet) no está aquí: ya viene en
`ros2_ws/src/romeo_drivers/romeo_drivers/data/modelos/`.

Los LLM afinados (`chat_model` en `romeo_bringup.launch.py`) tampoco: se
crean en Ollama con los `Modelfile` de `romeo_soup/`.

Para actualizar un modelo: reemplazar el archivo, regenerar las sumas con
`cd modelos && find ter ser asr voz -type f | sort | xargs sha256sum > SHA256SUMS`
y hacer commit.
