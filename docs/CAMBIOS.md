# Cambios del 22 de octubre de 2026

## Modelos y red

- La IP por defecto de la PC (`ROMEO_PC`) pasa a `192.168.0.218`.
- Modelos: se probó `romeo-qwen-v2` y se **volvió a los anteriores**:
  `llama3.2:1b` para el chat y `romeo` para el corrector de trivia
  (`romeo_bringup.launch.py`, `llama_client.py`, `agente_trivia.py`).

## STT

- `_stt_engine.py`: **sin respaldo local**. Con `stt_servidor` configurado
  el STT corre solo en la PC; si no contesta, esa frase se pierde y la Pi no
  carga whisper. (Antes intentaba transcribir en la Pi y fallaba con
  `'NoneType' object has no attribute 'create_stream'`.)

## TTS

- `scripts/tts_server.py` (nuevo): servidor de voz para la PC en el puerto
  8083. Responde `GET /hablar?texto=...` con un WAV, que es lo que
  `_voice_backends.py` ya pedía con `tts_servidor`. Misma voz y ajustes que la
  Pi (`es_MX-claude-high`).
- `_voice_backends.py`: **sin respaldo local**. Con `tts_servidor`
  configurado la Pi no carga ninguna voz; si la PC no contesta, esa frase no
  se dice (queda en el log).

Los modelos (LLM, STT y TTS) corren solo en la PC de la Universidad de Lima.
En la Pi solo quedan el VAD, la cámara (FER), el router y la capa afectiva.

## Respuestas del LLM

- `llama_client.py`: nueva `primera_frase()`, que deja la respuesta en una
  sola frase.
- `agente_chat.py`: responde una sola frase, y si la respuesta repite algo ya
  dicho en la sesión, la vuelve a pedir sin historial (`romeo-qwen-v2` copiaba
  su respuesta anterior palabra por palabra).
- `agente_trivia.py`: el comentario del veredicto se queda en una frase (el
  modelo copiaba el formato del prompt: "Pregunta: ... La persona dijo: ...").
