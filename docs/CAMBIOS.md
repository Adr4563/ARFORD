# Cambios del 22 de octubre de 2026

## Modelos y red

- `romeo_bringup.launch.py`: `chat_model` y `trivia_corrector_model` pasan a
  `romeo-qwen-v2`, y la IP por defecto de la PC (`ROMEO_PC`) a `192.168.0.218`.
- `llama_client.py`: `CHAT_MODEL` por defecto pasa a `romeo-qwen-v2`.
- `agente_trivia.py`: `CORRECTOR_LLM` por defecto pasa a `romeo-qwen-v2`.

## STT

- `_stt_engine.py`: arreglado el respaldo local. Con `stt_servidor`
  configurado y el servidor caído, transcribir en local fallaba en cada frase
  (`'NoneType' object has no attribute 'create_stream'`) porque whisper nunca
  se cargaba. `cargar(local=True)` ahora lo carga.

## TTS

- `scripts/tts_server.py` (nuevo): servidor de voz para la PC en el puerto
  8083. Responde `GET /hablar?texto=...` con un WAV, que es lo que
  `_voice_backends.py` ya pedía con `tts_servidor`. Misma voz y ajustes que la
  Pi (`es_MX-claude-high`).

## Respuestas del LLM

- `llama_client.py`: nueva `primera_frase()`, que deja la respuesta en una
  sola frase.
- `agente_chat.py`: responde una sola frase, y si la respuesta repite algo ya
  dicho en la sesión, la vuelve a pedir sin historial (`romeo-qwen-v2` copiaba
  su respuesta anterior palabra por palabra).
- `agente_trivia.py`: el comentario del veredicto se queda en una frase (el
  modelo copiaba el formato del prompt: "Pregunta: ... La persona dijo: ...").
