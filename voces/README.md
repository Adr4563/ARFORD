# Voces de Lora

Voz de salida de Lora: cómo se eligió, qué ajustes usa y cómo instalarla en la
Raspberry Pi. Todo corre **local, en CPU y sin internet**.

## Voz elegida

**GLaDOS en español** (voz Piper/VITS `es_ES-glados-medium`), ejecutada con
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) y con estos ajustes, elegidos
de oído en la placa de audio del robot ("versión 40" de las pruebas):

| Ajuste | Valor | Qué hace |
|---|---|---|
| `voz_vits_ruido` (`noise_scale`) | `0.667` | Variación del timbre (valor de fábrica del modelo) |
| `voz_vits_ruido_w` (`noise_scale_w`) | `0.5` | Variación del ritmo. En las pruebas se usó `0.8`, pero con ese valor cada frase salía distinta (la misma frase duraba entre 1.94 y 2.23 s); con `0.5` queda estable (1.95–2.02 s) sin cambiar el timbre |
| `voz_vits_duracion` (`length_scale`) | `1.15` | Un poco más pausada que la original |
| `voz_tono_semitonos` | `1.5` | Tono subido 1.5 semitonos (`asetrate` + `atempo` con ffmpeg, sin cambiar la duración) |

Genera una frase en ~1 s en la Raspberry Pi 4.

## Muestras

Todas dicen *"Hola, un gusto conocerte."*, la frase fija que se usó para comparar voces.

| Archivo | Qué es |
|---|---|
| [`muestras/glados_original.wav`](muestras/glados_original.wav) | GLaDOS **original**, sin ningún ajuste |
| [`muestras/glados_version40.wav`](muestras/glados_version40.wav) | La **versión 40** tal como se eligió (ritmo `0.8`) |
| [`muestras/glados_version40_ritmo_fijo.wav`](muestras/glados_version40_ritmo_fijo.wav) | La versión 40 con el ritmo fijo (`0.5`): **la que usa Lora** |
| [`muestras/glados_sin_ruido_robotico.wav`](muestras/glados_sin_ruido_robotico.wav) | Alternativa **sin ruido robótico**: menos "aire" (`noise_scale 0.33`) y tono subido conservando el timbre natural (`rubberband`, `formant=preserved`) |

## Instalación en la Raspberry Pi

```bash
mkdir -p ~/.lora/modelos && cd ~/.lora/modelos
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-es_ES-glados-medium.tar.bz2
tar xjf vits-piper-es_ES-glados-medium.tar.bz2 && rm vits-piper-es_ES-glados-medium.tar.bz2
```

Requiere `sherpa-onnx` y `numpy` en el Python de ROS 2, y `ffmpeg`/`mpv` en el
sistema. La ruta del modelo y los ajustes se configuran en
`ros2_ws/src/lora_bringup/config/lora_params.yaml` (sección `communication_node`),
con `voz_motor: "vits"`.

## Cómo se integró

- `lora_drivers/_voice_backends.py`: motor de voz nuevo `vits` (sherpa-onnx). Sintetiza,
  normaliza el volumen, sube el tono con ffmpeg y reproduce con mpv por la placa.
- `lora_drivers/_audio_salida.py`: la placa Waveshare solo acepta 16 kHz mono; mpv
  remuestrea con un filtro de alta calidad (con el remuestreo lineal de ALSA la voz
  sonaba metálica).
- `lora_drivers/communication_node.py`: parámetros `voz_vits_*` y `voz_tono_semitonos`.
- `lora_bringup/launch/lora_bringup.launch.py`: `voz_motor` por defecto `vits`.

## Otros cambios de esta etapa

- **Micrófono de la placa Waveshare**: `communication_node` escucha con el STT local
  (NVIDIA FastConformer en español + VAD, con normalización de volumen) y publica en
  `/lora/user_input`. El micrófono se pausa mientras Lora habla o suena música, para que
  no se escuche a sí misma, y se reconecta solo si la placa se desconecta. Se descartan
  las alucinaciones del reconocedor y lo que se dice mientras Lora está ocupada.
- **Chat sin voseo**: `lora-chat-libre-v6`, reentrenado (LoRA, en la PC por CPU) con el
  dataset pasado a tuteo. Frases fijas también en tuteo.
- **Trivia**: entra solo si se pide ("no quiero jugar" ya no la activa), sale con "no quiero
  jugar" y lo avisa en voz alta, anuncia la tanda con frases variadas ("¡Prepárate!…") y, si
  no entiende el tema, lo vuelve a preguntar en lugar de elegir uno al azar.
- **Nombre**: si no se dijo "me llamo…", Lora lo confirma ("¿Te llamas Adrián?").
- **Música a 55 %**: con la música fuerte la placa pedía demasiada corriente y se desconectaba.
- **Caras a 800×480** (resolución nativa de la pantalla de 5"), regeneradas desde los GIF con
  más calidad. La pantalla va a 65 Hz (`video=HDMI-A-1:800x480@65` en `cmdline.txt`): a 60 Hz
  aparecían puntos rojos por ruido en la señal HDMI.
- **Se quitó la página web del teléfono** (`_web_bridge.py` y el motor de voz `telefono`).
- **Página para mover el carrito**: `moving_control_node` sirve una página en el puerto
  8080 (`http://carrito.local`) con las 9 direcciones, giros y STOP; va por el mismo cable
  USB que usa Lora. El nodo ya no se cae si el carrito no está conectado y acepta diagonales.
- **Orquestador**: toma solo el nombre del usuario ("me llamo Adrián" → "Adrián"),
  saluda sin "soy Lora", anuncia "¡Vamos a jugar Trivia!" al entrar y se quitó el tema
  "Arte, música y cultura - Nivel 7".

## Voces evaluadas

Se probaron en la placa, con la misma frase, solo voces de mujer, locales y sin PyTorch:

| Voz | Motor | País | Resultado |
|---|---|---|---|
| Dii | Piper (OpenVoiceOS) | Colombia | Rápida (~1 s) |
| Daniela | Piper | Argentina | Rápida |
| Sharvard (voz 1) | Piper | España | Rápida |
| Laura, Gevy | Piper (HirCoir) | México | Rápidas |
| Chande, Copihue (+ versiones *small*) | sanoTTS (Piper destilado) | Colombia, Chile | Rápidas |
| F1–F5 | Supertonic 2 / 3 | Español sin país | 1.8 s / 6 s |
| Karen Savage | Mimic3 (M-AILABS) | España | Rápida |
| MBROLA `es3` | eSpeak NG + MBROLA | España | Instantánea, muy sintética |
| **GLaDOS** | Piper (sherpa-onnx) | España | **Elegida** |
| Kokoro `ef_dora`, MMS Colombia/Chile | Kokoro / MMS | — | Descartadas: más de 30 s por frase |

Descartadas por requisito: voces de hombre, voces en la nube (Azure, edge-tts) y
modelos que necesitan PyTorch o GPU (F5-TTS, XTTS, Orpheus, SpeechT5).

## Licencia

El modelo GLaDOS es una voz de la comunidad Piper que imita al personaje del
videojuego *Portal* (ver [rhasspy/piper#187](https://github.com/rhasspy/piper/issues/187#issuecomment-1802216304)).
No tiene una licencia explícita: usar solo con fines educativos y no comerciales.
