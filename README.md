# ARFORD — Socially Friendly Robot

**ARFORD** es un robot social amigable (*socially friendly robot*) y socialmente asistivo (SAR, *Socially Assistive Robot*), orientado a la **interacción humano-robot (HRI)** con **niños y jóvenes con TDAH** (Trastorno por Déficit de Atención e Hiperactividad) **y TEA** (Trastorno del Espectro Autista), entre otras necesidades de apoyo.

Su objetivo es ofrecer una interacción predecible, breve y motivadora: conversa (**Chat libre**) y juega **Trivia** por voz o texto, con una carita animada que expresa emociones, música, movimiento sobre un carrito mecanum y un juego de reconocimiento e imitación de emociones por cámara que ayuda a practicar la identificación de expresiones faciales.

Este repositorio contiene la versión **ROS 2 Jazzy** del robot: 4 paquetes que reemplazan al proceso Python monolítico original ([`Arquitecture-Agentic-RAG/deploy-raspberry-standalone/`](https://github.com/Adr4563/Arquitecture-Agentic-RAG)). La documentación técnica del workspace (nodos, tópicos, configuración, compilación y ejecución) está en **[`ros2_ws/README.md`](ros2_ws/README.md)**.

> **Estado:** el código está escrito, la sintaxis de cada `.py` pasa `python -m py_compile` y cada paquete tiene la estructura que exige `colcon`. **Todavía no se ha corrido `colcon build` ni `ros2 launch` en un ROS 2 Jazzy real, ni se ha probado contra el hardware.** Ver la [verificación](ros2_ws/README.md#verificación) antes de darlo por funcional.

> **Sobre el nombre:** el robot se llama **ARFORD**. Los identificadores internos conservan el prefijo `lora` por compatibilidad y **no deben cambiarse**: los 4 paquetes ROS 2 (`lora_brain`, `lora_drivers`, `lora_interfaces`, `lora_bringup`), los modelos de Ollama (`lora-chat-libre-v4`, `lora-trivia`, `lora-salida-trivia-v2`), el script `lora.sh` y el nombre USB de la placa de audio (`Lora Audio Board`, que determina `plughw:CARD=Lora`). Renombrarlos obliga a reconstruir el workspace, reimportar los modelos y reflashear la placa. Aparte, **LoRA** (en *fine-tunes LoRA*) es la técnica de fine-tuning, sin relación con el nombre del robot.

---

## Contenido

- [Qué hace ARFORD](#qué-hace-lora)
- [Hardware](#hardware)
- [Modelos de lenguaje (Ollama)](#modelos-de-lenguaje-ollama)
- [Voz](#voz)
- [Datos](#datos)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Inicio rápido](#inicio-rápido)

---

## Qué hace ARFORD

| Modo | Qué ocurre |
|---|---|
| **Chat libre** | Conversa de forma abierta y recuerda fragmentos de conversaciones anteriores cuando son relevantes. |
| **Trivia** | Ofrece 5 temas al azar entre 51 (248 preguntas en total) y hace una tanda de 5 preguntas. Reacciona a cada respuesta con voz, cara, música y movimiento. |
| **Juego de emociones** | Pide al usuario que imite una expresión ("¡Hazme una cara de feliz!"), la reconoce con la cámara y le dice si acertó. |

En cada reacción, ARFORD sigue siempre el mismo **orden**: primero habla, y después cambia de cara, suena la música y se mueve. Esa secuencia fija hace que la interacción sea predecible.

El usuario puede comunicarse **por voz o por texto**, desde el teclado de la Pi o desde una página web en el teléfono. Si falta algún componente (cámara, carrito, pantalla), el robot sigue conversando con lo que tenga disponible.

---

## Hardware

| Componente | Detalle | Conexión |
|---|---|---|
| **Computador principal** | Raspberry Pi 4, 8 GB de RAM, sin GPU (ARM Cortex-A72) | — |
| **Motores** | Carrito mecanum de 4 ruedas, driver L298N, controlado por un **ESP32-S3 DevKitC-1** | Serial/USB a 115200 baudios (`/dev/ttyACM0` por defecto) |
| **Cámara** | Raspberry Pi AI Camera (Sony IMX500) | `picamera2` |
| **Pantalla** | LCD/HDMI, sin sesión gráfica (headless) | `mpv --vo=drm` |
| **Micrófono y voz remotos** | Navegador del teléfono (Web Speech API) | Página web servida por la Pi en el puerto 8081 |

### Protocolo de motores

El firmware del ESP32-S3 (`carrito-mecanum-esp32/2-l298n-mecanum/mecanum_car_esp32s3.ino`, en el repositorio original) **no se modificó**. Recibe un comando de texto por línea:

| Comando | Movimiento |
|---|---|
| `F` / `B` | Adelante / atrás |
| `SL` / `SR` | Desplazamiento lateral (strafe) sin rotar el chasis |
| `RL` / `RR` | Rotar sobre su propio eje |
| `S` | Detenerse (también cualquier texto no reconocido) |
| `ROTATE360` | No existe en el firmware: `moving_control_node` lo convierte en 6 pulsos de `RR` con 0.4 s de pausa |

El firmware tiene un **watchdog**: si no llega otro comando en 500 ms, detiene los motores solo. Así se protege ante un cable desconectado o un proceso colgado en la Pi.

### Cámara y emociones

El NPU del IMX500 **no se usa** para las emociones, porque no existe un modelo de expresión facial publicado para su formato `.rpk`. La detección corre en la **CPU** con `onnxruntime` y dos modelos ONNX preentrenados:

- `face_detection_yunet_2023mar.onnx` (~230 KB): detector de caras YuNet.
- `emotion-ferplus-8.onnx` (~35 MB): clasificador FER+. De sus 8 clases se usan 4: `Feliz`, `Triste`, `Enojado` y `Neutral`.

Se hacen hasta 15 intentos de captura y detección, con 0.2 s entre intentos. Es obligatorio llamar a `picam2.stop()` **y** a `picam2.close()`: sin el `close()`, el siguiente `Picamera2()` del mismo proceso no puede adquirir la cámara.

### Pantalla

Un único proceso `mpv --idle` queda vivo toda la sesión, y cambiar de cara es un comando `loadfile` enviado por su socket IPC. Las caras son `.mp4` H.264 en lugar de `.gif` porque mpv las decodifica por hardware (`v4l2m2m`): usan ~10-23 % de CPU, frente a ~100-137 % con GIF.

Caras disponibles: `happy`, `sad`, `angry`, `content`, `speaking` y `countdown`.

El backend se elige automáticamente: **Linux sin `$DISPLAY` → DRM (mpv)**. En cualquier otro caso (Windows, o Linux con escritorio) se usa **Tkinter** (`face_viewer.py`), pensado para demos en PC.

---

## Modelos de lenguaje (Ollama)

Ollama corre en local (`http://localhost:11434`) y se usa su API compatible con OpenAI (`/v1/chat/completions`). Los tres modelos son fine-tunes **LoRA** de `qwen2.5:0.5b` (~397 MB cada uno) y se importan con `ollama create`, no con `ollama pull`.

| Variable de entorno | Valor por defecto | Rol |
|---|---|---|
| `CHAT_MODEL` | `lora-chat-libre-v4` | Genera las respuestas del Chat libre |
| `TRIVIA_MODEL` | `lora-trivia` | Reacciones del juego de emociones |
| `SALIDA_TRIVIA_MODEL` | `lora-salida-trivia-v2` | Decide si el mensaje es una RESPUESTA o una petición de SALIR a mitad de una pregunta de Trivia |

- Se eligió `qwen2.5:0.5b` en lugar de `llama3.2:3b` tras comparar 7 modelos: 484 MB de RAM y 1-3 s por respuesta, frente a 2.5 GB y 7-12 s, con la misma calidad medida.
- Conviene usar `OLLAMA_KEEP_ALIVE=-1` para que los modelos no se descarguen de memoria: recargarlos desde disco costaba entre 15 y 28 s en el peor caso.
- **El router y el corrector no usan LLM**, por velocidad y precisión:
  - `agent_router.py`: TF-IDF de n-gramas de caracteres + regresión logística (`scikit-learn`, `router_modelo.joblib`). Acierta ~95-96 % en validación y responde en microsegundos.
  - `agent_corrector.py`: comparación numérica si la respuesta esperada es un número; si es texto, normalización, coincidencia de palabras clave y comparación aproximada (`difflib`). Acierta el 98.4 % sobre 187 preguntas reales y nunca da por buena una respuesta incorrecta.
- Si Ollama no responde, `clasificar_salida_trivia()` devuelve `None` y el orquestador recurre a listas de palabras clave.

---

## Voz

### Salida (lo que dice ARFORD)

Hay tres motores de voz, que se eligen con el parámetro `voz_motor`:

| Motor | Cómo funciona | Cuándo usarlo |
|---|---|---|
| `telefono` (**por defecto**) | Sintetiza el **navegador del teléfono** (`speechSynthesis`), que recibe el texto por SSE | No consume CPU en la Pi. Requiere tener la página abierta y haber tocado "Activar voz" |
| `piper` | TTS neuronal 100 % local (ONNX), voz `es_MX-claude-high` (~60 MB) | Sin internet. Usa un solo hilo para dejar CPU a Ollama (sintetiza igual de rápido con 1 hilo que con 4) |
| `edge` | `edge-tts` (Microsoft, en la nube), voz `es-AR-ElenaNeural` | **Respaldo** automático si nadie conecta el teléfono en 20 s o si Piper no carga |

ARFORD espera a terminar de hablar antes de continuar con el turno.

### Entrada (lo que dice el usuario)

No hay reconocimiento de voz local en la Pi. Se probó `faster-whisper`, pero se cuelga sin dar error si se construye fuera del hilo principal, y ese problema no se resolvió.

La transcripción la hace el **navegador del teléfono** (`SpeechRecognition`). Funciona bien en Chrome para Android; Safari en iOS no lo implementa. Además exige un contexto seguro (HTTPS o localhost), y la Pi sirve HTTP plano en una IP de la red local, así que hay que activar una vez en cada teléfono el flag de Chrome que trata ese origen como seguro.

---

## Datos

- **Preguntas de Trivia:** `lora_brain/data/preguntas.jsonl`, con 248 preguntas en 51 temas, cargadas en memoria al iniciar. Cada fila tiene esta forma:
  ```json
  {"id": 1, "pregunta": "...", "cara": "Neutral",
   "respuesta_esperada": "...", "tema": "Arte, música y cultura - Nivel 1",
   "cara_respuesta_buena": "Neutral", "cara_respuesta_mala": "Neutral",
   "musical": "", "desplazamiento": "Adelante"}
  ```
  `desplazamiento` admite `Adelante`, `Atrás`, `Izquierda`, `Derecha` y `Girar 360°`. Hay 13 preguntas duplicadas por texto (con `id` distinto) que siguen sin resolverse.
- **Memoria episódica:** `memoria_episodica.py` busca con BM25 puro (`rank_bm25`, sin embeddings) sobre los últimos 300 turnos del Chat libre. Solo usa un recuerdo si comparte **al menos 2 palabras de contenido** con el mensaje actual. La memoria es compartida entre todos los usuarios.
- **Registro del Chat libre:** `registro_chat.py` guarda cada turno en JSONL para revisarlo después. Nunca se entrena directamente con este archivo, para que el modelo no refuerce sus propios errores.
- **Personalidad:** `personalidad.py` arma un system prompt con un perfil Big Five (OCEAN) fijo. Se omite si el modelo ya tiene la personalidad incorporada por fine-tuning.
- **Música:** `lora_drivers/data/musica/` contiene `angry-birds`, `danza-kuduro`, `minecraft`, `more-than-words-heaven`, `plantas-vs-zombies`, `super-mario-bros` y `zelda` (mp3).

---

## Estructura del repositorio

```
Socially-Friendly-Robot-Lora-Astro/
├── README.md              este archivo: qué es ARFORD, hardware, IA y datos
├── ros2_ws/               workspace ROS 2 Jazzy (el software del robot)
│   ├── README.md          documentación técnica del workspace
│   └── src/
│       ├── lora_interfaces/   mensajes y servicios
│       ├── lora_drivers/      nodos de hardware: voz, cámara y pantalla, motores
│       ├── lora_brain/        el cerebro: máquina de estados de la conversación
│       └── lora_bringup/      launch y parámetros
└── firmware-audio-board/  firmware de la placa de audio ESP32-S3
    ├── README.md          documentación del firmware
    └── main/              el código (C, ESP-IDF)
```

La **placa de audio** es una Waveshare ESP32-S3-AUDIO-Board que aporta los micrófonos y el altavoz que a la Raspberry Pi le faltan. Su firmware la expone como **tarjeta de sonido USB**, de modo que la Pi la usa como un dispositivo de audio normal. Ver [`firmware-audio-board/README.md`](firmware-audio-board/README.md).

---

## Inicio rápido

Requisitos: Raspberry Pi OS (o Ubuntu 24.04 / WSL2) con **ROS 2 Jazzy**, `mpv` y **Ollama** con los 3 modelos importados.

```bash
cd ros2_ws
pip install -r src/lora_drivers/requirements.txt -r src/lora_brain/requirements.txt
colcon build --symlink-install
source install/setup.bash
ros2 launch lora_bringup lora_bringup.launch.py
```

Después, escribe en la terminal o abre `http://<ip-de-la-pi>:8081/` en el teléfono. Los pasos completos, las opciones de configuración y cómo comprobar que todo funciona están en [`ros2_ws/README.md`](ros2_ws/README.md#instalación-y-ejecución).
