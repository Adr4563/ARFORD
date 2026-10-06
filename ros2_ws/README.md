# Workspace ROS 2 Jazzy de Lora

Documentación técnica de los 4 paquetes ROS 2 del robot Lora. Para saber qué es Lora, su hardware, los modelos de IA y los datos que usa, ver el [README principal](../README.md).

> **Estado:** escrito y verificado en una máquina **sin ROS 2 instalado**. La sintaxis de cada `.py` pasa `python -m py_compile` y cada paquete tiene la estructura que exige `colcon` (`package.xml`, `setup.py` o `CMakeLists.txt`, `resource/<paquete>`, `entry_points`). **Todavía no se ha corrido `colcon build` ni `ros2 launch`.** Antes de darlo por funcional hay que seguir la sección [Verificación](#verificación) en la Raspberry Pi (o en WSL2 con ROS 2 Jazzy).

---

## Contenido

- [Paquetes](#paquetes)
- [Nodos](#nodos)
- [Tópicos y servicios](#tópicos-y-servicios)
- [Interfaces (`lora_interfaces`)](#interfaces-lora_interfaces)
- [Configuración](#configuración)
- [Instalación y ejecución](#instalación-y-ejecución)
- [Verificación](#verificación)
- [Flujo de un turno](#flujo-de-un-turno)
- [Decisiones de diseño](#decisiones-de-diseño)
- [Migración desde el proyecto original](#migración-desde-el-proyecto-original)
- [Limitaciones conocidas](#limitaciones-conocidas)

---

## Paquetes

```
ros2_ws/src/
├── lora_interfaces/            mensajes y servicios (ament_cmake)
│   ├── msg/  FaceCommand, MotionCommand, UserInput
│   └── srv/  Speak, PlayMusic, DetectEmotion, IsVoiceClientConnected
├── lora_drivers/               3 LifecycleNode de hardware (ament_python)
│   └── lora_drivers/
│       ├── communication_node.py        voz, música y página web
│       ├── visualization_faces_node.py  cámara → emoción + cara animada
│       ├── moving_control_node.py       motores (Serial → ESP32-S3)
│       ├── _voice_backends.py, _web_bridge.py, _music_player.py
│       ├── _face_display.py, face_viewer.py, _emotion_detector.py
│       ├── _cart_serial.py
│       └── data/  faces/ (mp4 + gif), musica/ (mp3), modelos/ (onnx)
├── lora_brain/                 el "cerebro" (ament_python)
│   └── lora_brain/
│       ├── orchestrator_node.py         máquina de estados de la conversación
│       ├── _ros_bridge.py               publishers y clientes de servicio
│       ├── agent_router.py, agent_corrector.py, agent_behavior.py
│       ├── llama_client.py              HTTP directo a Ollama
│       ├── memoria_episodica.py, personalidad.py, preguntas.py
│       ├── registro_chat.py, perf_monitor.py
│       └── data/  preguntas.jsonl, router_modelo.joblib
└── lora_bringup/               launch + parámetros (ament_cmake)
    ├── launch/lora_bringup.launch.py
    └── config/lora_params.yaml
```

`lora_bringup` usa `ament_cmake` porque no tiene nodos Python propios, solo `launch/` y `config/`. Es el patrón habitual en los paquetes que solo lanzan el sistema, y no cambia la forma de invocarlo.

---

## Nodos

| Nodo | Paquete | Tipo | Responsabilidad |
|---|---|---|---|
| `orchestrator_node` | `lora_brain` | `Node` | Máquina de estados de la sesión (Trivia / Chat libre): enrutamiento, corrección, reacciones, llamadas a Ollama y memoria episódica |
| `communication_node` | `lora_drivers` | `LifecycleNode` | Comunicación verbal y no verbal: síntesis de voz, música, página web (Flask + SSE) y entrada de texto por teclado, web o voz |
| `visualization_faces_node` | `lora_drivers` | `LifecycleNode` | Muestra la cara animada y detecta la emoción del usuario por cámara |
| `moving_control_node` | `lora_drivers` | `LifecycleNode` | Traduce `MotionCommand` al protocolo Serial del ESP32-S3 |

Los tres drivers son `LifecycleNode`: `on_configure` prepara los objetos **sin tocar el hardware**, y solo `on_activate` abre el Serial, arranca Flask o muestra la primera cara. El launch los configura y los activa automáticamente al arrancar, sin necesidad de un lifecycle manager externo.

---

## Tópicos y servicios

| Nombre | Tipo | Origen → destino | Notas |
|---|---|---|---|
| `/lora/user_input` | tópico `UserInput` | `communication_node` → `orchestrator_node` | Une en un solo tópico el teclado, el texto web y la voz transcrita |
| `/lora/face_command` | tópico `FaceCommand` | `orchestrator_node` → `visualization_faces_node` | |
| `/lora/motion_command` | tópico `MotionCommand` | `orchestrator_node` → `moving_control_node` | |
| `/lora/speak` | servicio `Speak` | `orchestrator_node` → `communication_node` | **Bloquea** hasta terminar de hablar |
| `/lora/play_music` | servicio `PlayMusic` | `orchestrator_node` → `communication_node` | Bloquea solo si `esperar=true` |
| `/lora/detect_emotion` | servicio `DetectEmotion` | `orchestrator_node` → `visualization_faces_node` | **Bloquea**, ~8-9 s en el peor caso |
| `/lora/is_voice_client_connected` | servicio `IsVoiceClientConnected` | `orchestrator_node` → `communication_node` | Indica si hay un teléfono con la voz activada |

Las llamadas a Ollama **no** pasan por ROS 2: `llama_client.py` usa HTTP directo desde `lora_brain`, porque no es hardware.

---

## Interfaces (`lora_interfaces`)

### Mensajes

| Mensaje | Campos |
|---|---|
| `FaceCommand` | `string face_name`: `happy` \| `sad` \| `angry` \| `content` \| `speaking` \| `countdown` |
| `MotionCommand` | `string command`: `F` \| `B` \| `SL` \| `SR` \| `RL` \| `RR` \| `S` \| `ROTATE360` |
| `UserInput` | `string text`, `string source`: `stdin` \| `web` \| `voice_browser` |

Los valores de `MotionCommand` son los mismos comandos de texto que ya entiende el firmware del ESP32-S3 (ver el [protocolo de motores](../README.md#protocolo-de-motores)).

### Servicios

| Servicio | Petición | Respuesta |
|---|---|---|
| `Speak` | `string text` | `bool success` |
| `PlayMusic` | `string filename`, `bool esperar` | `bool reproducido` |
| `DetectEmotion` | — | `string emotion` (`Feliz` \| `Triste` \| `Enojado` \| `Neutral` \| `""`), `float32 confidence`, `bool detected` |
| `IsVoiceClientConnected` | — | `bool connected` |

`DetectEmotion` devuelve `detected=false` si no hay cámara, si faltan dependencias (`onnxruntime`, `opencv`, `picamera2`) o si no detectó ninguna cara. **Nunca** propaga una excepción al orquestador.

`PlayMusic` con `esperar=true` se usa en las preguntas musicales, donde la canción es el enunciado y el usuario no puede responder antes de escucharla. Con `esperar=false` devuelve en cuanto empieza la reproducción.

---

## Configuración

### Argumentos del launch

| Argumento | Por defecto | Efecto |
|---|---|---|
| `voz_motor` | `vits` | Motor de voz (`vits` \| `piper` \| `edge`) de `communication_node` y `orchestrator_node` |
| `carrito_port` | `/dev/ttyACM0` | Puerto Serial del ESP32-S3 |
| `chat_model` | `lora-chat-libre-v6` | Se pasa al orquestador como `CHAT_MODEL` |
| `trivia_model` | `lora-trivia` | Se pasa como `TRIVIA_MODEL` |
| `salida_trivia_model` | `lora-salida-trivia-v2` | Se pasa como `SALIDA_TRIVIA_MODEL` |
| `chat_server_host` | `http://localhost:11434` | Se pasa como `CHAT_SERVER_HOST` |

### Parámetros ROS 2 (`src/lora_bringup/config/lora_params.yaml`)

| Nodo | Parámetro | Por defecto |
|---|---|---|
| `communication_node` | `voz_motor` | `vits` (`vits` \| `piper` \| `edge`) |
| `communication_node` | `voz_vits_dir` | `/home/pi/.lora/modelos/vits-piper-es_ES-glados-medium` |
| `communication_node` | `voz_vits_ruido` / `voz_vits_ruido_w` / `voz_vits_duracion` | `0.667` / `0.5` / `1.15` (ver [`voces/`](../voces/)) |
| `communication_node` | `voz_tono_semitonos` | `1.5` |
| `communication_node` | `voz_edge` | `es-AR-ElenaNeural` |
| `communication_node` | `voz_piper_modelo` | `""` (usa `~/piper-voces/es_CO-dii.onnx`) |
| `communication_node` | `mic_dispositivo` / `audio_salida` | placa Waveshare (`plughw:CARD=Board,DEV=0`) |
| `communication_node` | `audio_muestreo` | `16000` (la placa solo acepta 16 kHz) |
| `communication_node` | `stt_modelo` | `fastconformer` (`fastconformer` \| `base` \| `tiny`) |
| `communication_node` | `musica_volumen` | `55` (la voz no se toca) |
| `moving_control_node` | `carrito_port` | `/dev/ttyACM0` |
| `moving_control_node` | `control_web_port` | `8080` (página para mover el carrito: `http://carrito.local`) |
| `orchestrator_node` | `voz_motor` | `vits` (**debe coincidir** con el de `communication_node`) |

### Variables de entorno de `lora_brain`

| Variable | Por defecto | Uso |
|---|---|---|
| `CHAT_SERVER_HOST`, `CHAT_MODEL`, `TRIVIA_MODEL`, `SALIDA_TRIVIA_MODEL` | ver [Modelos](../README.md#modelos-de-lenguaje-ollama) | Conexión con Ollama. El launch las fija a partir de sus argumentos |
| `LORA_LOGS_DIR` | `~/.lora/logs` | Carpeta de métricas de `perf_monitor.py` (`tiempos.csv`, `recursos.csv`) |
| `PERF_MUESTREO_SEG` | `5` | Intervalo de muestreo de CPU y RAM |
| `CHAT_LIBRE_REGISTRO` | `~/.lora/chat_libre_training/conversaciones.jsonl` | Archivo del registro del Chat libre |
| `CHAT_LIBRE_REGISTRAR` | `1` | `0` o `false` desactiva el registro |

---

## Instalación y ejecución

Requisitos: Raspberry Pi OS (o Ubuntu 24.04 / WSL2) con **ROS 2 Jazzy**, `mpv` y **Ollama**. Todos los comandos se ejecutan desde la carpeta `ros2_ws`.

```bash
# 1. Dependencias del sistema y de Python
sudo apt install mpv
pip install -r src/lora_drivers/requirements.txt
pip install -r src/lora_brain/requirements.txt
# picamera2 viene con Raspberry Pi OS; no se instala con pip.
# piper-tts es opcional (solo para voz_motor:=piper).

# 2. Compilar
colcon build --symlink-install
source install/setup.bash

# 3. Tener Ollama en marcha con los 3 modelos importados
#    (lora-chat-libre-v6, lora-trivia y lora-salida-trivia-v2, con `ollama create`)

# 4. Lanzar todo
ros2 launch lora_bringup lora_bringup.launch.py
```

Variantes:

```bash
ros2 launch lora_bringup lora_bringup.launch.py voz_motor:=edge
ros2 launch lora_bringup lora_bringup.launch.py carrito_port:=/dev/ttyUSB0
```

Para interactuar, escribe en la terminal donde corre `communication_node` o abre `http://<ip-de-la-pi>:8081/` en el teléfono.

---

## Verificación

```bash
ros2 node list
# communication_node, visualization_faces_node, moving_control_node, orchestrator_node

ros2 lifecycle get /communication_node
# debe decir "active" pocos segundos después de arrancar

ros2 topic echo /lora/user_input
# al escribir en la terminal o en la página web, el texto aparece aquí

ros2 service call /lora/is_voice_client_connected lora_interfaces/srv/IsVoiceClientConnected
```

**Prueba completa:** escribe tu nombre cuando Lora te salude, pide Trivia, elige un tema y contesta una pregunta con respuesta numérica (por ejemplo, una multiplicación). Comprueba que Lora dice el veredicto, cambia de cara y, si la pregunta lo tiene configurado, se mueve y reproduce música.

---

## Flujo de un turno

### Chat libre

1. Llega el texto por `/lora/user_input`.
2. `agent_router` lo clasifica como Chat libre (sin LLM).
3. `memoria_episodica` busca un recuerdo relevante.
4. `llama_client` genera la respuesta con `CHAT_MODEL`, sin system prompt: el comportamiento viene del fine-tuning.
5. `registro_chat` guarda el turno.
6. `FaceCommand` `speaking` → `Speak` (bloquea) → `FaceCommand` `content`.

### Trivia

1. Al entrar en Trivia se ofrecen 5 temas al azar y el usuario elige uno (coincidencia exacta o aproximada, sin LLM).
2. `preguntas.py` carga una tanda de 5 preguntas del tema.
3. Se hace la pregunta con la cara `speaking` y `Speak`. Si es musical, primero suena la canción completa (`PlayMusic` con `esperar=true`).
4. Con la respuesta del usuario, `agent_corrector` decide si es correcta, `agent_behavior` elige la cara y Lora dice una frase fija elegida al azar.
5. La reacción sigue un **orden estricto**: primero `Speak` (bloquea) y después `FaceCommand`, `PlayMusic` (`esperar=false`) y `MotionCommand`, en paralelo.
6. Se pasa a la siguiente pregunta o se cierra con el resumen de aciertos.

### Juego de emociones

1. Se elige al azar una de las emociones que pide la pregunta y se publica esa cara como referencia.
2. Lora lo pide por voz: "¡Hazme una cara de feliz!".
3. `DetectEmotion` captura y clasifica la cara (hasta 15 intentos).
4. El veredicto compara la emoción pedida con la detectada. Lora dice una frase fija ("¡Correcto!" o "¡Incorrecto!") y luego cambia la cara, suena la música y se mueven los motores.
5. La tanda completa corre de una vez, sin volver al bucle principal entre preguntas.

### Salir de Trivia a mitad de una pregunta

Si el mensaje es corto (4 palabras o menos) y no contiene ninguna palabra clave de salida, se toma como respuesta sin consultar al LLM. Si no, decide `SALIDA_TRIVIA_MODEL`, y si Ollama no responde se usan listas de palabras clave.

---

## Decisiones de diseño

- **`Speak` es un servicio síncrono, no una action.** Se renuncia a la cancelación y al streaming a cambio de simplicidad; en el original ya nada usaba el streaming.
- **Orden voz → cara → (música + motores en paralelo).** Es un requisito del diseño. Lo garantiza `orchestrator_node.py::_reaccionar_veredicto()`, que espera la respuesta de `Speak` antes de publicar lo demás.
- **Llamar a servicios desde un callback sin bloqueo mutuo (deadlock).** `client.call()` se queda bloqueado con un `SingleThreadedExecutor`; la solución es `call_async()` con una espera activa corta y un `MultiThreadedExecutor`. Está documentado en `lora_brain/_ros_bridge.py::_llamar_servicio_sync()` y es lo más fácil de romper al modificar este código.
- **Degradación elegante.** Sin cámara, carrito o `mpv`, el nodo correspondiente sigue vivo y responde `detected=false` o `success=false`, sin lanzar nunca una excepción no controlada. El robot sigue conversando.
- **"Salir" termina la sesión, no el proceso.** En el original, "salir" cerraba el script. Aquí Lora se despide y queda lista para el siguiente usuario, como corresponde a un robot desplegado (`orchestrator_node.py::_finalizar_sesion()`).
- **Mandan las mediciones de latencia.** El router y el corrector dejaron de usar el LLM por mediciones hechas en la Pi. Ninguna capa nueva (tópicos, servicios, DDS) debería añadir una latencia perceptible a la conversación.
- **STT: `whisper-tiny` con 2 hilos, por medición.** `_stt_engine.py` usa sherpa-onnx porque corre sobre `onnxruntime`, que ya está instalado para el detector de emociones, y porque no arrastra el fallo de hilos de `faster-whisper`. Medido en la Pi con un audio de 3.8 s (`scripts/bench_stt.py`): `tiny`/2 hilos da 2.22 s sin carga y **2.45 s con Ollama generando**, contra 4.41 s de `base`/2 hilos. Dos resultados contraintuitivos: **más hilos es más lento** (4 hilos sube a 3.45 s y dispara la varianza) y el mínimo no está en 1 hilo (3.29 s bajo carga). El precio de `tiny` es que transcribe "Hola Laura" donde `base` acierta "Hola Lora"; para Trivia lo absorbe el `agent_corrector`, pero si el nombre llega a importar hay que volver a `base`.
- **Recursos muy limitados.** La Pi 4 no tiene GPU y reparte la CPU entre Ollama (~3 de sus 4 núcleos mientras genera, con ~545 MB de RAM residente), la síntesis de voz, la decodificación de video y la cámara. Cada nodo nuevo añade su propio proceso y la sobrecarga de DDS.

---

## Migración desde el proyecto original

| Original (`deploy-raspberry-standalone/`) | Nuevo | Paquete |
|---|---|---|
| `Orchestrator_Management.py` | `orchestrator_node.py` | `lora_brain` |
| `Agents/Agent_Router.py` | `agent_router.py` | `lora_brain` |
| `Agents/Agent_Corrector.py` | `agent_corrector.py` | `lora_brain` |
| `Agents/Agent_Behavior.py` | `agent_behavior.py` | `lora_brain` (ya no toca hardware; recibe un `bridge`) |
| `Clients/Llama_Client.py` | `llama_client.py` | `lora_brain` |
| `personalidad.py`, `memoria_episodica.py`, `registro_chat.py`, `preguntas.py`, `perf_monitor.py` | mismo nombre | `lora_brain` |
| — | `_ros_bridge.py` (nuevo) | `lora_brain` |
| `voz_server.py` | `_web_bridge.py` | `lora_drivers` (`communication_node`) |
| `Clients/Voice_Output_Client.py` | `_voice_backends.py` | `lora_drivers` (`communication_node`) |
| `Clients/Musica_Client.py` | `_music_player.py` | `lora_drivers` (`communication_node`) |
| `display.py` + `face_viewer.py` | `_face_display.py` + `face_viewer.py` | `lora_drivers` (`visualization_faces_node`) |
| `Clients/Camara_Client.py` + `ai-camera/reconocer_emocion.py` | `_emotion_detector.py` | `lora_drivers` (`visualization_faces_node`) |
| `Clients/Carrito_Client.py` | `_cart_serial.py` | `lora_drivers` (`moving_control_node`) |
| `mecanum_car_esp32s3.ino` | **sin cambios** | fuera de este repositorio |

**No se portaron** las herramientas de entrenamiento, porque no forman parte del robot en ejecución: `router_training/`, `chat_training/`, `trivia_training/`, `salida_trivia_training/`, `personalidad_training/`, `chat_libre_training/`, `excel_a_jsonl.py` y `perf_report.py`. Se siguen usando desde el repositorio original.

---

## Limitaciones conocidas

- **Sin probar en ROS 2 real ni en el hardware.** La sintaxis está verificada; el comportamiento, no.
- **`ROTATE360` sin calibrar.** Son 6 pulsos de `RR` aproximados, y no se sabe cuántos grados gira cada uno.
- **Sin reconocimiento de voz local.** Depende del navegador del teléfono (Chrome para Android).
- **Sin streaming por ROS 2.** `llama_client.py` admite streaming token a token (`on_token`), pero `Speak` recibe el texto completo.
- **`voz_motor` duplicado.** Está declarado en `communication_node` y en `orchestrator_node`. Si se cambia en el YAML, hay que cambiarlo en los dos; el argumento del launch ya actualiza ambos.
- **Juego de emociones lento.** Un turno tarda ~16.8 s: cámara 8.7 s (52 %), voz 4.8 s (28.5 %) y LLM 3.3 s (19.4 %).
- **13 preguntas duplicadas** en `src/lora_brain/lora_brain/data/preguntas.jsonl`.
