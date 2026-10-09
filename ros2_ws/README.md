# Workspace ROS 2 de ARFORD

Documentación técnica de los 4 paquetes ROS 2 del robot ARFORD. Para saber qué es ARFORD, su hardware, los modelos de IA y los datos que usa, ver el [README principal](../README.md).

> **Estado:** escrito y verificado en una máquina **sin ROS 2 instalado**. La sintaxis de cada `.py` pasa `python -m py_compile` y cada paquete tiene la estructura que exige `colcon` (`package.xml`, `setup.py` o `CMakeLists.txt`, `resource/<paquete>`, `entry_points`). **Todavía no se ha corrido `colcon build` ni `ros2 launch`.** Antes de darlo por funcional hay que seguir la sección [Verificación](#verificación) en la Raspberry Pi, que corre **ROS 2 lyrical** sobre Ubuntu.

---

## Contenido

- [Paquetes](#paquetes)
- [Nodos](#nodos)
- [Tópicos y servicios](#tópicos-y-servicios)
- [Interfaces (`romeo_interfaces`)](#interfaces-romeo_interfaces)
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
├── romeo_interfaces/            mensajes y servicios (ament_cmake)
│   ├── msg/  FaceCommand, MotionCommand, UserInput
│   └── srv/  Speak, PlayMusic, DetectEmotion
├── romeo_drivers/               3 LifecycleNode de hardware (ament_python)
│   └── romeo_drivers/
│       ├── communication_node.py        voz, música y entrada por teclado
│       ├── visualization_faces_node.py  cámara → emoción + cara animada
│       ├── moving_control_node.py       motores (Serial → ESP32-S3)
│       ├── _voice_backends.py, _music_player.py, _stt_engine.py
│       ├── _face_display.py, face_viewer.py, _emotion_detector.py
│       ├── _cart_serial.py
│       └── data/  faces/ (mp4 + gif), musica/ (mp3), modelos/ (onnx)
├── romeo_brain/                 el "cerebro" (ament_python)
│   └── romeo_brain/
│       ├── orchestrator_node.py         máquina de estados de la conversación
│       ├── _ros_bridge.py               publishers y clientes de servicio
│       ├── agent_router.py, agent_corrector.py, agent_behavior.py
│       ├── llama_client.py              HTTP directo a Ollama
│       ├── memoria_episodica.py, preguntas.py
│       ├── registro_chat.py, perf_monitor.py
│       └── data/  preguntas.jsonl, router_modelo.joblib
└── romeo_bringup/               launch + parámetros (ament_cmake)
    ├── launch/romeo_bringup.launch.py
    └── config/romeo_params.yaml
```

`romeo_bringup` usa `ament_cmake` porque no tiene nodos Python propios, solo `launch/` y `config/`. Es el patrón habitual en los paquetes que solo lanzan el sistema, y no cambia la forma de invocarlo.

---

## Nodos

| Nodo | Paquete | Tipo | Responsabilidad |
|---|---|---|---|
| `orchestrator_node` | `romeo_brain` | `Node` | Máquina de estados de la sesión (Trivia / Chat libre): enrutamiento, corrección, reacciones, llamadas a Ollama y memoria episódica |
| `communication_node` | `romeo_drivers` | `LifecycleNode` | Comunicación verbal y no verbal: síntesis de voz, música, y entrada por micrófono (STT local) o teclado |
| `visualization_faces_node` | `romeo_drivers` | `LifecycleNode` | Muestra la cara animada y detecta la emoción del usuario por cámara |
| `moving_control_node` | `romeo_drivers` | `LifecycleNode` | Traduce `MotionCommand` al protocolo Serial del ESP32-S3 |

Los tres drivers son `LifecycleNode`: `on_configure` prepara los objetos **sin tocar el hardware**, y solo `on_activate` abre el Serial, precarga la voz o muestra la primera cara. El launch los configura y los activa automáticamente al arrancar, sin necesidad de un lifecycle manager externo.

### Qué archivo es nodo y qué archivo es librería

Son **4 nodos** en total, pero bastantes más archivos. Lo que define un nodo es el `console_scripts` del `setup.py`: solo esos se lanzan con `ros2 run`. Los demás son librerías, y el **guion bajo inicial** lo marca.

La regla: el nodo habla con ROS 2 (tópicos, servicios, parámetros, lifecycle) y delega el trabajo real; la librería hace el trabajo y **no importa `rclpy`**. Hoy se cumple sin excepción: `rclpy` solo aparece en esos 4 archivos, y en ninguna de las 16 librerías.

| Nodo | Sus librerías | Qué encapsulan |
|---|---|---|
| `communication_node` | `_voice_backends.py` | Piper (síntesis de voz) |
| | `_music_player.py` | `mpv` (reproducción de música) |
| | `_stt_engine.py` | sherpa-onnx + VAD (transcripción) |
| `visualization_faces_node` | `_face_display.py` | `mpv` (backend DRM) / Tkinter (backend PC) |
| | `face_viewer.py` | el visor Tkinter, **lanzado como subproceso** |
| | `_emotion_detector.py` | ONNX (YuNet + FER+) |
| `moving_control_node` | `_cart_serial.py` | `pyserial` y el protocolo del ESP32-S3 |
| `orchestrator_node` | `_ros_bridge.py` | publishers y clientes de servicio del cerebro |
| | `agent_router.py`, `agent_corrector.py`, `agent_behavior.py` | los 3 agentes sin LLM |
| | `llama_client.py`, `memoria_episodica.py`, `registro_chat.py`, `preguntas.py`, `perf_monitor.py` | Ollama, memoria, registro, dataset, métricas |

**Por qué no están fusionadas dentro de cada nodo**, aunque la carpeta quede plana:

- **Se pueden usar sin ROS 2.** `scripts/bench_stt.py` importa `_stt_engine` directamente, y `scripts/stt_server.py` lo expone por HTTP (`romeo.sh api`). Si viviera dentro de `communication_node.py`, importarlo arrastraría `rclpy` y el benchmark no correría en una máquina sin ROS 2 — que es justo donde se mide.
- **`face_viewer.py` no puede fusionarse:** `_face_display.py` lo ejecuta por ruta como proceso aparte, así que tiene que ser un archivo suelto en disco.
- **Cambian por razones distintas.** Si el firmware del ESP32-S3 cambia el protocolo, se toca `_cart_serial.py`; si se pasa de tópico a action, se toca `moving_control_node.py`. Juntarlos obliga a leer lo uno para cambiar lo otro.
- **Tamaño.** Fusionar dejaría `communication_node.py` en ~830 líneas mezclando lifecycle con Piper, `mpv` y sherpa-onnx, contra las ~310 de hoy.

El mismo criterio decide qué se vuelve nodo: hardware o red → nodo; cálculo local → librería. Por eso `agent_router.py` (microsegundos, sin hardware) es una librería y no un servicio ROS 2.

---

## Tópicos y servicios

| Nombre | Tipo | Origen → destino | Notas |
|---|---|---|---|
| `/romeo/user_input` | tópico `UserInput` | `communication_node` → `orchestrator_node` | Une en un solo tópico la voz transcrita y el teclado |
| `/romeo/face_command` | tópico `FaceCommand` | `orchestrator_node` → `visualization_faces_node` | |
| `/romeo/motion_command` | tópico `MotionCommand` | `orchestrator_node` → `moving_control_node` | |
| `/romeo/speak` | servicio `Speak` | `orchestrator_node` → `communication_node` | **Bloquea** hasta terminar de hablar |
| `/romeo/play_music` | servicio `PlayMusic` | `orchestrator_node` → `communication_node` | Bloquea solo si `esperar=true` |
| `/romeo/detect_emotion` | servicio `DetectEmotion` | `orchestrator_node` → `visualization_faces_node` | **Bloquea**, ~8-9 s en el peor caso |

Las llamadas a Ollama **no** pasan por ROS 2: `llama_client.py` usa HTTP directo desde `romeo_brain`, porque no es hardware.

---

## Interfaces (`romeo_interfaces`)

### Mensajes

| Mensaje | Campos |
|---|---|
| `FaceCommand` | `string face_name`: `happy` \| `sad` \| `angry` \| `content` \| `speaking` \| `countdown` |
| `MotionCommand` | `string command`: `F` \| `B` \| `SL` \| `SR` \| `RL` \| `RR` \| `S` \| `ROTATE360` |
| `UserInput` | `string text`, `string source`: `voice_board` \| `stdin` |

Los valores de `MotionCommand` son los mismos comandos de texto que ya entiende el firmware del ESP32-S3 (ver el [protocolo de motores](../README.md#protocolo-de-motores)).

### Servicios

| Servicio | Petición | Respuesta |
|---|---|---|
| `Speak` | `string text` | `bool success` |
| `PlayMusic` | `string filename`, `bool esperar` | `bool reproducido` |
| `DetectEmotion` | — | `string emotion` (`Feliz` \| `Triste` \| `Enojado` \| `Neutral` \| `""`), `float32 confidence`, `bool detected` |

`DetectEmotion` devuelve `detected=false` si no hay cámara, si faltan dependencias (`onnxruntime`, `opencv`, `picamera2`) o si no detectó ninguna cara. **Nunca** propaga una excepción al orquestador.

`PlayMusic` con `esperar=true` se usa en las preguntas musicales, donde la canción es el enunciado y el usuario no puede responder antes de escucharla. Con `esperar=false` devuelve en cuanto empieza la reproducción.

---

## Configuración

### Argumentos del launch

| Argumento | Por defecto | Efecto |
|---|---|---|
| `mic_activo` | `true` | `false` desactiva el STT y deja solo el teclado |
| `carrito_port` | `/dev/ttyACM0` | Puerto Serial del ESP32-S3 |
| `chat_model` | `romeo-chat-libre-v4` | Se pasa al orquestador como `CHAT_MODEL` |
| `trivia_model` | `romeo-trivia` | Se pasa como `TRIVIA_MODEL` |
| `salida_trivia_model` | `romeo-salida-trivia-v2` | Se pasa como `SALIDA_TRIVIA_MODEL` |
| `chat_server_host` | `http://localhost:11434` | Se pasa como `CHAT_SERVER_HOST` |

### Parámetros ROS 2 (`src/romeo_bringup/config/romeo_params.yaml`)

| Nodo | Parámetro | Por defecto |
|---|---|---|
| `communication_node` | `voz_piper_modelo` | `""` (usa `~/piper-voces/es_MX-claude-high.onnx`) |
| `communication_node` | `mic_activo` | `true` |
| `communication_node` | `mic_dispositivo` | `Romeo` (subcadena del nombre; `""` = default del sistema) |
| `communication_node` | `stt_tam` | `tiny` (`tiny` \| `base` \| `small`) |
| `communication_node` | `stt_hilos` | `2` (más hilos resulta **más lento**, está medido) |
| `moving_control_node` | `carrito_port` | `/dev/ttyACM0` |

`orchestrator_node` no declara parámetros propios: el motor de voz lo decide enteramente `communication_node`.

### Variables de entorno de `romeo_brain`

| Variable | Por defecto | Uso |
|---|---|---|
| `CHAT_SERVER_HOST`, `CHAT_MODEL`, `TRIVIA_MODEL`, `SALIDA_TRIVIA_MODEL` | ver [Modelos](../README.md#modelos-de-lenguaje-ollama) | Conexión con Ollama. El launch las fija a partir de sus argumentos |
| `ROMEO_LOGS_DIR` | `~/.romeo/logs` | Carpeta de métricas de `perf_monitor.py` (`tiempos.csv`, `recursos.csv`) |
| `PERF_MUESTREO_SEG` | `5` | Intervalo de muestreo de CPU y RAM |
| `CHAT_LIBRE_REGISTRO` | `~/.romeo/chat_libre_training/conversaciones.jsonl` | Archivo del registro del Chat libre |
| `CHAT_LIBRE_REGISTRAR` | `1` | `0` o `false` desactiva el registro |

### Variables de entorno de `romeo_drivers` (STT)

Los parámetros ROS 2 `stt_tam` y `stt_hilos` tienen prioridad sobre `ROMEO_STT_TAM` y `ROMEO_STT_HILOS`: `communication_node` los pasa al constructor de `ReconocedorVoz`. Las demás solo se configuran por entorno.

| Variable | Por defecto | Uso |
|---|---|---|
| `ROMEO_STT_MODELOS_DIR` | `~/.romeo/modelos` | Carpeta de los modelos de sherpa-onnx |
| `ROMEO_STT_TAM` | `tiny` | Modelo de whisper (lo sobreescribe `stt_tam`) |
| `ROMEO_STT_HILOS` | `2` | Hilos de onnxruntime (lo sobreescribe `stt_hilos`) |
| `ROMEO_STT_IDIOMA` | `es` | Idioma del audio |
| `ROMEO_STT_NORMALIZAR` | `1` | `0` desactiva la normalización de volumen |

La normalización **no es opcional en la práctica**: el micrófono de la placa entrega la señal muy floja (pico del 26-31 % de la escala hablándole de cerca), y con ese nivel whisper alucina en vez de transcribir — devolvía `[MÚSICA]` o repetía "hola, hola, hola…". Ver el detalle en `_stt_engine.py`.

### Modelos de STT (una sola vez, en la Pi)

```bash
mkdir -p ~/.romeo/modelos && cd ~/.romeo/modelos
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2
tar xvf sherpa-onnx-whisper-tiny.tar.bz2 && rm sherpa-onnx-whisper-tiny.tar.bz2
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
```

`silero_vad.onnx` es el VAD: sin él, `transcribir()` sigue funcionando (y el bench también), pero **no se puede escuchar el micrófono en continuo**, que es justo lo que hace `communication_node`.

---

## Instalación y ejecución

Requisitos: la Pi del robot corre **Ubuntu con ROS 2 lyrical** (`source /opt/ros/lyrical/setup.bash`); en otra máquina sirve cualquier ROS 2 reciente. Además `mpv` y **Ollama**. Todos los comandos se ejecutan desde la carpeta `ros2_ws`.

```bash
# 1. Dependencias del sistema y de Python
sudo apt install mpv
pip install -r src/romeo_drivers/requirements.txt
pip install -r src/romeo_brain/requirements.txt
# picamera2 viene con Raspberry Pi OS; no se instala con pip.
# piper-tts lo instala el requirements.txt de romeo_drivers: es la voz de ARFORD.

# 2. Compilar
colcon build --symlink-install
source install/setup.bash

# 3. Tener Ollama en marcha con los 3 modelos importados
#    (romeo-chat-libre-v4, romeo-trivia y romeo-salida-trivia-v2, con `ollama create`)

# 4. Lanzar todo
ros2 launch romeo_bringup romeo_bringup.launch.py
```

Variantes:

```bash
ros2 launch romeo_bringup romeo_bringup.launch.py mic_activo:=false
ros2 launch romeo_bringup romeo_bringup.launch.py carrito_port:=/dev/ttyUSB0
```

### Cómo hablarle a ARFORD

**Por voz (el camino normal).** El micrófono de la placa [ESP32-S3-AUDIO-Board](../firmware-audio-board/README.md) entra a la Pi como tarjeta de sonido USB, y `communication_node` lo transcribe con `_stt_engine.py` (sherpa-onnx, whisper-tiny, ~2.45 s por frase). No hay que pulsar nada: el VAD corta las frases por el silencio. Lo transcrito se publica en `/romeo/user_input` con `source: voice_board`.

**Verificado en la Pi** (4 de octubre de 2026): el micrófono entrega señal real (RMS 148.6, pico 738 sobre 48.000 muestras) y la cadena completa transcribe en **1.57 s** — modelo cargado en 4.0 s, VAD activo. El micrófono de esta placa da la señal muy floja, así que la normalización de `_stt_engine.py` llega al tope de ganancia (20×); es el comportamiento previsto, no un fallo.

> **Requisito de audio:** `sounddevice` necesita un servidor de audio corriendo. Su PortAudio intenta PulseAudio primero y, si no lo encuentra, **aborta en vez de caer a ALSA** (`PaErrorCode -9999`), así que el micrófono no funciona aunque `arecord` sí grabe. En la Pi se resolvió con:
>
> ```bash
> sudo apt install -y pipewire pipewire-pulse wireplumber
> systemctl --user enable --now pipewire.socket pipewire-pulse.socket wireplumber.service
> sudo loginctl enable-linger pi   # que arranque sin sesión abierta
> ```

**Por teclado (respaldo).** `communication_node` también lee de stdin. Pero **`ros2 launch` no le da una stdin interactiva a los nodos que arranca**, así que lo que escribas en la terminal del launch no le llega. Dos formas:

```bash
# Opción A -- publicar en el tópico directamente (la más cómoda para probar)
ros2 topic pub --once /romeo/user_input romeo_interfaces/msg/UserInput \
    "{text: 'quiero jugar trivia', source: 'stdin'}"
```

```bash
# Opción B -- correr communication_node aparte, en su propia terminal
ros2 run romeo_drivers communication_node --ros-args \
    --params-file install/romeo_bringup/share/romeo_bringup/config/romeo_params.yaml

# es un LifecycleNode: hay que activarlo a mano si no lo lanzó el launch
ros2 lifecycle set /communication_node configure
ros2 lifecycle set /communication_node activate
```

### Micrófono y altavoz comparten dispositivo

Los dos son la misma placa, así que el micrófono capta la propia voz de ARFORD. Si no se hiciera nada, el STT la transcribiría como si fuera el usuario y ARFORD se respondería en bucle. `communication_node` descarta lo que llegue del micrófono mientras está hablando (`Speak`) o mientras suena una canción con `esperar=true`.

La música de fondo (`esperar=false`) **sí puede colarse**: ese servicio vuelve en cuanto lanza `mpv`, así que el nodo no sabe cuándo termina de sonar. Es una limitación conocida y está anotada en el código.

---

## Verificación

```bash
ros2 node list
# communication_node, visualization_faces_node, moving_control_node, orchestrator_node

ros2 lifecycle get /communication_node
# debe decir "active" pocos segundos después de arrancar

ros2 topic echo /romeo/user_input
# al escribir en la terminal, el texto aparece aquí

ros2 service call /romeo/speak romeo_interfaces/srv/Speak "{text: 'probando la voz'}"
```

**Prueba completa:** escribe tu nombre cuando ARFORD te salude, pide Trivia, elige un tema y contesta una pregunta con respuesta numérica (por ejemplo, una multiplicación). Comprueba que ARFORD dice el veredicto, cambia de cara y, si la pregunta lo tiene configurado, se mueve y reproduce música.

---

## Flujo de un turno

### Chat libre

1. Llega el texto por `/romeo/user_input`.
2. `agent_router` lo clasifica como Chat libre (sin LLM).
3. `memoria_episodica` busca un recuerdo relevante.
4. `llama_client` genera la respuesta con `CHAT_MODEL`, sin system prompt: el comportamiento viene del fine-tuning.
5. `registro_chat` guarda el turno.
6. `FaceCommand` `speaking` → `Speak` (bloquea) → `FaceCommand` `content`.

### Trivia

1. Al entrar en Trivia se ofrecen 5 temas al azar y el usuario elige uno (coincidencia exacta o aproximada, sin LLM).
2. `preguntas.py` carga una tanda de 5 preguntas del tema.
3. Se hace la pregunta con la cara `speaking` y `Speak`. Si es musical, primero suena la canción completa (`PlayMusic` con `esperar=true`).
4. Con la respuesta del usuario, `agent_corrector` decide si es correcta, `agent_behavior` elige la cara y ARFORD dice una frase fija elegida al azar.
5. La reacción sigue un **orden estricto**: primero `Speak` (bloquea) y después `FaceCommand`, `PlayMusic` (`esperar=false`) y `MotionCommand`, en paralelo.
6. Se pasa a la siguiente pregunta o se cierra con el resumen de aciertos.

### Juego de emociones

1. Se elige al azar una de las emociones que pide la pregunta y se publica esa cara como referencia.
2. ARFORD lo pide por voz: "¡Hazme una cara de feliz!".
3. `DetectEmotion` captura y clasifica la cara (hasta 15 intentos).
4. El veredicto compara la emoción pedida con la detectada. ARFORD dice una frase fija ("¡Correcto!" o "¡Incorrecto!") y luego cambia la cara, suena la música y se mueven los motores.
5. La tanda completa corre de una vez, sin volver al bucle principal entre preguntas.

### Salir de Trivia a mitad de una pregunta

Si el mensaje es corto (4 palabras o menos) y no contiene ninguna palabra clave de salida, se toma como respuesta sin consultar al LLM. Si no, decide `SALIDA_TRIVIA_MODEL`, y si Ollama no responde se usan listas de palabras clave.

---

## Decisiones de diseño

- **`Speak` es un servicio síncrono, no una action.** Se renuncia a la cancelación y al streaming a cambio de simplicidad; en el original ya nada usaba el streaming.
- **Orden voz → cara → (música + motores en paralelo).** Es un requisito del diseño. Lo garantiza `orchestrator_node.py::_reaccionar_veredicto()`, que espera la respuesta de `Speak` antes de publicar lo demás.
- **Llamar a servicios desde un callback sin bloqueo mutuo (deadlock).** `client.call()` se queda bloqueado con un `SingleThreadedExecutor`; la solución es `call_async()` con una espera activa corta y un `MultiThreadedExecutor`. Está documentado en `romeo_brain/_ros_bridge.py::_llamar_servicio_sync()` y es lo más fácil de romper al modificar este código.
- **Degradación elegante.** Sin cámara, carrito o `mpv`, el nodo correspondiente sigue vivo y responde `detected=false` o `success=false`, sin lanzar nunca una excepción no controlada. El robot sigue conversando.
- **"Salir" termina la sesión, no el proceso.** En el original, "salir" cerraba el script. Aquí ARFORD se despide y queda lista para el siguiente usuario, como corresponde a un robot desplegado (`orchestrator_node.py::_finalizar_sesion()`).
- **Mandan las mediciones de latencia.** El router y el corrector dejaron de usar el LLM por mediciones hechas en la Pi. Ninguna capa nueva (tópicos, servicios, DDS) debería añadir una latencia perceptible a la conversación.
- **STT: `whisper-tiny` con 2 hilos, por medición.** `_stt_engine.py` usa sherpa-onnx porque corre sobre `onnxruntime`, que ya está instalado para el detector de emociones, y porque no arrastra el fallo de hilos de `faster-whisper`. Medido en la Pi con un audio de 3.8 s (`scripts/bench_stt.py`): `tiny`/2 hilos da 2.22 s sin carga y **2.45 s con Ollama generando**, contra 4.41 s de `base`/2 hilos. Dos resultados contraintuitivos: **más hilos es más lento** (4 hilos sube a 3.45 s y dispara la varianza) y el mínimo no está en 1 hilo (3.29 s bajo carga). El precio de `tiny` es que transcribe "Hola Laura" donde `base` acierta "Hola Romeo"; para Trivia lo absorbe el `agent_corrector`, pero si el nombre llega a importar hay que volver a `base`.
- **Recursos muy limitados.** La Pi 4 no tiene GPU y reparte la CPU entre Ollama (~3 de sus 4 núcleos mientras genera, con ~545 MB de RAM residente), la síntesis de voz, la decodificación de video y la cámara. Cada nodo nuevo añade su propio proceso y la sobrecarga de DDS.

---

## Migración desde el proyecto original

| Original (`deploy-raspberry-standalone/`) | Nuevo | Paquete |
|---|---|---|
| `Orchestrator_Management.py` | `orchestrator_node.py` | `romeo_brain` |
| `Agents/Agent_Router.py` | `agent_router.py` | `romeo_brain` |
| `Agents/Agent_Corrector.py` | `agent_corrector.py` | `romeo_brain` |
| `Agents/Agent_Behavior.py` | `agent_behavior.py` | `romeo_brain` (ya no toca hardware; recibe un `bridge`) |
| `Clients/Llama_Client.py` | `llama_client.py` | `romeo_brain` |
| `memoria_episodica.py`, `registro_chat.py`, `preguntas.py`, `perf_monitor.py` | mismo nombre | `romeo_brain` |
| `personalidad.py` | **eliminado** — la personalidad viene horneada en los fine-tunes, así que el system prompt siempre salía vacío (los valores OCEAN están en el [README principal](../README.md)) | — |
| — | `_ros_bridge.py` (nuevo) | `romeo_brain` |
| `voz_server.py` | **eliminado** — la página web y la voz por teléfono se reemplazaron por el micrófono de la placa + `_stt_engine.py` | — |
| `Clients/Voice_Output_Client.py` | `_voice_backends.py` | `romeo_drivers` (`communication_node`) |
| `Clients/Musica_Client.py` | `_music_player.py` | `romeo_drivers` (`communication_node`) |
| `display.py` + `face_viewer.py` | `_face_display.py` + `face_viewer.py` | `romeo_drivers` (`visualization_faces_node`) |
| `Clients/Camara_Client.py` + `ai-camera/reconocer_emocion.py` | `_emotion_detector.py` | `romeo_drivers` (`visualization_faces_node`) |
| `Clients/Carrito_Client.py` | `_cart_serial.py` | `romeo_drivers` (`moving_control_node`) |
| `mecanum_car_esp32s3.ino` | **sin cambios** | fuera de este repositorio |

**No se portaron** las herramientas de entrenamiento, porque no forman parte del robot en ejecución: `router_training/`, `chat_training/`, `trivia_training/`, `salida_trivia_training/`, `personalidad_training/`, `chat_libre_training/`, `excel_a_jsonl.py` y `perf_report.py`. Se siguen usando desde el repositorio original.

---

## Limitaciones conocidas

- **Sin probar en ROS 2 real ni en el hardware.** Están verificados la sintaxis de todos los `.py`, los `package.xml`, el YAML de parámetros y los tipos de los parámetros del launch. El comportamiento en la Pi, no: hace falta `colcon build` y una sesión con el robot.
- **`ROTATE360` sin calibrar.** Son 6 pulsos de `RR` aproximados, y no se sabe cuántos grados gira cada uno.
- **El micrófono da la señal muy floja.** Funciona (RMS 148.6 medido en la Pi), pero la normalización de `_stt_engine.py` tiene que amplificar al tope (20×) para que whisper no alucine. Hablarle de cerca sigue siendo necesario.
- **El audio depende de pipewire.** `sounddevice` aborta si no hay servidor de audio, en vez de caer a ALSA. Si el micrófono deja de funcionar, comprobar `systemctl --user is-active pipewire-pulse` antes de buscar en otro sitio.
- **Sin streaming por ROS 2.** `llama_client.py` admite streaming token a token (`on_token`), pero `Speak` recibe el texto completo.
- **Un solo motor de voz, sin respaldo.** Piper es la única voz (edge-tts se eliminó por sintetizar en la nube). Si falta el modelo `.onnx`, ARFORD arranca igual pero **mudo**: avisa en el log y el texto solo queda ahí.
- **Juego de emociones lento.** Un turno tarda ~16.8 s: cámara 8.7 s (52 %), voz 4.8 s (28.5 %) y LLM 3.3 s (19.4 %).
- **La música de fondo se cuela en el micrófono.** `PlayMusic` con `esperar=false` devuelve en cuanto lanza `mpv`, así que `communication_node` no sabe cuándo termina de sonar y no puede silenciar el STT ese rato. Con `Speak` y con `esperar=true` sí se silencia.
- **Sin watchdog en los motores.** Si `orchestrator_node` muere justo después de publicar `F`, el carrito sigue andando: nadie manda `S`. Conviene un timeout en `_cart_serial.py`.
- **Una sola pregunta por tema compartido.** `preguntas.py` ya no parte la columna `tema` por `/`, así que una pregunta pertenece a un único tema. El dataset hoy no usa esa capacidad; si hiciera falta, necesita un separador que no aparezca en los nombres de los temas (`Socialización / presentación - Nivel 1` lleva uno dentro).
- **La memoria episódica es compartida entre usuarios** y devuelve la *respuesta vieja de ARFORD*, no lo que contó el usuario. Sin stemming: `partido` y `partidos` no suman overlap, así que engancha menos de lo que parece.
