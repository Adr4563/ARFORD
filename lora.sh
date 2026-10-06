#!/usr/bin/env bash
#
# lora.sh - atajos para manejar el robot en la Raspberry Pi.
#
# Todas las rutas de aquí están COMPROBADAS en la Pi, no supuestas.
# Ejecutar en la Raspberry:
#
#     ./lora.sh ayuda
#
# Si el script no arranca, darle permisos una vez:  chmod +x lora.sh

# Sin `set -u` a propósito: los scripts de entorno de ROS 2 leen variables sin
# inicializar (AMENT_TRACE_SETUP_FILES y compañía) y abortarían el script.
set -o pipefail

# --------------------------------------------------------------- rutas ------

ROS_SETUP="/opt/ros/lyrical/setup.bash"
WS="$HOME/ros2_ws"
VENV="$HOME/.lora/venv/bin/python"
STT="$HOME/lora-stt"
VOZ_PIPER="$HOME/piper-voces/es_MX-claude-high.onnx"

# La placa ESP32-S3 se expone como tarjeta de sonido USB con este nombre.
# Si se cambia el firmware, cambia también aquí.
PLACA="plughw:CARD=Board,DEV=0"
JACK="plughw:0,0"

# Voz de Microsoft usada como alternativa a Piper (necesita internet).
VOZ_EDGE="es-MX-DaliaNeural"

rojo()  { printf '\033[31m%s\033[0m\n' "$*"; }
verde() { printf '\033[32m%s\033[0m\n' "$*"; }
info()  { printf '\033[36m== %s\033[0m\n' "$*"; }

entorno() {
    # shellcheck disable=SC1090
    [ -f "$ROS_SETUP" ] || { rojo "No existe $ROS_SETUP -- ¿está ROS 2 instalado?"; exit 1; }
    source "$ROS_SETUP"
    [ -f "$WS/install/setup.bash" ] && source "$WS/install/setup.bash"
}

# ------------------------------------------------------------ comandos ------

cmd_ayuda() {
    cat <<'FIN'
lora.sh - atajos del robot

  ENTORNO Y COMPILACION
    build            Compila el workspace (colcon build --symlink-install)
    build-limpio     Borra build/ install/ log/ y compila desde cero
    estado           Comprueba ROS, paquetes, nodos y dependencias

  EJECUCION
    run              Arranca el sistema completo (los 4 nodos)
    run-edge         Igual, pero con la voz por edge-tts en vez del teléfono
    nodos            Lista los nodos activos (con el robot ya corriendo)
    ciclo            Estado de los 3 LifecycleNode

  VOZ Y AUDIO
    hablar TEXTO     Lo dice por el altavoz de la placa (voz de Piper)
    hablar-edge TEXTO   Igual, con la voz de Microsoft
    musica           Reproduce la música de Mario por la placa
    micro [SEGS]     Graba del micrófono de la placa y lo transcribe
    audio            Comprueba qué salidas de audio existen

  RECONOCIMIENTO DE VOZ
    stt ARCHIVO      Transcribe un .wav
    bench            Mide la velocidad del STT
    api              Arranca la API HTTP de STT en el puerto 8082

  TODO
    prueba           Ejecuta la comprobación completa de punta a punta

Ejemplos:
    ./lora.sh hablar "Hola, soy un asistente virtual"
    ./lora.sh micro 5
    ./lora.sh prueba
FIN
}

cmd_build() {
    entorno
    info "Compilando el workspace"
    cd "$WS" && colcon build --symlink-install
}

cmd_build_limpio() {
    entorno
    info "Borrando build/ install/ log/"
    cd "$WS" && rm -rf build install log
    info "Compilando desde cero"
    colcon build --symlink-install
}

cmd_estado() {
    entorno
    info "ROS 2"
    echo "  distro: ${ROS_DISTRO:-(ninguno)}"

    info "Paquetes del robot"
    local encontrados
    encontrados=$(ros2 pkg list 2>/dev/null | grep -c '^lora_')
    ros2 pkg list 2>/dev/null | grep '^lora_' | sed 's/^/  /'
    if [ "$encontrados" -eq 4 ]; then
        verde "  los 4 paquetes registrados"
    else
        rojo "  faltan paquetes (encontrados: $encontrados de 4)"
        rojo "  revisa que el package.xml sea XML válido -- los guiones dobles"
        rojo "  dentro de un comentario lo rompen en silencio"
    fi

    info "Ejecutables"
    for p in lora_brain lora_drivers; do
        ros2 pkg executables "$p" 2>/dev/null | sed 's/^/  /'
    done

    info "Dependencias de Python (las usa el runtime de los nodos)"
    for m in rclpy sherpa_onnx onnxruntime flask numpy serial cv2 rank_bm25 sklearn; do
        if python3 -c "import $m" 2>/dev/null; then
            echo "  OK     $m"
        else
            echo "  FALTA  $m"
        fi
    done

    info "Ollama"
    if curl -sS --max-time 5 http://localhost:11434/api/tags >/dev/null 2>&1; then
        verde "  responde en el puerto 11434"
        # La lista se pide UNA vez y se guarda. Hacer `ollama list | grep -q`
        # dentro del bucle daba falsos negativos: `grep -q` corta en cuanto
        # encuentra, eso le manda SIGPIPE a `ollama` (código 141), y con
        # `pipefail` la tubería entera se da por fallida. Fallaba justo en los
        # modelos que aparecían al principio de la lista.
        local modelos; modelos=$(ollama list 2>/dev/null)
        for m in lora-chat-libre-v4 lora-trivia lora-salida-trivia-v2; do
            if printf '%s\n' "$modelos" | grep -q "^$m"; then
                echo "  OK     $m"
            else
                echo "  FALTA  $m"
            fi
        done
    else
        rojo "  no responde -- el orquestador no podrá generar respuestas"
    fi

    info "Salidas de audio"
    aplay -l 2>/dev/null | grep '^card' | sed 's/^/  /'
}

cmd_run() {
    entorno
    info "Arrancando el robot -- Ctrl+C para parar"
    echo "  Página web: http://$(hostname -I | awk '{print $1}'):8081/"
    ros2 launch lora_bringup lora_bringup.launch.py
}

cmd_run_edge() {
    entorno
    info "Arrancando con voz edge-tts"
    ros2 launch lora_bringup lora_bringup.launch.py voz_motor:=edge
}

cmd_nodos() { entorno; ros2 node list; }

cmd_ciclo() {
    entorno
    for n in communication_node visualization_faces_node moving_control_node; do
        printf '  %-28s %s\n' "$n" "$(ros2 lifecycle get "/$n" 2>/dev/null || echo 'no responde')"
    done
}

# Genera un wav a 48kHz estéreo y lo reproduce por la placa y por el jack,
# para no depender de dónde esté enchufado el altavoz.
_reproducir() {
    local wav="$1"
    amixer -c Board sset PCM 100% unmute >/dev/null 2>&1
    amixer -c 0 sset PCM 100% unmute >/dev/null 2>&1
    aplay -D "$PLACA" "$wav" 2>/dev/null && verde "  reproducido por la placa" || rojo "  la placa no respondió"
    aplay -D "$JACK"  "$wav" 2>/dev/null && verde "  reproducido por el jack"  || rojo "  el jack no respondió"
}

cmd_hablar() {
    local texto="${*:-Hola, soy un asistente virtual.}"
    info "Diciendo: $texto"
    [ -f "$VOZ_PIPER" ] || { rojo "No existe la voz $VOZ_PIPER"; exit 1; }
    local tmp; tmp=$(mktemp -d)
    echo "$texto" | piper --model "$VOZ_PIPER" --output_file "$tmp/v.wav" 2>/dev/null
    ffmpeg -y -loglevel error -i "$tmp/v.wav" -ar 48000 -ac 2 "$tmp/f.wav"
    _reproducir "$tmp/f.wav"
    rm -rf "$tmp"
}

cmd_hablar_edge() {
    local texto="${*:-Hola, soy un asistente virtual.}"
    info "Diciendo con voz $VOZ_EDGE: $texto"
    local tmp; tmp=$(mktemp -d)
    python3 -m edge_tts --voice "$VOZ_EDGE" --text "$texto" --write-media "$tmp/v.mp3" 2>/dev/null \
        || { rojo "edge-tts falló (¿hay internet?)"; rm -rf "$tmp"; exit 1; }
    ffmpeg -y -loglevel error -i "$tmp/v.mp3" -ar 48000 -ac 2 "$tmp/f.wav"
    _reproducir "$tmp/f.wav"
    rm -rf "$tmp"
}

cmd_musica() {
    local mp3="$WS/src/lora_drivers/lora_drivers/data/musica/super-mario-bros.mp3"
    [ -f "$mp3" ] || { rojo "No encuentro $mp3"; exit 1; }
    info "Música de Mario"
    local tmp; tmp=$(mktemp -d)
    ffmpeg -y -loglevel error -i "$mp3" -ar 48000 -ac 2 "$tmp/m.wav"
    _reproducir "$tmp/m.wav"
    rm -rf "$tmp"
}

cmd_micro() {
    local segs="${1:-5}"
    info "Grabando ${segs}s del micrófono de la placa -- HABLA AHORA"
    local wav="/tmp/lora_micro.wav"
    arecord -D "$PLACA" -f S16_LE -r 16000 -c 1 -d "$segs" "$wav" 2>/dev/null \
        || { rojo "No se pudo grabar de la placa"; exit 1; }

    # Un RMS cercano a cero significa que el micrófono no capta: distinguirlo
    # de un fallo de transcripción ahorra mucho tiempo de diagnóstico.
    "$VENV" - "$wav" <<'PY'
import sys, wave, numpy as np
w = wave.open(sys.argv[1], "rb")
d = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(float)
rms = float(np.sqrt((d**2).mean())) if d.size else 0.0
print("  nivel de señal: RMS %.0f  pico %d" % (rms, abs(d).max() if d.size else 0))
print("  -> SILENCIO: el micrófono no capta" if rms < 50 else "  -> HAY SEÑAL")
PY

    info "Transcribiendo"
    cd "$STT" && "$VENV" -c "
from lora_drivers import _stt_engine as m
print('  texto:', repr(m.reconocedor.transcribir_wav('$wav')))
"
}

cmd_audio() {
    info "Salidas de reproducción"; aplay -l 2>/dev/null | grep '^card' | sed 's/^/  /'
    info "Entradas de captura";     arecord -l 2>/dev/null | grep '^card' | sed 's/^/  /' || echo "  ninguna"
}

cmd_stt() {
    local wav="${1:-$STT/prueba_es.wav}"
    [ -f "$wav" ] || { rojo "No existe $wav"; exit 1; }
    cd "$STT" && "$VENV" -c "
from lora_drivers import _stt_engine as m
import time; t = time.perf_counter()
txt = m.reconocedor.transcribir_wav('$wav')
print('  tiempo: %.2fs' % (time.perf_counter() - t))
print('  texto :', repr(txt))
"
}

cmd_bench() {
    "$VENV" "$STT/scripts/bench_stt.py" --wav "$STT/prueba_es.wav" --repeticiones 5
}

cmd_api() {
    info "API de STT en http://$(hostname -I | awk '{print $1}'):8082 -- Ctrl+C para parar"
    "$VENV" "$STT/scripts/stt_server.py"
}

cmd_prueba() {
    cmd_estado
    echo
    info "Transcripción de un archivo conocido"
    cmd_stt
    echo
    info "Saludo por el altavoz"
    cmd_hablar "Hola, soy un asistente virtual."
    echo
    verde "Comprobación terminada."
    echo "Falta lo único que no se puede automatizar: arrancar el robot con"
    echo "  ./lora.sh run   y hablarle desde http://$(hostname -I | awk '{print $1}'):8081/"
}

# ---------------------------------------------------------------- main ------

case "${1:-ayuda}" in
    ayuda|-h|--help) cmd_ayuda ;;
    build)           cmd_build ;;
    build-limpio)    cmd_build_limpio ;;
    estado)          cmd_estado ;;
    run)             cmd_run ;;
    run-edge)        cmd_run_edge ;;
    nodos)           cmd_nodos ;;
    ciclo)           cmd_ciclo ;;
    hablar)          shift; cmd_hablar "$@" ;;
    hablar-edge)     shift; cmd_hablar_edge "$@" ;;
    musica)          cmd_musica ;;
    micro)           shift; cmd_micro "$@" ;;
    audio)           cmd_audio ;;
    stt)             shift; cmd_stt "$@" ;;
    bench)           cmd_bench ;;
    api)             cmd_api ;;
    prueba)          cmd_prueba ;;
    *)               rojo "Comando desconocido: $1"; echo; cmd_ayuda; exit 1 ;;
esac
