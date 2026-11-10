#!/usr/bin/env bash
# Recolector de recursos de la Raspberry mientras corre Romeo.
#
# Escribe una fila cada INTERVALO segundos con el uso general del sistema
# (CPU total, temperatura, RAM) y el de los procesos clave de Romeo (la
# camara/FER y mpv de las caras). Es la contraparte en la Pi de las
# mediciones por modulo: da el "Recursos de Raspberry" de la tabla de
# metricas (%, MB, C).
#
# No toca ROS ni Romeo: solo lee /proc y vcgencmd. Se puede arrancar y
# parar cuando se quiera.
#
# Uso en la Pi:
#     ~/ARFORD/ros2_ws/src/romeo_drivers/scripts/recolector_sistema.sh &
#     (Ctrl+C o kill para pararlo)
#
# Variables:
#     ROMEO_SISTEMA_CSV   destino (por defecto ~/romeo/mediciones/sistema_<fecha>.csv)
#     ROMEO_SISTEMA_SEG   intervalo en segundos (por defecto 3)

set -u

INTERVALO="${ROMEO_SISTEMA_SEG:-3}"
FECHA="$(date +%Y%m%d_%H%M%S)"
CSV="${ROMEO_SISTEMA_CSV:-$HOME/romeo/mediciones/sistema_${FECHA}.csv}"
mkdir -p "$(dirname "$CSV")"

echo "t,cpu_total_pct,temp_c,ram_usada_mb,ram_total_mb,cpu_camara_pct,cpu_mpv_pct" > "$CSV"
echo "[recolector] escribiendo en $CSV cada ${INTERVALO}s (Ctrl+C para parar)"

# %CPU total: se calcula a partir de /proc/stat entre dos lecturas.
leer_cpu_total() {
  read -r _ u n s i w irq sirq _ < /proc/stat
  echo $((u + n + s + irq + sirq)) $((u + n + s + i + w + irq + sirq))
}

prev=($(leer_cpu_total))
sleep "$INTERVALO"

while true; do
  AHORA="$(date +%s)"

  # CPU total %
  cur=($(leer_cpu_total))
  d_busy=$(( ${cur[0]} - ${prev[0]} ))
  d_tot=$(( ${cur[1]} - ${prev[1]} ))
  prev=(${cur[@]})
  if [ "$d_tot" -gt 0 ]; then
    cpu_total=$(awk "BEGIN{printf \"%.1f\", 100*$d_busy/$d_tot}")
  else
    cpu_total="0.0"
  fi

  # Temperatura (vcgencmd si esta; si no, /sys/class/thermal)
  if command -v vcgencmd >/dev/null 2>&1; then
    temp=$(vcgencmd measure_temp 2>/dev/null | grep -oE '[0-9]+\.[0-9]+')
  else
    t_raw=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo 0)
    temp=$(awk "BEGIN{printf \"%.1f\", $t_raw/1000}")
  fi
  [ -z "$temp" ] && temp="0.0"

  # RAM (MB) de /proc/meminfo
  mem_total_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo)
  mem_avail_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  ram_usada=$(( (mem_total_kb - mem_avail_kb) / 1024 ))
  ram_total=$(( mem_total_kb / 1024 ))

  # %CPU de los procesos de Romeo: la camara (FER/visualization/expression)
  # y mpv de las caras. ps suma el %CPU de todos los que casen.
  cpu_cam=$(ps -eo %cpu,comm,args 2>/dev/null \
    | grep -iE 'expression_faces|visualization_faces|emotion_detector' \
    | grep -v grep | awk '{s+=$1} END{printf "%.1f", s+0}')
  cpu_mpv=$(ps -eo %cpu,args 2>/dev/null \
    | grep -E 'mpv .*romeo_face' | grep -v grep \
    | awk '{s+=$1} END{printf "%.1f", s+0}')
  [ -z "$cpu_cam" ] && cpu_cam="0.0"
  [ -z "$cpu_mpv" ] && cpu_mpv="0.0"

  echo "${AHORA},${cpu_total},${temp},${ram_usada},${ram_total},${cpu_cam},${cpu_mpv}" >> "$CSV"
  sleep "$INTERVALO"
done
