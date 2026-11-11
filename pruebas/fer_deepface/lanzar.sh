#!/usr/bin/env bash
# Lanza el panel de DeepFace en vivo en la Pi.
#
#   bash lanzar.sh          # arranca y deja el link en pantalla
#   bash lanzar.sh stop     # lo detiene
#
# La camara solo la puede abrir un proceso: ROMEO debe estar APAGADO.
set -e
AQUI="$(cd "$(dirname "$0")" && pwd)"
PUERTO="${ROMEO_FER_PUERTO:-8070}"
LOG=/tmp/deepface.log

if [ "$1" = "stop" ]; then
  pkill -f servidor_deepface.py && echo "panel DeepFace detenido" || echo "no estaba corriendo"
  exit 0
fi

# Si Romeo tiene la camara, avisar y salir (no pelear por la camara).
if pgrep -af expression_faces_node | grep -qv pgrep; then
  echo "ERROR: Romeo (expression_faces_node) esta usando la camara."
  echo "Apaga Romeo antes de lanzar el panel de DeepFace."
  exit 1
fi

# Matar una instancia previa del propio panel.
pkill -f servidor_deepface.py 2>/dev/null || true
sleep 1

cd "$AQUI"
# setsid + </dev/null: el panel sobrevive aunque se cierre la sesion SSH.
setsid python3 servidor_deepface.py > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "panel DeepFace lanzado (PID $PID), esperando a la camara..."

# Esperar a que /estado responda (la camara tarda ~unos segundos en asentar).
IP="$(hostname -I | awk '{print $1}')"
for i in $(seq 1 30); do
  if curl -s -m 2 "http://localhost:${PUERTO}/estado" >/dev/null 2>&1; then
    echo
    echo "  LISTO ->  http://${IP}:${PUERTO}"
    echo
    echo "  (ver el log:    tail -f ${LOG})"
    echo "  (detener:       bash lanzar.sh stop)"
    exit 0
  fi
  sleep 1
done

echo "El panel no respondio a tiempo. Revisa el log:"
tail -20 "$LOG"
exit 1
