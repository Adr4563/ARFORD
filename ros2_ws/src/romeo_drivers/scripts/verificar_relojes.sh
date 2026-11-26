#!/usr/bin/env bash
# Verifica el desfase de reloj entre la Raspberry (donde corre esto) y la
# workstation (PC de inferencia), al inicio de cada sesion de captura.
#
# POR QUE. Las metricas de turno (Pi) y de hardware de la PC se cruzan por
# marca de tiempo epoch. Epoch es UTC en las dos maquinas, pero si los
# relojes estan desviados, el cruce junta momentos distintos. La Pi no tiene
# reloj de bateria: si estuvo apagada o sin NTP, puede ir horas desfasada.
#
# Este script mide el desfase Pi-PC y lo DEJA ESCRITO en el log de la sesion.
# Si el desfase supera el umbral, avisa para que corrijas antes de capturar.
#
# Uso en la Pi:
#     ROMEO_PC=10.143.30.94 ~/ARFORD/.../verificar_relojes.sh
#     (o pasa la IP como primer argumento)
#
# Variables:
#     ROMEO_PC          IP de la workstation (o 1er argumento)
#     ROMEO_RELOJ_LOG   donde anotar el resultado (default ~/romeo/mediciones/relojes.log)
#     ROMEO_RELOJ_MAX   desfase maximo tolerable en segundos (default 2)

set -u

PC="${1:-${ROMEO_PC:-}}"
LOG="${ROMEO_RELOJ_LOG:-$HOME/romeo/mediciones/relojes.log}"
MAX="${ROMEO_RELOJ_MAX:-2}"
mkdir -p "$(dirname "$LOG")"

if [ -z "$PC" ]; then
  echo "ERROR: falta la IP de la PC. Usa: $0 <ip-pc>  o  export ROMEO_PC=<ip>"
  exit 2
fi

AHORA="$(date '+%Y-%m-%d %H:%M:%S')"
EPOCH_PI=$(date +%s)

# El epoch de la PC se pide por SSH. Requiere que la PC acepte SSH desde la Pi;
# si no, se puede pedir a mano y pasar con ROMEO_EPOCH_PC.
if [ -n "${ROMEO_EPOCH_PC:-}" ]; then
  EPOCH_PC="$ROMEO_EPOCH_PC"
else
  # PowerShell en la PC: epoch UTC correcto (UtcNow, no hora local).
  EPOCH_PC=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8 "Adrian@${PC}" \
    "powershell -NoProfile -Command \"[DateTimeOffset]::UtcNow.ToUnixTimeSeconds()\"" 2>/dev/null | tr -d '\r')
fi

if [ -z "$EPOCH_PC" ]; then
  echo "ADVERTENCIA: no pude leer el reloj de la PC por SSH ($PC)."
  echo "  Pidelo a mano en la PC con:  [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()"
  echo "  y vuelve a correr con:       ROMEO_EPOCH_PC=<ese numero> $0 $PC"
  exit 3
fi

DESFASE=$(( EPOCH_PI - EPOCH_PC ))
ABS=${DESFASE#-}

# Estado de chrony (informativo).
LEAP=$(chronyc tracking 2>/dev/null | awk -F': *' '/Leap status/{print $2}')
[ -z "$LEAP" ] && LEAP="(chrony no responde)"

LINEA="$AHORA | epoch_pi=$EPOCH_PI epoch_pc=$EPOCH_PC desfase=${DESFASE}s | chrony=$LEAP"
echo "$LINEA" | tee -a "$LOG"

if [ "$ABS" -le "$MAX" ]; then
  echo "OK: desfase ${DESFASE}s (<= ${MAX}s). Puedes capturar."
  exit 0
else
  echo "CORRIGE ANTES DE CAPTURAR: desfase ${DESFASE}s supera ${MAX}s."
  echo "  En la PC:  [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()  -> tomar epoch"
  echo "  En la Pi:  sudo date -u -s \"AAAA-MM-DD HH:MM:SS\"  (hora UTC de la PC)"
  echo "             sudo systemctl restart chrony"
  exit 1
fi
