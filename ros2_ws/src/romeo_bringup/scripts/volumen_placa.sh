#!/bin/sh
# Fija el volumen del parlante de la placa ESP32-S3.
#
# POR QUÉ HACE FALTA: el mezclador ALSA de la placa es ajeno a los
# parámetros ROS 2 -- `voz_volumen` solo le dice a mpv con qué volumen
# reproducir, y por debajo sigue estando el mezclador del códec ES8311. Si
# ese queda en el valor de fábrica (54%/60%), la voz se oye baja por más
# que mpv esté al 100.
#
# Y no se arregla con `alsactl store`: la placa es USB y no queda guardada
# en /var/lib/alsa/asound.state, así que cada arranque vuelve al valor de
# fábrica. De ahí este script.
#
# Para que se aplique solo en cada arranque, instalarlo como servicio:
#
#   sudo cp volumen_placa.sh /usr/local/bin/
#   sudo tee /etc/systemd/system/romeo-volumen.service > /dev/null <<'EOF'
#   [Unit]
#   Description=Volumen del parlante de Romeo (placa ESP32-S3)
#   After=sound.target
#   [Service]
#   Type=oneshot
#   RemainAfterExit=yes
#   ExecStart=/usr/local/bin/volumen_placa.sh
#   [Install]
#   WantedBy=multi-user.target
#   EOF
#   sudo systemctl enable --now romeo-volumen.service
#
# Uso a mano:  ./volumen_placa.sh [porcentaje]   (por defecto 50)
set -e

VOL="${1:-50}"

# -c Board, NO -c <número>: el índice de la placa cambia entre arranques
# (se la ha visto como card 1 y como card 3, porque es USB y el orden
# depende de cuándo se enumere). Con el número equivocado, amixer ajusta
# otra tarjeta -- por ejemplo vc4hdmi1 -- sin dar ningún error.
CARD=Board

# La placa expone DOS controles PCM (,0 y ,1) y hay que subir los dos:
# tocar uno solo no cambia nada audible. El rango del hardware es 0-50, así
# que 20% ≈ -40 dB y 100% = 0 dB.
#
# Se reintenta porque este script puede correr antes de que el USB acabe de
# enumerar la placa.
i=1
while [ "$i" -le 5 ]; do
    if amixer -c "$CARD" sset 'PCM',0 "${VOL}%" unmute >/dev/null 2>&1 &&
       amixer -c "$CARD" sset 'PCM',1 "${VOL}%" unmute >/dev/null 2>&1; then
        echo "[volumen] placa $CARD al ${VOL}% (PCM,0 y PCM,1)"
        exit 0
    fi
    echo "[volumen] la placa $CARD todavía no responde, reintento $i/5..."
    i=$((i + 1))
    sleep 2
done

echo "[volumen] no se pudo ajustar: ¿está conectada la placa? (aplay -l)" >&2
exit 1
