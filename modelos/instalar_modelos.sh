#!/usr/bin/env bash
# Instala los modelos que no vienen con el codigo (TER, SER, STT, VAD y voz)
# en las carpetas donde los busca Romeo. Se corre en la Pi y en la PC
# despues de cada `git pull`:
#
#     bash modelos/instalar_modelos.sh
#
# Los .onnx y .tar.bz2 de esta carpeta van por Git LFS (GitHub no acepta
# archivos de mas de 100 MB). Sin `git lfs` instalado el clone trae solo
# punteros de texto en su lugar; el sha256 de abajo lo detecta y avisa.
#
# Solo copia lo que cambio: correrlo dos veces no hace nada la segunda.
#
# Destinos (los mismos que esperan _stt_engine.py, ter.py, _ser_engine.py
# y romeo_params.yaml):
#     ter/   -> ~/.romeo/modelos/ter/
#     ser/   -> ~/.romeo/modelos/ser/
#     asr/   -> ~/.romeo/modelos/     (whisper-tiny se descomprime ahi)
#     voz/   -> ~/.romeo/modelos/     (vits-piper, para sherpa-onnx)
#               ~/piper-voces/        (es_MX-claude-high.onnx, para piper)
set -euo pipefail

AQUI="$(cd "$(dirname "$0")" && pwd)"
DESTINO="${ROMEO_MODELOS_DIR:-$HOME/.romeo/modelos}"
PIPER="${ROMEO_PIPER_DIR:-$HOME/piper-voces}"
cd "$AQUI"

# 1. Comprobar que nada llego roto (o como puntero de LFS).
if ! sha256sum --quiet -c SHA256SUMS; then
    echo "[modelos] sha256 no coincide. Si faltaba git-lfs:"
    echo "    sudo apt install git-lfs && git lfs install && git lfs pull"
    exit 1
fi
echo "[modelos] sha256 correcto"

# 2. Copiar solo lo que cambio.
copiar() {  # copiar ORIGEN DESTINO
    if ! cmp -s "$1" "$2" 2>/dev/null; then
        mkdir -p "$(dirname "$2")"
        cp "$1" "$2"
        echo "[modelos] actualizado $2"
    fi
}
for f in ter/* ser/*; do
    copiar "$f" "$DESTINO/$f"
done
copiar asr/silero_vad.onnx "$DESTINO/silero_vad.onnx"
copiar voz/es_MX-claude-high.onnx "$PIPER/es_MX-claude-high.onnx"
copiar voz/es_MX-claude-high.onnx.json "$PIPER/es_MX-claude-high.onnx.json"

# Los .tar.bz2 se descomprimen solo si cambiaron: se guarda su sha256 en
# un archivo oculto al lado para comparar la proxima vez.
for tar in asr/*.tar.bz2 voz/*.tar.bz2; do
    nombre="$(basename "$tar")"
    suma="$(sha256sum "$tar" | cut -d' ' -f1)"
    if [ "$(cat "$DESTINO/.$nombre.sha256" 2>/dev/null)" != "$suma" ]; then
        mkdir -p "$DESTINO"
        tar xjf "$tar" -C "$DESTINO"
        echo "$suma" > "$DESTINO/.$nombre.sha256"
        echo "[modelos] descomprimido $nombre"
    fi
done

echo "[modelos] listo: $DESTINO y $PIPER"
