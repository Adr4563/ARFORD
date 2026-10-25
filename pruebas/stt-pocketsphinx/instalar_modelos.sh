#!/usr/bin/env bash
#
# Instala pocketsphinx (apt) y baja los dos modelos de espanol a
# ~/.romeo/modelos/sphinx/. Se puede correr varias veces: lo que ya esta
# bajado no se vuelve a bajar.
#
#   bash instalar_modelos.sh
#
# Origen: el SourceForge oficial de CMU Sphinx.
#   cmu-es     cmusphinx-es-5.2.tar.gz (33 MB) + es-20k.lm.gz (23.5 MB) + es.dict
#   ciempiess  CIEMPIESS_Spanish_Models_581h.zip (160 MB)

set -eo pipefail

D="${ROMEO_SPHINX_DIR:-$HOME/.romeo/modelos/sphinx}"
B="https://downloads.sourceforge.net/project/cmusphinx/Acoustic%20and%20Language%20Models"
MD5_CIEMPIESS="8c61026849014416d34c945ffea45edc"

if ! command -v pocketsphinx >/dev/null; then
    echo "== instalando pocketsphinx (apt)"
    sudo apt install -y pocketsphinx
fi
command -v arecord >/dev/null || sudo apt install -y alsa-utils

# --- CMU Sphinx espanol ---------------------------------------------------
mkdir -p "$D/cmu-es" && cd "$D/cmu-es"
if [ ! -d cmusphinx-es-5.2 ]; then
    echo "== bajando cmusphinx-es-5.2"
    curl -fSL -o es.tar.gz "$B/Spanish/cmusphinx-es-5.2.tar.gz"
    tar xzf es.tar.gz && rm es.tar.gz
fi
[ -f es.dict ] || curl -fSL -o es.dict "$B/Spanish/es.dict"
if [ ! -f es-20k.lm ]; then
    [ -f es-20k.lm.gz ] || curl -fSL -o es-20k.lm.gz "$B/Spanish/es-20k.lm.gz"
    # pocketsphinx 5 no lee el .gz: sin descomprimir arranca SIN modelo de
    # lenguaje y solo lo avisa en stderr.
    gunzip -k es-20k.lm.gz
fi
# El feat.params de este paquete salio con las plantillas sin rellenar
# (-nfilt __CFG_NUM_FILT__ ...). Los valores reales son los de
# etc/sphinx_train.cfg del mismo paquete.
H="cmusphinx-es-5.2/model_parameters/voxforge_es_sphinx.cd_ptm_4000"
if grep -q __CFG "$H/feat.params"; then
    printf '%s\n' "-lowerf 130" "-upperf 6800" "-nfilt 25" "-transform dct" \
        "-lifter 22" "-feat 1s_c_d_dd" "-svspec 0-12/13-25/26-38" \
        "-agc none" "-cmn current" "-varnorm no" > "$H/feat.params"
    echo "== feat.params de cmu-es corregido"
fi

# --- CIEMPIESS (UNAM) ------------------------------------------------------
mkdir -p "$D/ciempiess" && cd "$D/ciempiess"
if [ ! -d CIEMPIESS_Spanish_Models_581h ]; then
    echo "== bajando CIEMPIESS (160 MB)"
    curl -fSL -o c.zip "$B/Mexican%20Spanish/CIEMPIESS_Spanish_Models_581h.zip"
    echo "$MD5_CIEMPIESS  c.zip" | md5sum -c -
    # Sin `unzip`: no viene en la imagen de la Pi y python3 ya esta.
    python3 -m zipfile -e c.zip . && rm c.zip
fi

echo
echo "Listo:"
du -sh "$D/cmu-es" "$D/ciempiess"
