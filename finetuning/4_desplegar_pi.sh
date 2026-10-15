#!/usr/bin/env bash
# Convierte el adaptador a GGUF, lo importa a Ollama y lo copia a la Raspberry.
#
#   bash 4_desplegar_pi.sh
set -e

PI=pi@10.143.75.137
ADAPTER=adapters/arford-empatico-es
NOMBRE=arford-empatico
BASE=llama3.2:1b

echo "=== 1. Fusionar adaptador con el modelo base ==="
python - <<'PY'
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

ruta = "adapters/arford-empatico-es"
model = AutoPeftModelForCausalLM.from_pretrained(ruta, device_map="cpu")
model = model.merge_and_unload()
model.save_pretrained("merged")
AutoTokenizer.from_pretrained(ruta).save_pretrained("merged")
print("modelo fusionado -> merged/")
PY

echo "=== 2. Convertir a GGUF ==="
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggerganov/llama.cpp
pip install -q -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
python llama.cpp/convert_hf_to_gguf.py merged --outfile ${NOMBRE}.gguf --outtype q8_0

echo "=== 3. Modelfile de Ollama ==="
cat > Modelfile <<EOF
FROM ./${NOMBRE}.gguf
PARAMETER temperature 0.7
PARAMETER stop <|eot_id|>
SYSTEM Eres ARFORD, un robot social que acompana a ninos. Escuchas con empatia, validas lo que sienten y haces preguntas abiertas. No das consejos ni diagnosticos. Respondes breve, en una o dos frases.
EOF

echo "=== 4. Copiar a la Raspberry ==="
scp ${NOMBRE}.gguf Modelfile ${PI}:~/

echo "=== 5. Importar en Ollama de la Pi ==="
ssh ${PI} "cd ~ && ollama create ${NOMBRE} -f Modelfile && ollama list | grep ${NOMBRE}"

echo
echo "Listo. Probar con:"
echo "  ssh ${PI} \"ollama run ${NOMBRE} 'Hoy me siento triste'\""
