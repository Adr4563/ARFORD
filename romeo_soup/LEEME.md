# Romeo empático con Soup

## Opción A: en tu PC con GPU NVIDIA (Windows)

```powershell
cd romeo_soup
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install "soup-cli[train]"
soup doctor                                   # debe detectar la GPU
soup train --config soup.yaml --dry-run       # valida config y datos
soup train --config soup.yaml                 # ~10 min
soup chat --model ./output/romeo-empatico     # probarlo
soup export --model ./output/romeo-empatico --format gguf --quant q8_0 -o romeo-empatico.q8_0.gguf
ollama create romeo-empatico -f Modelfile
```

## Opción B: Google Colab (GPU T4 gratis)

Abre `entrenar_con_soup_colab.ipynb` y ejecuta las celdas en orden.

## Usarlo en Romeo

`ros2 launch romeo_bringup romeo_bringup.launch.py chat_model:=romeo-empatico`

El system prompt va en el `Modelfile`; como el nombre empieza por `romeo`, `agente_chat.py` no manda uno propio.
