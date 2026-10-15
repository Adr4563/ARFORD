"""Busca la respuesta de terapeuta mas parecida a lo que escribes.

    python bert.py "me siento solo"      # una consulta
    python bert.py                       # modo interactivo
"""
import json
import sys
import warnings

warnings.filterwarnings("ignore")
from sentence_transformers import SentenceTransformer, util

MODELO = "adapters/arford-bert-terapia"
PARES = "data/pares_terapia.jsonl"

m = SentenceTransformer(MODELO)
respuestas = sorted({json.loads(l)["terapeuta"] for l in open(PARES, encoding="utf-8")})
emb = m.encode(respuestas, convert_to_tensor=True, batch_size=64, show_progress_bar=False)


def buscar(texto):
    e = m.encode(texto, convert_to_tensor=True)
    for i, h in enumerate(util.semantic_search(e, emb, top_k=3)[0], 1):
        print(f"  {i}. ({h['score']:.3f}) {respuestas[h['corpus_id']]}")


if len(sys.argv) > 1:
    buscar(" ".join(sys.argv[1:]))
else:
    print(f"BERT TerapIA listo ({len(respuestas)} respuestas). Ctrl+C para salir.\n")
    while True:
        try:
            q = input("tu> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q:
            buscar(q)
            print()
