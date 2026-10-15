"""Afina BERT espanol (BETO) con los pares usuario->terapeuta de TerapIA.

Aprende a acercar en el espacio vectorial lo que dice el usuario y la
respuesta que le dio el terapeuta. Sirve para recuperacion (RAG).

    python 5_bert_embeddings.py
"""
import json
import random

from sentence_transformers import InputExample, SentenceTransformer, losses, models
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader

BASE = "hiiamsid/sentence_similarity_spanish_es"   # BETO (BERT espanol) en safetensors
SALIDA = "adapters/arford-bert-terapia"
DATOS = "data/pares_terapia.jsonl"
EPOCAS = 2

pares = [json.loads(l) for l in open(DATOS, encoding="utf-8")]
random.seed(42)
random.shuffle(pares)
corte = int(len(pares) * 0.95)
train, test = pares[:corte], pares[corte:]
print(f"Train: {len(train)} pares  |  Test: {len(test)} pares")

# --- BERT + pooling de media = modelo de embeddings ---
word = models.Transformer(BASE, max_seq_length=256)
pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode="mean")
model = SentenceTransformer(modules=[word, pool])

# --- MultipleNegativesRanking: el resto del lote son negativos ---
ejemplos = [InputExample(texts=[p["usuario"], p["terapeuta"]]) for p in train]
loader = DataLoader(ejemplos, shuffle=True, batch_size=32, drop_last=True)
loss = losses.MultipleNegativesRankingLoss(model)

# --- Evaluador: dado un mensaje, encuentra la respuesta correcta? ---
consultas = {str(i): p["usuario"] for i, p in enumerate(test)}
corpus = {str(i): p["terapeuta"] for i, p in enumerate(test)}
relevantes = {str(i): {str(i)} for i in range(len(test))}
ev = InformationRetrievalEvaluator(consultas, corpus, relevantes, name="terapia",
                                   accuracy_at_k=[1, 3, 5], ndcg_at_k=[5], map_at_k=[5],
                                   precision_recall_at_k=[1, 5], show_progress_bar=False)

print("\n--- BERT base, sin entrenar ---")
antes = ev(model)
print(f"Acc@1: {antes['terapia_cosine_accuracy@1']:.3f}  NDCG@5: {antes['terapia_cosine_ndcg@5']:.3f}")

model.fit(train_objectives=[(loader, loss)], epochs=EPOCAS, warmup_steps=int(len(loader) * 0.1),
          output_path=SALIDA, show_progress_bar=True)

print("\n--- BERT afinado con TerapIA ---")
despues = ev(model)
print(f"Acc@1: {despues['terapia_cosine_accuracy@1']:.3f}  NDCG@5: {despues['terapia_cosine_ndcg@5']:.3f}")

with open(f"{SALIDA}/metricas.json", "w") as f:
    json.dump({"antes": {k: float(v) for k, v in antes.items()},
               "despues": {k: float(v) for k, v in despues.items()}}, f, indent=2)

print(f"\nModelo: {SALIDA}")
print(f"Acc@1: {antes['terapia_cosine_accuracy@1']:.3f} -> {despues['terapia_cosine_accuracy@1']:.3f}")
