#!/usr/bin/env python3
"""Exporta el TER en espanol a ONNX. Se corre UNA vez en la PC (torch,
transformers y onnx) y se copia a la Pi:

    python exportar_ter.py
    scp ter.onnx ter.json tokenizer.json tokenizer_config.json pi@IP_DE_LA_PI:.romeo/modelos/ter/

Medido en la PC: ONNX y torch difieren en 4e-6 como mucho.
"""

import json

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODELO = "pysentimiento/robertuito-emotion-analysis"

tok = AutoTokenizer.from_pretrained(MODELO)
model = AutoModelForSequenceClassification.from_pretrained(MODELO).eval()
tok.save_pretrained(".")
enc = tok(["hola"], return_tensors="pt")
torch.onnx.export(model, (enc["input_ids"], enc["attention_mask"]), "ter.onnx",
                  input_names=["input_ids", "attention_mask"], output_names=["logits"],
                  dynamic_axes={"input_ids": {0: "b", 1: "n"}, "attention_mask": {0: "b", 1: "n"}},
                  opset_version=17, dynamo=False)
with open("ter.json", "w", encoding="utf-8") as f:
    json.dump({"id2label": model.config.id2label}, f, indent=1)

e = tok(["hoy me siento muy triste"], return_tensors="np")
lo = ort.InferenceSession("ter.onnx").run(
    None, {"input_ids": e["input_ids"], "attention_mask": e["attention_mask"]})[0]
with torch.no_grad():
    lt = model(**tok(["hoy me siento muy triste"], return_tensors="pt")).logits.numpy()
print(f"listo, diferencia torch/onnx {np.abs(lo - lt).max():.1e}")
