#!/usr/bin/env python3
"""Exporta el modelo SER de HuggingFace a ONNX. Se corre UNA vez en la PC
(necesita torch, transformers y onnx); despues se copia a la Pi:

    python exportar_ser.py
    scp ser.onnx ser.json pi@IP_DE_LA_PI:.romeo/modelos/ser/

Comprueba al final que ONNX y torch dan lo mismo (medido: 1.7e-5 de
diferencia maxima en los logits).
"""

import json

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODELO = "somosnlp-hackathon-2022/wav2vec2-base-finetuned-sentiment-mesd"

fe = AutoFeatureExtractor.from_pretrained(MODELO)
model = AutoModelForAudioClassification.from_pretrained(MODELO).eval()

torch.onnx.export(model, (torch.randn(1, 32000),), "ser.onnx",
                  input_names=["input_values"], output_names=["logits"],
                  dynamic_axes={"input_values": {1: "n"}}, opset_version=17, dynamo=False)
with open("ser.json", "w", encoding="utf-8") as f:
    json.dump({"id2label": model.config.id2label, "do_normalize": bool(fe.do_normalize),
               "sampling_rate": fe.sampling_rate}, f, ensure_ascii=False, indent=1)

audio = np.random.default_rng(0).normal(0, 0.05, 48000).astype("float32")
iv = fe(audio, sampling_rate=16000, return_tensors="np")["input_values"]
with torch.no_grad():
    lt = model(torch.from_numpy(iv)).logits.numpy()
lo = ort.InferenceSession("ser.onnx").run(None, {"input_values": iv})[0]
print(f"listo: ser.onnx + ser.json, diferencia torch/onnx {np.abs(lt - lo).max():.1e}")
