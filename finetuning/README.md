# Fine-tuning QLoRA — diálogo empático en español

Método del libro *A Hands-On Guide to Fine-Tuning LLMs* (Godoy, 2025), aplicado a ARFORD.

## Fuentes

| | |
|---|---|
| **Paper del dataset** | Rashkin et al., [*Towards Empathetic Open-domain Conversation Models*](https://arxiv.org/abs/1811.00207), ACL 2019 |
| **Dataset** | [Estwld/empathetic_dialogues_llm](https://huggingface.co/datasets/Estwld/empathetic_dialogues_llm) — 19,533 conversaciones, etiquetadas por emoción · [repo original](https://github.com/facebookresearch/EmpatheticDialogues) |
| **Método** | [LoRA](https://arxiv.org/abs/2106.09685) · [QLoRA](https://arxiv.org/abs/2305.14314) · [Attention](https://arxiv.org/abs/1706.03762) |
| **Código del libro** | https://github.com/dvgodoy/FineTuningLLMs |

### Papers de contexto (robots asistivos + LLM)

- [Human-mediated LLMs for Robotic Intervention in Children with ASD](https://arxiv.org/html/2402.00260)
- [Emotion Co-regulation with LLM-powered Socially Assistive Robots](https://arxiv.org/pdf/2507.10427) (2025)
- [ASD-Chat: Dialogue Intervention System based on LLM and VB-MAPP](https://arxiv.org/pdf/2409.01867)
- [Systematic Review: AR/VR, AI y Robótica para TEA](https://arxiv.org/pdf/2409.18162)

Ninguno publica su dataset (son datos clínicos con menores). Sirven como método y justificación.

## Pasos

```bash
python 1_traducir_dataset.py 19533 24   # EN -> ES, 24 hilos (~3h)
python 2_finetune.py                     # QLoRA sobre Qwen2.5-0.5B
jupyter notebook 3_metricas.ipynb        # curvas de loss y pruebas
```

## Configuración

| | |
|---|---|
| Cuantización | 4-bit NF4 + double quant |
| LoRA | r=16, alpha=32, dropout=0.05 |
| Entrenamiento | 3 épocas, lr=2e-4 cosine, batch efectivo 16 |
| Memoria | gradient checkpointing + accumulation |
| GPU | RTX A5000 24GB |

Otro modelo base: `python 2_finetune.py Qwen/Qwen2.5-1.5B-Instruct`
