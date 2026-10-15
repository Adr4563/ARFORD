"""Fine-tuning QLoRA siguiendo el metodo del libro de Godoy.

    python 2_finetune.py                       # qwen2.5-0.5b (rapido)
    python 2_finetune.py Qwen/Qwen2.5-1.5B-Instruct
"""
import json
import sys

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MODELO = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
SALIDA = f"adapters/{MODELO.split('/')[-1].lower()}-empatico-es"
DATOS = "data/entrenamiento.jsonl"

# --- Dataset ---
# Solo 'messages': la etiqueta de emocion se descarta para el SFT
filas = [{"messages": json.loads(l)["messages"]}
         for l in open(DATOS, encoding="utf-8")]
ds = Dataset.from_list(filas).train_test_split(test_size=0.05, seed=42)
print(f"Train: {len(ds['train'])}  Eval: {len(ds['test'])}")

# --- Modelo cuantizado 4-bit (Cap. 2 del libro) ---
bf16 = torch.cuda.is_bf16_supported()
nf4 = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float32,
)
model = AutoModelForCausalLM.from_pretrained(MODELO, device_map="cuda:0", quantization_config=nf4)
tokenizer = AutoTokenizer.from_pretrained(MODELO)

# --- LoRA (Cap. 3) ---
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")

# --- Entrenamiento (Cap. 5) ---
cfg = SFTConfig(
    output_dir=SALIDA,
    max_length=512,
    packing=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=25,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none",
    bf16=bf16,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    peft_config=lora,
    args=cfg,
)

trainer.train()
trainer.save_model(SALIDA)

# --- Metricas a JSON para el notebook ---
hist = trainer.state.log_history
with open(f"{SALIDA}/metricas.json", "w") as f:
    json.dump(hist, f, indent=2)

train_loss = [h for h in hist if "loss" in h]
eval_loss = [h for h in hist if "eval_loss" in h]
print(f"\nAdaptador: {SALIDA}")
if train_loss:
    print(f"Loss train: {train_loss[0]['loss']:.3f} -> {train_loss[-1]['loss']:.3f}")
if eval_loss:
    print(f"Loss eval:  {eval_loss[0]['eval_loss']:.3f} -> {eval_loss[-1]['eval_loss']:.3f}")
