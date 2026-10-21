import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Cargamos un modelo NLP Generativo Real Ultra-Ligero
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print("Cargando modelo NLP Generativo (Text Generation)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", device_map="auto")

# Historial para que el NLP recuerde la conversación seguida
historial_conversacion = [
    {"role": "system", "content": "Eres un amigo muy agradable en un chat casual. Hablas en español. Tus respuestas deben ser SIEMPRE muy cortas (máximo 1 o 2 frases), naturales y debes hacer comentarios o preguntas para mantener vivo el diálogo."}
]

print("\n🧠 ¡Modelo NLP Generativo listo! Conversación iniciada...")

while True:
    entrada_usuario = input("\nTú: ")
    if entrada_usuario.lower() in ["salir", "exit", "adiós", "adios"]:
        print("NLP: ¡Nos vemos luego!")
        break
        
    if not entrada_usuario.strip():
        continue

    # Agregamos lo que dijiste al historial conversacional seguido
    historial_conversacion.append({"role": "user", "content": entrada_usuario})

    # Aplicamos el formato de chat para que el NLP entienda los turnos de la plática
    texto_formateado = tokenizer.apply_chat_template(historial_conversacion, tokenize=False, add_generation_prompt=True)
    
    # Preparamos los datos para el procesador
    model_inputs = tokenizer([texto_formateado], return_tensors="pt").to(model.device)

    # El NLP genera la respuesta palabra por palabra basándose en TODO el historial previo
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=60,      # Limita a que responda corto
            do_sample=True,         # Permite respuestas creativas y variadas
            temperature=0.7,        # Controla qué tan casual habla
            top_p=0.9
        )
    
    # Cortamos los tokens de entrada para quedarnos solo con lo que el NLP inventó
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    respuesta_nlp = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Imprimimos la respuesta generada por el NLP
    print(f"NLP: {respuesta_nlp}")

    # Guardamos la respuesta de la IA en el historial para el siguiente turno
    historial_conversacion.append({"role": "assistant", "content": respuesta_nlp})
