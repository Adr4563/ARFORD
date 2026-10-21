import torch
from transformers import AutoTokenizer, AutoModel

# 1. Cargamos el modelo ligero de dccuchile
MODEL_NAME = "dccuchile/albeto-base"
print("Descargando el modelo de Hugging Face (solo la primera vez)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# 2. Base de datos con respuestas cortas y conversacionales
respuestas_ia = [
    "¡Hola! Pues todo muy bien por aquí, ¿tú qué tal estás?",
    "Uf, lamento escuchar eso. ¿Quieres hablar de otra cosa para despejar la mente?",
    "¡Qué buena noticia! Cuéntame más sobre eso.",
    "A veces los días son complicados. ¿Qué planes tienes para distraerte hoy?",
    "¡Totalmente de acuerdo! Por cierto, ¿viste lo que pasó ayer?"
]

def obtener_vector(texto):
    """Convierte texto en un vector matemático usando el modelo NLP."""
    inputs = tokenizer(texto, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

print("Procesando base de datos de respuestas...")
vectores_respuestas = torch.cat([obtener_vector(r) for r in respuestas_ia])

print("\n🤖 ¡Chatbot listo en tu PC! Escribe algo para comenzar...")

# 3. Bucle interactivo de chat
while True:
    entrada_usuario = input("\nTú: ")
    if entrada_usuario.lower() in ["salir", "exit", "adiós", "adios"]:
        print("🤖 ¡Nos vemos pronto!")
        break
        
    if not entrada_usuario.strip():
        continue

    vector_usuario = obtener_vector(entrada_usuario)
    similitudes = torch.nn.functional.cosine_similarity(vector_usuario, vectores_respuestas)
    mejor_opcion_idx = torch.argmax(similitudes).item()
    
    print(f"IA: {respuestas_ia[mejor_opcion_idx]}")
