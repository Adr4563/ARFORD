# Evaluacion de la arquitectura RAG
# pip install ollama numpy
# ollama pull nomic-embed-text

import numpy as np
import ollama

EMBED_MODEL = "nomic-embed-text"   # convierte texto en vectores (busqueda)
CHAT_MODEL = "qwen2.5:0.5b"        # genera la respuesta

# 1. Corpus
docs = [
    "ARFORD usa una Raspberry Pi 4 con 8 GB de RAM.",
    "El carrito mecanum lo controla un ESP32-S3 por serial a 115200 baudios.",
    "El watchdog del ESP32 detiene los motores si no llega comando en 500 ms.",
    "La trivia tiene 248 preguntas en 51 temas.",
    "Las emociones se detectan con onnxruntime y el modelo FER+.",
    "Solo se usan 4 emociones: Feliz, Triste, Enojado y Neutral.",
]

# 2. Embeddings del corpus
vectores = []
for d in docs:
    v = ollama.embed(model=EMBED_MODEL, input=d).embeddings[0]
    vectores.append(v)
vectores = np.array(vectores)

# 3. Preguntas de prueba: (pregunta, palabra que debe aparecer)
preguntas = [
    ("Cuanta RAM tiene la Raspberry?", "8"),
    ("Que controla los motores?", "esp32"),
    ("Cuanto tarda el watchdog?", "500"),
    ("Cuantas preguntas de trivia hay?", "248"),
    ("Cuantas emociones reconoce?", "4"),
    ("De que color es el robot?", "no tengo"),  # no esta en el corpus
]

# 4. Evaluar
aciertos = 0

for pregunta, esperado in preguntas:
    # buscar los 2 documentos mas parecidos (similitud coseno)
    q = ollama.embed(model=EMBED_MODEL, input=pregunta).embeddings[0]
    scores = vectores @ q / (np.linalg.norm(vectores, axis=1) * np.linalg.norm(q))
    top = np.argsort(-scores)[:2]
    contexto = "\n".join(docs[i] for i in top)

    # preguntar al modelo con ese contexto
    r = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "Responde en una frase usando solo el contexto. Si no esta, di: No tengo esa informacion.\n\n" + contexto},
            {"role": "user", "content": pregunta},
        ],
        options={"temperature": 0},
    )
    respuesta = r.message.content.strip()

    ok = esperado in respuesta.lower()
    aciertos += ok

    print(f"\n[{'OK' if ok else 'FALLO'}] {pregunta}")
    print(f"  {respuesta}")

print(f"\nResultado: {aciertos}/{len(preguntas)}")
