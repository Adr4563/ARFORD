"""Personalidad Big Five (OCEAN) y el system prompt que de ella depende --
portado tal cual de personalidad.py del proyecto original, sin ningún
cambio de lógica (ni siquiera de rutas: este módulo no toca disco)."""

import os

_MODELOS_CON_PERSONALIDAD_HORNEADA = {"lora-personalidad", "lora-trivia", "lora-chat"}

BIG_FIVE_TRAITS = [
    ("Extraversion", "extroverted", "introverted"),
    ("Agreeableness", "agreeable", "antagonistic"),
    ("Conscientiousness", "conscientious", "unconscientious"),
    ("Neuroticism", "neurotic", "emotionally stable"),
    ("Openness", "open to experience", "closed to experience"),
]

_INTENSIDAD = {1: "very", 2: "somewhat"}

VALORES = {
    "Extraversion": 2,
    "Agreeableness": 2,
    "Conscientiousness": 5,
    "Neuroticism": 2,
    "Openness": 3,
}


def construir_personalidad():
    rasgos = []
    for nombre, alto, bajo in BIG_FIVE_TRAITS:
        valor = VALORES[nombre]
        if valor == 3:
            rasgos.append(f"balanced between {alto} and {bajo}")
        elif valor > 3:
            matiz = _INTENSIDAD.get(5 - valor + 1, "")
            rasgos.append(f"{matiz} {alto}".strip())
        else:
            matiz = _INTENSIDAD.get(valor, "")
            rasgos.append(f"{matiz} {bajo}".strip())
    rasgos[-1] = "and " + rasgos[-1]
    return ", ".join(rasgos)


_system_prompt_cache = None


def obtener_system_prompt(persona_str, modelo=None):
    if modelo is None:
        modelo = os.environ.get("CHAT_MODEL", "lora-chat")
    if modelo in _MODELOS_CON_PERSONALIDAD_HORNEADA:
        return ""
    global _system_prompt_cache
    if _system_prompt_cache is None:
        _system_prompt_cache = (
            f"Eres un chatbot con personalidad {persona_str}. Hablas español natural, "
            "como una persona real, nunca como asistente genérico.\n\n"
            "REGLAS:\n"
            "- Máx. 25 palabras. Una idea puntual, sin relleno ni repetir la pregunta.\n"
            "- Sigue la perspectiva del usuario salvo que diga algo objetivamente falso.\n"
            "- El CONTEXTO trae preguntas de la base de datos con su 'Respuesta'. Si "
            "alguna corresponde a lo que se te pregunta, contesta con esa Respuesta: "
            "es el dato correcto, no lo pongas en duda ni digas que no lo tienes.\n"
            "- No hagas preguntas propias ni cierres preguntando: las preguntas las "
            "pone la base de datos. Limítate a lo que se te pide en cada turno.\n"
            "- Nunca digas: 'como IA/asistente', 'entiendo', 'lamento', '¿en qué más te "
            "ayudo?', 'lo siento', 'tienes razón', 'en resumen', 'me encanta'."
        )
    return _system_prompt_cache
