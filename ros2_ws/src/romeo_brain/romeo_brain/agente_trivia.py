# -*- coding: utf-8 -*-
"""agente_trivia -- el juego de preguntas y la expresion (cara, musica).

El LLM juzga Y comenta la respuesta del usuario en UNA sola llamada
(juzgar): se le da la pregunta, la respuesta esperada y lo que dijo la
persona, y devuelve si acerto y una frase para decirsela. No hay
comparacion de texto ni corrector aparte: el LLM compara directo.

La expresion (que cara poner, que musica sonar) sale de los campos de la
propia pregunta en preguntas.jsonl.
"""

import os

from . import llama_client

# Modelo del LLM que juzga y comenta las respuestas de trivia.
CORRECTOR_LLM = os.environ.get("TRIVIA_CORRECTOR_MODEL", "romeo")

# Los ejemplos no son adorno: sin ellos un modelo pequeño DESCRIBE la
# instruccion en vez de cumplirla o se inventa el veredicto. Con dos
# ejemplos resueltos contesta en el formato correcto.
#
# TODO VA EN EL MENSAJE DE USUARIO, sin turno system: Gemma no lo procesa
# aparte (su plantilla lo mete dentro del primer mensaje) y con system
# separado llegaba a decir SI a "Peru" cuando la respuesta era China.
_PROMPT_JUEZ = (
    "Eres Romeo, un robot que juega trivia con niños. Te doy la pregunta, "
    "la respuesta correcta y lo que dijo la persona.\n"
    "Decide si acerto (aunque lo diga con otras palabras o mal transcrito).\n"
    "Responde EXACTAMENTE en dos lineas:\n"
    "VEREDICTO: SI o NO\n"
    "COMENTARIO: una frase corta dirigida A LA PERSONA, tuteando.\n"
    "\n"
    "Ejemplo 1:\n"
    "Pregunta: ¿Capital de Francia?\n"
    "Respuesta correcta: Paris\n"
    "La persona dijo: Paris obvio\n"
    "VEREDICTO: SI\n"
    "COMENTARIO: ¡Esa te la sabias!\n"
    "\n"
    "Ejemplo 2:\n"
    "Pregunta: ¿Cuantas patas tiene una araña?\n"
    "Respuesta correcta: 8\n"
    "La persona dijo: seis creo\n"
    "VEREDICTO: NO\n"
    "COMENTARIO: Casi, pero son 8."
)


def juzgar(pregunta, esperada, respuesta_usuario):
    """(acerto, comentario) del LLM en una sola llamada.

    El LLM compara lo que dijo la persona con la respuesta esperada y
    devuelve si acerto (bool) y la frase para decirsela. Se le pasa tambien
    la PREGUNTA, no solo la respuesta: sin ella no puede juzgar "de cuero"
    por "piel de animales"."""
    texto = llama_client.generar_respuesta(
        [{"role": "user",
          "content": f"{_PROMPT_JUEZ}\n\nPregunta: {pregunta}\n"
                     f"Respuesta correcta: {esperada}\n"
                     f"La persona dijo: {respuesta_usuario}"}],
        temperature=0.1, modelo=CORRECTOR_LLM)

    acerto = comentario = None
    for linea in texto.splitlines():
        arriba = linea.upper()
        if "VEREDICTO" in arriba:
            cola = arriba.split("VEREDICTO", 1)[1]
            acerto = ("SI" in cola or "SÍ" in cola)
        elif "COMENTARIO" in arriba:
            comentario = linea.split(":", 1)[-1].strip() or None
    print(f"[trivia] {CORRECTOR_LLM}: esperada={esperada!r} "
          f"usuario={respuesta_usuario!r} -> "
          f"{'acierto' if acerto else 'fallo'}")
    return bool(acerto), comentario


# ─── Expresion: cara, musica (venia de agent_behavior.py) ───────────
# El dataset usa nombres en español (columna 'cara'); FaceCommand.face_name
# espera las claves en inglés de faces/*.mp4|*.gif. "Neutral" mapea a
# "content" porque es la misma cara de reposo -- no hay un asset de
# "neutral" aparte, igual que en el original.
_CARA_A_DISPLAY = {
    "Feliz": "happy",
    "Triste": "sad",
    "Enojado": "angry",
    "Neutral": "content",
}


def elegir_cara_pregunta(pregunta, acerto):
    """Cara para el veredicto de una pregunta CON respuesta_esperada."""
    campo = "cara_respuesta_buena" if acerto else "cara_respuesta_mala"
    valor = (pregunta.get(campo) or "").strip()
    return _CARA_A_DISPLAY.get(valor)


def cara_para_emocion(nombre):
    """Traduce Feliz/Triste/Enojado/Neutral (columna 'cara') a la clave que
    entiende FaceCommand -- usado por el Juego de emociones/imitación para
    mostrar la cara ANTES de pedirle al usuario que la imite."""
    return _CARA_A_DISPLAY.get((nombre or "").strip())


def expresar_musica(pregunta, bridge, esperar=False, logger=None):
    """Si la pregunta trae algo en 'musical', le pide al `bridge` que la
    reproduzca (PlayMusic.srv vía communication_node). Vacío/ausente -> no
    hace nada, devuelve None. `esperar=True` para Reconocimiento Musical
    (la música es el enunciado, hay que escucharla completa antes de
    responder) -- ver la nota en romeo_interfaces/srv/PlayMusic.srv."""
    musica = (pregunta.get("musical") or "").strip()
    if not musica:
        return None
    if logger is not None:
        logger.info(f"[música] {musica}")
    bridge.reproducir_musica(musica, esperar=esperar)
    return musica
