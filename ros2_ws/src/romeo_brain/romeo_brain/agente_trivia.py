# -*- coding: utf-8 -*-
"""agente_trivia -- el juego de preguntas y la expresion (cara, musica).

juzgar hace DOS llamadas. La primera solo decide SI o NO: se le da la
pregunta, la respuesta esperada y lo que dijo la persona. La segunda, con
el veredicto ya decidido, escribe el comentario como lo diria una persona:
celebra o corrige, de vez en cuando cuenta un dato breve de la respuesta
(PROB_DATO) y pasa a la siguiente ("No, era el emu, el ave mas veloz de
Australia. Vamos con la siguiente."). Separadas para que el comentario no
pueda contradecir al
veredicto. Si la segunda falla, se usan las frases de
data/frases/respuestas_trivia_*.txt.

ANTES el LLM juzgaba y comentaba en UNA llamada, con ejemplos y formato de
dos lineas. Medido con gemma3:4b sobre 8 respuestas, daba por buenas
"el de Madonna" (Thriller) y hasta "no se", y comentaba con apodos
("¡Inténtalo de nuevo, pequeño!"). Con la pregunta corta de SI/NO acierta
7 de 8 en ~2 s (llama3.1:8b, 8 de 8 en ~12 s).

La expresion (que cara poner, que musica sonar) sale de los campos de la
propia pregunta en preguntas.jsonl.
"""

import os
import random

from . import aeb, frases, llama_client

# Modelo del LLM que juzga las respuestas de trivia.
CORRECTOR_LLM = os.environ.get("TRIVIA_CORRECTOR_MODEL", "romeo")

# TODO VA EN EL MENSAJE DE USUARIO, sin turno system: Gemma no lo procesa
# aparte (su plantilla lo mete dentro del primer mensaje) y con system
# separado llegaba a decir SI a "Peru" cuando la respuesta era China.
_PROMPT_JUEZ = (
    "Pregunta de trivia: {pregunta}\n"
    "Respuesta correcta: {esperada}\n"
    "La persona respondio: {usuario}\n\n"
    "¿La respuesta de la persona es la respuesta correcta (aunque la diga con "
    "otras palabras o con faltas)? Si dice otra cosa, o no sabe, es NO. "
    "Contesta solo SI o NO.")

_PROMPT_COMENTARIO = (
    "Eres Romeo, un robot que juega trivia en voz alta con una persona.\n"
    "Pregunta: {pregunta}\n"
    "Respuesta correcta: {esperada}\n"
    "La persona respondio: {usuario}\n"
    "{resultado}\n\n"
    "Dile algo como lo diria una persona, con tus propias palabras y en "
    "{frases} cortas: {que_hacer}{dato}{tono} Termina con una frase corta para "
    "pasar a la siguiente pregunta, como 'Vamos con la siguiente', sin "
    "preguntar nada. Tutea, sin apodos como 'pequeño' o 'campeón', sin "
    "palabras con genero como 'listo' o 'lista', sin emojis. Escribe solo lo "
    "que vas a decir.")

# Cuantas veces cuenta un dato curioso. Siempre se hacia pesado (y alargaba
# cada turno a 15-20 s), y cuantos mas datos mas se inventa: gemma3:4b dijo
# que 54 es capicua y que "Mir" significa "espejo".
PROB_DATO = float(os.environ.get("ROMEO_TRIVIA_DATO", "0.3"))


def _comentar(pregunta, esperada, respuesta_usuario, acerto, comportamiento=None):
    """El comentario del veredicto, ya decidido, dicho como una persona.

    `comportamiento` es el AEB del momento (aeb.mapear): con el, la
    reaccion cambia segun el animo de Romeo (aeb.tono_trivia). None o
    neutral = como siempre."""
    tono = aeb.tono_trivia(comportamiento, acerto)
    if acerto:
        resultado, que_hacer = "ACERTO.", "celebra que acerto."
    else:
        resultado, que_hacer = ("SE EQUIVOCO.", "dile con amabilidad que no "
                                "era y cual era la respuesta correcta.")
    con_dato = random.random() < PROB_DATO
    dato = (" Agrega un dato curioso breve sobre la respuesta correcta, solo "
            "si estas seguro de que es cierto." if con_dato else "")
    try:
        texto = llama_client.generar_respuesta(
            [{"role": "user",
              "content": _PROMPT_COMENTARIO.format(
                  pregunta=pregunta, esperada=esperada, usuario=respuesta_usuario,
                  resultado=resultado, que_hacer=que_hacer, dato=dato,
                  tono=(f" {tono} No digas como se siente la persona ni como "
                        f"te sientes tu: que se note solo en como lo dices.") if tono else "",
                  # Con tono hace falta una frase mas: la de animo no cabia en dos.
                  frases="tres frases" if (con_dato or tono) else "dos frases")}],
            temperature=0.7, max_tokens=(90 if con_dato else 50) + (25 if tono else 0),
            modelo=CORRECTOR_LLM).strip()
        texto = " ".join(texto.split())   # sin los saltos de linea del modelo
    except Exception:  # noqa: BLE001 - sin comentario del LLM, frase fija
        texto = ""
    return texto or frases.una(
        frases.RESPUESTAS_TRIVIA_ACIERTO if acerto else frases.RESPUESTAS_TRIVIA_ERROR,
        esperada=esperada)


def juzgar(pregunta, esperada, respuesta_usuario, comportamiento=None):
    """(acerto, comentario). `comportamiento`: ver _comentar.

    El LLM compara lo que dijo la persona con la respuesta esperada. Se le
    pasa tambien la PREGUNTA, no solo la respuesta: sin ella no puede
    juzgar "de cuero" por "piel de animales"."""
    texto = llama_client.generar_respuesta(
        [{"role": "user",
          "content": _PROMPT_JUEZ.format(pregunta=pregunta, esperada=esperada,
                                         usuario=respuesta_usuario)}],
        temperature=0.0, modelo=CORRECTOR_LLM)
    acerto = texto.strip().upper().startswith(("SI", "SÍ"))
    comentario = _comentar(pregunta, esperada, respuesta_usuario, acerto, comportamiento)
    print(f"[trivia] {CORRECTOR_LLM}: esperada={esperada!r} "
          f"usuario={respuesta_usuario!r} -> "
          f"{'acierto' if acerto else 'fallo'}")
    return acerto, comentario


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
