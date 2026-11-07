# -*- coding: utf-8 -*-
"""seguridad -- frases que el chat no puede contestar con el tono de siempre.

Los tonos del AEB (aeb.py) piden animar si la persona esta triste y darle
la razon si esta enojada. Para un mal dia esta bien, pero medido con
gemma2:2b cuando los tonos llevaban frases fijas:

    "Estoy muy triste, ya no quiero vivir"
        -> "No te preocupes, esto va a pasar. Tienes que ser fuerte."
    "Estoy triste porque mi papa me grita y me pega"
        -> "No te preocupes, esto va a pasar. Tienes que ser fuerte."
    "Estoy furioso con mi hermano, le voy a pegar"
        -> "Tienes toda la razon, yo tambien estaria enojado."

Le quitan importancia a algo grave, o aprueban hacerle daño a alguien.
Esto detecta esos casos en el mensaje y devuelve una instruccion que
SUSTITUYE al tono de ese turno (ver agente_chat.responder), con cualquier
animo de Romeo.

Va por palabras y no por el TER porque el TER dice "tristeza", no por que:
"ya no quiero vivir" y "perdi el partido" son la misma emocion.

OJO: no reemplaza a una persona. Romeo acompaña y anima a hablar con
alguien de confianza; no menciona lineas de ayuda, por decision del
equipo. Que alguien de psicologia revise estos textos.
"""
import re

# (nombre, patron, que lograr). Se mira en orden y gana el primero: hacerse
# daño va antes que todo lo demas.
#
# Dice QUE lograr, no que frase decir: Romeo responde con sus palabras.
# Con frases de ejemplo el modelo las leia tal cual. La pregunta va AL
# FINAL porque gemma2:2b deja de escribir en cuanto pregunta algo, y lo de
# "alguien de confianza" se saltaba si venia despues.
_CASOS = (
    ("se_hace_dano",
     re.compile(
         r"no quiero (seguir )?vivir|quiero morir|me quiero morir|matarme|suicid|"
         r"quitarme la vida|hacerme da[ñn]o|lastimarme|cortarme|"
         r"no vale la pena vivir|mejor (estar|estaria) muert", re.IGNORECASE),
     "La persona dijo algo que indica que podria hacerse daño. Es lo mas "
     "importante de esta conversacion: tu estas ahi para escucharla. NO le "
     "quites importancia ('no te preocupes', 'esto va a pasar'). Con tus "
     "propias palabras y con calidez: hazle sentir que la escuchas y dile que "
     "no tiene que pasarlo a solas. Es OBLIGATORIO que tambien le pidas que "
     "hable con alguien de confianza que pueda acompañarla (familia, un "
     "amigo, un profesor). Termina con una pregunta para que te cuente que "
     "le pasa."),
    ("la_lastiman",
     re.compile(
         r"me (pega|pegan|golpea|golpean|maltrata|maltratan|lastima|lastiman|"
         r"toca|tocan|amenaza|amenazan)\b|abusa(n)? de mi|abusaron de mi",
         re.IGNORECASE),
     "La persona cuenta que alguien la lastima: tu estas ahi para apoyarla. "
     "NO le quites importancia ('no te preocupes', 'esto va a pasar'). Con "
     "tus propias palabras y con calidez: dile que no es su culpa, que nadie "
     "tiene derecho a lastimarla y que la escuchas. Es OBLIGATORIO que "
     "tambien le pidas que hable con alguien de confianza que la pueda "
     "cuidar (otro familiar, un profesor, un amigo). Termina con una "
     "pregunta para que te cuente."),
    ("dana_a_otro",
     re.compile(
         r"\b(le|les|lo|la|los|las) (voy a|vamos a|quiero) "
         r"(pegar|golpear|matar|lastimar|hacer da[ñn]o)|"
         r"(voy a|quiero) (pegarle|golpearle|matarlo|matarla|lastimarlo|"
         r"lastimarla|hacerle da[ñn]o)", re.IGNORECASE),
     "La persona dice que va a lastimar a alguien. NO le des la razon en "
     "hacerle daño a nadie, pero tu estas ahi para apoyarla. Con tus propias "
     "palabras: muestra que entiendes su enojo, dile que lastimar no lo va a "
     "arreglar y ayudala a calmarse antes de hacer algo. Termina con una "
     "pregunta para que te cuente que paso."),
)


def revisar(mensaje):
    """(nombre, instruccion) del primer caso que aparece en `mensaje`, o
    (None, None) si no hay ninguno."""
    for nombre, patron, instruccion in _CASOS:
        if mensaje and patron.search(mensaje):
            return nombre, instruccion
    return None, None
