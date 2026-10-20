# -*- coding: utf-8 -*-
"""agente_chat -- la conversacion libre.

Es el unico de los dos agentes que usa un LLM: le pasa el mensaje del
usuario a CHAT_MODEL (por Ollama) y devuelve lo que responda.

Sin system prompt a proposito: el comportamiento viene horneado en el
fine-tuning del modelo.

MANTIENE EL HILO. Antes se mandaba un unico mensaje por turno, sin nada de
lo anterior, asi que el modelo no podia resolver referencias:

    Usuario: me gustan los robots que hablan
    Romeo:   ¿Te gustan mas los que parecen humanos o los robóticos?
    Usuario: los robóticos
    Romeo:   (sin contexto, no sabia de que se hablaba)

Ahora se envian los ultimos TURNOS_HISTORIAL turnos. El limite existe
porque cada turno son tokens que el modelo tiene que leer: con un 1B en
una Pi, un historial largo cuesta latencia y acaba desbordando la ventana.

El historial es de la SESION: se borra con reiniciar(), que el orquestador
llama al empezar una sesion nueva. Para recuperar cosas de conversaciones
pasadas haria falta memoria episodica, que es otra cosa y no esta.
"""
import os

from . import llama_client

# Cuantos turnos (par usuario+Romeo) se recuerdan. 6 son ~12 mensajes, que
# con respuestas cortas caben de sobra en la ventana de un modelo 1B.
TURNOS_HISTORIAL = int(os.environ.get("ROMEO_TURNOS_HISTORIAL", "6"))

_historial = []


def reiniciar():
    """Olvida la conversacion. Se llama al arrancar una sesion nueva: el
    usuario siguiente no deberia heredar el hilo del anterior."""
    _historial.clear()


def responder(mensaje_usuario):
    """Devuelve la respuesta de Romeo, teniendo en cuenta los turnos
    anteriores de esta sesion."""
    mensajes = _historial + [{"role": "user", "content": mensaje_usuario}]
    texto = llama_client.generar_respuesta(mensajes).strip()

    _historial.append({"role": "user", "content": mensaje_usuario})
    _historial.append({"role": "assistant", "content": texto})
    # Se recorta por el principio, de dos en dos, para no dejar un turno a
    # medias (un 'user' sin su 'assistant' confunde al modelo).
    del _historial[:max(0, len(_historial) - TURNOS_HISTORIAL * 2)]
    return texto
