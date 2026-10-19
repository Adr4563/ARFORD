# -*- coding: utf-8 -*-
"""agente_chat -- la conversacion libre.

Es el unico de los dos agentes que usa un LLM: le pasa el mensaje del
usuario a CHAT_MODEL (arford-empatico-full por Ollama) y devuelve lo que
responda.

Sin system prompt a proposito: el comportamiento empatico viene horneado en
el fine-tuning (320 conversaciones de ALIA/TerapIA, ver finetuning/). Y sin
memoria: antes se pegaba delante un recuerdo de memoria_episodica, que
buscaba por palabras compartidas y era compartida entre todos los usuarios.
"""
from . import llama_client


def responder(mensaje_usuario):
    """Devuelve la respuesta de Romeo al mensaje."""
    mensajes = [{"role": "user", "content": mensaje_usuario}]
    return llama_client.generar_respuesta(mensajes).strip()
