# -*- coding: utf-8 -*-
"""Parte las conversaciones de TerapIA en trozos que quepan en max_length.

EL PROBLEMA: las sesiones de TerapIA tienen 38-70 turnos (44 de media), que
son ~1700 tokens. Con max_length=1024 el entrenamiento truncaba cada una y el
modelo solo veia los primeros ~25 turnos: el 57% del contenido. Lo que se
perdia no era relleno -- los turnos finales son donde el terapeuta valida,
acompana y cierra la sesion, justo lo que al modelo le falta.

LA SOLUCION: cortar cada conversacion en trozos de turnos COMPLETOS que
quepan en el limite. Una sesion de 70 turnos da 2-3 ejemplos en vez de uno
recortado. No se inventa nada: son las mismas 320 conversaciones de
ALIA/TerapIA, solo que entera en vez de a medias.

Cada trozo empieza en un turno de usuario y termina en uno de asistente, para
que el modelo siempre aprenda a responder y nunca a preguntar por el usuario.

    python 7_partir_conversaciones.py
"""
import json
import warnings

warnings.filterwarnings("ignore")
from transformers import AutoTokenizer

BASE = "unsloth/Llama-3.2-1B-Instruct"
ENTRADA = "data/terapia.jsonl"
SALIDA = "data/terapia_trozos.jsonl"
MAX_TOKENS = 1024
# Margen para la plantilla de chat y para el turno que cierra el trozo: se mide
# ANTES de anadir el ultimo par, asi que un turno largo del asistente puede
# empujar el total. Con 64 se pasaban 21 trozos de los 729; con 160, ninguno.
MARGEN = 160

tok = AutoTokenizer.from_pretrained(BASE)


def mide(mensajes):
    texto = tok.apply_chat_template(mensajes, tokenize=False)
    return len(tok(texto)["input_ids"])


def partir(mensajes):
    """Devuelve una lista de trozos. Cada uno empieza en 'user' y acaba en
    'assistant', y mide como mucho MAX_TOKENS - MARGEN."""
    limite = MAX_TOKENS - MARGEN
    trozos = []
    actual = []
    for msg in mensajes:
        if msg["role"] == "user" and actual and mide(actual + [msg]) > limite:
            # Cerrar el trozo en el ultimo 'assistant' y arrancar otro.
            while actual and actual[-1]["role"] != "assistant":
                actual.pop()
            if actual:
                trozos.append(actual)
            actual = [msg]
            continue
        actual.append(msg)

    while actual and actual[-1]["role"] != "assistant":
        actual.pop()
    if actual:
        trozos.append(actual)

    # Un trozo de un solo par usuario/asistente no ensena contexto, pero
    # tampoco estorba; lo que no sirve es un trozo sin respuesta.
    return [t for t in trozos if len(t) >= 2]


convs = [json.loads(linea) for linea in open(ENTRADA, encoding="utf-8")]

total_turnos = 0
total_trozos = 0
descartados = 0
with open(SALIDA, "w", encoding="utf-8") as salida:
    for conv in convs:
        mensajes = conv["messages"]
        total_turnos += len(mensajes)
        trozos = partir(mensajes)
        if not trozos:
            descartados += 1
            continue
        for i, trozo in enumerate(trozos):
            registro = dict(conv)
            registro["messages"] = trozo
            registro["trozo"] = i + 1
            registro["trozos_total"] = len(trozos)
            salida.write(json.dumps(registro, ensure_ascii=False) + "\n")
            total_trozos += 1

turnos_usados = sum(
    len(json.loads(l)["messages"]) for l in open(SALIDA, encoding="utf-8"))

print(f"conversaciones  : {len(convs)}")
print(f"trozos          : {total_trozos}  ({total_trozos / len(convs):.1f} por conversacion)")
print(f"turnos originales: {total_turnos}")
print(f"turnos usados    : {turnos_usados}  ({100 * turnos_usados / total_turnos:.0f}%)")
if descartados:
    print(f"descartadas     : {descartados}")
print(f"\n{SALIDA}")
