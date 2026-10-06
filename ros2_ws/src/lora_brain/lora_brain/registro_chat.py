# -*- coding: utf-8 -*-
"""Registro de conversaciones de Chat libre -- portado tal cual de
registro_chat.py del proyecto original (mismo formato JSONL, mismo criterio
de "nunca romper el turno por un problema de disco"). Lo sigue leyendo
memoria_episodica.py, sin cambios.

Único cambio: REGISTRO ya no es relativo al archivo del módulo (que ahora
vive en el paquete instalado) sino bajo el HOME del proceso, igual que
perf_monitor.py -- configurable con CHAT_LIBRE_REGISTRO si hace falta otra
ruta (ej. compartida entre reinicios del nodo)."""
import json
import os
import time

REGISTRO = os.environ.get(
    "CHAT_LIBRE_REGISTRO",
    os.path.expanduser("~/.lora/chat_libre_training/conversaciones.jsonl"),
)
ACTIVO = os.environ.get("CHAT_LIBRE_REGISTRAR", "1") not in ("0", "false", "False")


def registrar(mensaje_usuario, respuesta, modelo):
    """Agrega un turno de Chat libre al registro. Silencioso ante cualquier
    error: perder una línea del log nunca puede cortar la conversación."""
    if not ACTIVO or not mensaje_usuario or not respuesta:
        return
    fila = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "modelo": modelo,
        "usuario": mensaje_usuario.strip(),
        "respuesta": respuesta.strip(),
        "veredicto": None,
        "correccion": None,
    }
    try:
        os.makedirs(os.path.dirname(REGISTRO), exist_ok=True)
        with open(REGISTRO, "a", encoding="utf-8") as f:
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"[registro] no se pudo guardar el turno ({e})")
