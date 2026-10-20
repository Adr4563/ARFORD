# -*- coding: utf-8 -*-
"""Terminal para conversar con Romeo desde la PC, sin robot ni ROS.

Usa el MISMO agente_chat del robot -- el que acaba de aprender a mantener el
hilo -- contra el Ollama local. Sirve para probar como conversa sin tener que
encender la Raspberry ni hablarle al microfono.

    python chat_romeo.py                 # modelo por defecto
    python chat_romeo.py romeo           # otro modelo de ollama list

Comandos dentro del chat:
    /nuevo     empieza una sesion nueva (olvida el hilo)
    /historial muestra lo que recuerda ahora mismo
    /modelo X  cambia de modelo en caliente
    /salir     terminar
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "ros2_ws", "src", "romeo_brain"))

if len(sys.argv) > 1:
    os.environ["CHAT_MODEL"] = sys.argv[1]
os.environ.setdefault("CHAT_SERVER_HOST", "http://127.0.0.1:11434")

from romeo_brain import agente_chat, llama_client  # noqa: E402

print()
print(f"  Romeo por terminal -- modelo {llama_client.CHAT_MODEL} "
      f"en {llama_client.CHAT_SERVER_HOST}")
print(f"  recuerda {agente_chat.TURNOS_HISTORIAL} turnos. "
      f"/nuevo /historial /modelo /salir")
print()

while True:
    try:
        entrada = input("tu> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if not entrada:
        continue

    if entrada in ("/salir", "/exit", "/quit"):
        break
    if entrada == "/nuevo":
        agente_chat.reiniciar()
        print("  (sesion nueva, hilo olvidado)\n")
        continue
    if entrada == "/historial":
        if not agente_chat._historial:
            print("  (vacio)\n")
        for m in agente_chat._historial:
            quien = "tu   " if m["role"] == "user" else "romeo"
            print(f"  {quien}: {m['content'][:70]}")
        print()
        continue
    if entrada.startswith("/modelo"):
        partes = entrada.split()
        if len(partes) > 1:
            llama_client.CHAT_MODEL = partes[1]
            agente_chat.reiniciar()
            print(f"  (modelo: {partes[1]}, hilo olvidado)\n")
        else:
            print(f"  modelo actual: {llama_client.CHAT_MODEL}\n")
        continue

    t0 = time.perf_counter()
    try:
        respuesta = agente_chat.responder(entrada)
    except Exception as e:  # noqa: BLE001 - la terminal no se cae por un fallo
        print(f"  [error] {e}\n")
        continue
    print(f"romeo> {respuesta}")
    print(f"       ({time.perf_counter() - t0:.1f}s, "
          f"{len(agente_chat._historial) // 2} turnos en memoria)\n")

print("  hasta luego")
