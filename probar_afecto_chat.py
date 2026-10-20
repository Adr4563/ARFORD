# -*- coding: utf-8 -*-
"""Conversar con Romeo desde la PC viendo como se mueve su animo.

Igual que probar_en_pi.py pero SIN necesitar ROS instalado: finge
ament_index_python para que frases.py y preguntas.py encuentren sus datos
en el fuente. Sirve para probar el TER y ROSE en la PC antes de llevarlo a
la Pi.

    python probar_afecto_chat.py                 # IA en este mismo equipo
    python probar_afecto_chat.py 10.143.161.32   # IA en otra maquina

Comandos:
    /animo      el estado afectivo ahora mismo
    /nuevo      sesion nueva (olvida el hilo y el animo)
    /historial  lo que recuerda
    /salir
"""
import os
import sys
import types

RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(RAIZ, "ros2_ws", "src", "romeo_brain"))

# ament_index_python localiza los datos instalados del paquete (frases,
# preguntas, router). Aca no hay instalacion, asi que se apunta al fuente.
_amb = types.ModuleType("ament_index_python")
_pk = types.ModuleType("ament_index_python.packages")
_pk.get_package_share_directory = lambda nombre: os.path.join(
    RAIZ, "ros2_ws", "src", nombre, nombre)
_amb.packages = _pk
sys.modules["ament_index_python"] = _amb
sys.modules["ament_index_python.packages"] = _pk

PC = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
os.environ["CHAT_SERVER_HOST"] = f"http://{PC}:11434"
os.environ.setdefault("CHAT_MODEL", "llama3.2:1b")

from romeo_brain import agente_chat, estado_afectivo, frases, llama_client, ter  # noqa: E402

afecto = estado_afectivo.EstadoAfectivo()


def linea_animo():
    v, a, d = afecto.vector
    return f"v{v:+.2f} a{a:+.2f} d{d:+.2f}  cara={afecto.cara()}"


print()
print("  === Romeo con animo (PULSE + ROSE + TER) ===")
print(f"  modelo    {llama_client.CHAT_MODEL} en {PC}")
print(f"  empatia   {afecto.empatia}   peso del texto "
      f"{estado_afectivo._PESO_TEXTO}")
print(f"  animo     {linea_animo()}")
print("  comandos  /animo  /nuevo  /historial  /salir")
print()
print(f"ROMEO> {frases.una(frases.PRESENTACIONES)} "
      f"{frases.una(frases.SALUDOS_APERTURA)}")
print()

esperando_nombre = True

while True:
    try:
        entrada = input("tu> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if not entrada:
        continue

    if entrada == "/salir":
        break

    if entrada == "/animo":
        v, a, d = afecto.vector
        b = afecto.animo_base
        print(f"  vector [{v:+.2f}, {a:+.2f}, {d:+.2f}]")
        print(f"  base   [{b[0]:+.2f}, {b[1]:+.2f}, {b[2]:+.2f}]")
        print(f"  cara   {afecto.cara()}")
        print()
        continue

    if entrada == "/nuevo":
        agente_chat.reiniciar()
        afecto.reiniciar()
        esperando_nombre = True
        print("  (sesion nueva)")
        print(f"ROMEO> {frases.una(frases.PRESENTACIONES)} "
              f"{frases.una(frases.SALUDOS_APERTURA)}")
        print()
        continue

    if entrada == "/historial":
        for m in getattr(agente_chat, "_historial", []):
            print(f"    {m['role']:9s} {m['content'][:68]}")
        print()
        continue

    if esperando_nombre:
        esperando_nombre = False
        print(f"ROMEO> {frases.una(frases.BIENVENIDAS, nombre=entrada)}")
        print()
        continue

    # TER + ROSE, en el mismo punto que orchestrator_node: antes de
    # responder, y una sola vez por mensaje.
    afecto.decaer()
    emo, conf = ter.analizar(entrada)
    vector, c = estado_afectivo.percibir_del_texto(emo, conf)
    if vector is not None:
        afecto.empatizar(estado_afectivo.fusionar([(vector, c)]))
        print(f"  [afecto] el texto expresa {emo} ({conf:.2f})   {linea_animo()}")
    else:
        print(f"  [afecto] nada emocional en el texto   {linea_animo()}")

    try:
        print(f"ROMEO> {agente_chat.responder(entrada)}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {str(e)[:110]}")
    print()

print()
print(f"  animo final: {linea_animo()}")
print()
