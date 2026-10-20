# -*- coding: utf-8 -*-
"""Perfila un turno de Romeo FUNCION POR FUNCION, para ver donde se va el
tiempo de verdad.

Recorre el camino completo de un mensaje -- TER, router, afecto, agentes,
LLM -- con cProfile, y ademas cronometra cada etapa por separado. Al final
imprime las funciones mas caras ordenadas por tiempo acumulado.

Hay DOS medidas y conviene no confundirlas:

  tiempo PROPIO (tottime)  lo que tarda la funcion sin contar a quien llama
  tiempo ACUMULADO (cumtime) incluye todo lo que llama por dentro

Una funcion con cumtime alto y tottime bajo no es lenta: esta esperando a
otra cosa (tipicamente la red). Eso es justo lo que pasa con el LLM y el
STT, que viven en la PC: su coste es espera, no computo.

    python perfilar_turno.py                   # IA en este equipo
    python perfilar_turno.py 10.143.161.32     # IA en otra maquina
    python perfilar_turno.py --sin-red         # solo lo local, sin LLM
"""
import cProfile
import io as _io
import os
import pstats
import sys
import time
import types

RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(RAIZ, "ros2_ws", "src", "romeo_brain"))

# Stubs de ROS2: romeo_brain importa rclpy y romeo_interfaces, que solo
# existen en la Pi. Se sustituye lo minimo que el orquestador toca al
# importarse -- aqui no se construye ningun nodo, solo se perfilan las
# funciones de logica.
for _mod, _attrs in (
    ("rclpy", {"init": lambda **k: None, "shutdown": lambda: None}),
    ("rclpy.callback_groups", {"ReentrantCallbackGroup": lambda: None}),
    ("rclpy.executors", {"MultiThreadedExecutor": object}),
    ("rclpy.node", {"Node": object}),
    ("romeo_interfaces", {}),
    ("romeo_interfaces.msg", {n: type(n, (), {}) for n in
                              ("UserInput", "FaceCommand", "MotionCommand")}),
    ("romeo_interfaces.srv", {n: type(n, (), {"Request": type("R", (), {})})
                              for n in ("DetectEmotion", "PlayMusic", "Speak")}),
):
    _m = types.ModuleType(_mod)
    for _k, _v in _attrs.items():
        setattr(_m, _k, _v)
    sys.modules[_mod] = _m

# ament_index_python localiza los datos del paquete instalado; aca se apunta
# al fuente para poder perfilar sin ROS.
_amb = types.ModuleType("ament_index_python")
_pk = types.ModuleType("ament_index_python.packages")
_pk.get_package_share_directory = lambda n: os.path.join(
    RAIZ, "ros2_ws", "src", n, n)
_amb.packages = _pk
sys.modules["ament_index_python"] = _amb
sys.modules["ament_index_python.packages"] = _pk

SIN_RED = "--sin-red" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
PC = args[0] if args else "127.0.0.1"
os.environ["CHAT_SERVER_HOST"] = f"http://{PC}:11434"
os.environ.setdefault("CHAT_MODEL", "llama3.2:1b")

from romeo_brain import agente_chat, agente_trivia, estado_afectivo  # noqa: E402
from romeo_brain import frases, preguntas, ter  # noqa: E402
from romeo_brain.orchestrator_node import enrutar  # noqa: E402

MENSAJES = [
    "hola, que tal estas",
    "me siento muy triste y solo hoy",
    "quiero jugar una trivia",
    "cuentame algo interesante",
]

etapas = []


def cronometrar(nombre, fn, *a, **k):
    """Mide UNA etapa y guarda el resultado."""
    t0 = time.perf_counter()
    r = fn(*a, **k)
    etapas.append((nombre, (time.perf_counter() - t0) * 1000))
    return r


def un_turno(mensaje, afecto):
    """El camino que recorre un mensaje dentro del orquestador."""
    cronometrar("afecto.decaer", afecto.decaer)
    emo, conf = cronometrar("ter.analizar", ter.analizar, mensaje)
    vec, c = cronometrar("pulse.percibir", estado_afectivo.percibir_del_texto,
                         emo, conf)
    if vec is not None:
        fus = cronometrar("pulse.fusionar", estado_afectivo.fusionar, [(vec, c)])
        cronometrar("rose.empatizar", afecto.empatizar, fus, c)
    cronometrar("afecto.cara", afecto.cara)
    ruta = cronometrar("router.enrutar", enrutar, mensaje)
    if ruta == "TRIVIA":
        cronometrar("banco.categorias", preguntas.categorias_disponibles)
        cronometrar("banco.preguntas", preguntas.preguntas_por_categoria,
                    "Ciencia", cantidad=1)
    elif not SIN_RED:
        cronometrar("LLM.responder (red)", agente_chat.responder, mensaje)
    cronometrar("frases.una", frases.una, frases.RESPUESTAS_TRIVIA_ACIERTO)
    return ruta


def recorrido():
    afecto = estado_afectivo.EstadoAfectivo()
    for m in MENSAJES:
        un_turno(m, afecto)
    # El corrector de trivia, que es otro camino con red.
    if not SIN_RED:
        cronometrar("trivia.evaluar", agente_trivia.evaluar_respuesta, "21", "21")
        cronometrar("trivia.comentar (red)", agente_trivia.comentar,
                    "Cuanto es 3 por 7?", "21", "21", True)


print()
print("=" * 72)
print(f"  Perfilado de {len(MENSAJES)} turnos   "
      f"{'(SIN red: solo lo local)' if SIN_RED else f'(IA en {PC})'}")
print("=" * 72)

perfil = cProfile.Profile()
t0 = time.perf_counter()
perfil.enable()
recorrido()
perfil.disable()
total_ms = (time.perf_counter() - t0) * 1000

# ── Por etapa: donde se va el tiempo del turno ────────────────────────
print()
print("  TIEMPO POR ETAPA (sumando los turnos)")
print("  " + "-" * 68)
por_etapa = {}
for nombre, ms in etapas:
    d = por_etapa.setdefault(nombre, [])
    d.append(ms)
for nombre, ms in sorted(por_etapa.items(), key=lambda x: -sum(x[1])):
    s = sum(ms)
    barra = "#" * max(1, int(s / total_ms * 45))
    print(f"  {nombre:24s} {s:8.2f} ms  n={len(ms):2d}  "
          f"{s/total_ms*100:5.1f}%  {barra}")
print("  " + "-" * 68)
print(f"  {'TOTAL':24s} {total_ms:8.2f} ms")

# ── Funcion por funcion ───────────────────────────────────────────────
print()
print("  LAS 18 FUNCIONES MAS CARAS (por tiempo acumulado)")
print("  " + "-" * 68)
buf = _io.StringIO()
st = pstats.Stats(perfil, stream=buf).sort_stats("cumulative")
st.print_stats(18)
salida = buf.getvalue()
empezo = False
for linea in salida.splitlines():
    if "ncalls" in linea:
        empezo = True
        print("  " + linea.strip())
        continue
    if empezo and linea.strip():
        # Se acorta la ruta del archivo, que ocupa media pantalla.
        print("  " + linea.strip()[:150])

print()
print("  Nota: cumtime alto con tottime bajo = la funcion ESPERA, no calcula.")
print("  Tipicamente es red (el LLM y el STT viven en la PC).")
print()
