# -*- coding: utf-8 -*-
"""Prueba el flujo COMPLETO de Romeo desde la terminal, sin ROS ni robot.

Corre el OrchestratorNode de verdad -- presentacion, nombre, router,
trivia con correccion, juego de caras y chat libre -- sustituyendo solo la
capa de hardware: en vez de hablar por el altavoz, imprime; en vez de mover
motores o leer la camara, escribe lo que haria.

    python probar_flujo.py                  # modelo por defecto
    python probar_flujo.py arford-empatico-full

Escribe como si le hablaras. Comandos: /estado  /salir
"""
import os
import sys
import types

RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(RAIZ, "ros2_ws", "src", "romeo_brain"))

if len(sys.argv) > 1:
    os.environ["CHAT_MODEL"] = sys.argv[1]
os.environ.setdefault("CHAT_SERVER_HOST", "http://127.0.0.1:11434")
os.environ.setdefault("TRIVIA_CORRECTOR_MODEL", "romeo")

# --- Stubs de ROS2 -------------------------------------------------------
# romeo_brain importa rclpy y romeo_interfaces, que solo existen en la Pi.
# Se sustituyen por lo minimo que el orquestador toca al construirse.
for mod, attrs in (
    ("rclpy", {"init": lambda **k: None, "shutdown": lambda: None}),
    ("rclpy.callback_groups", {"ReentrantCallbackGroup": lambda: None}),
    ("rclpy.executors", {"MultiThreadedExecutor": object}),
    ("rclpy.node", {}),
    ("romeo_interfaces", {}),
    ("romeo_interfaces.msg", {n: type(n, (), {}) for n in
                              ("UserInput", "FaceCommand", "MotionCommand")}),
    ("romeo_interfaces.srv", {n: type(n, (), {"Request": type("R", (), {})}) for n in
                              ("DetectEmotion", "PlayMusic", "Speak")}),
):
    m = types.ModuleType(mod)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[mod] = m


# ament_index_python localiza los datos instalados del paquete (frases,
# preguntas, router). Aca no hay instalacion, asi que se apunta al fuente.
_amb = types.ModuleType("ament_index_python")
_pk = types.ModuleType("ament_index_python.packages")
_pk.get_package_share_directory = lambda nombre: os.path.join(
    RAIZ, "ros2_ws", "src", nombre, nombre)
_amb.packages = _pk
sys.modules["ament_index_python"] = _amb
sys.modules["ament_index_python.packages"] = _pk


class _Log:
    """Sustituye al logger de ROS2: lo que Romeo 'dice' sale por pantalla."""

    def info(self, t):
        if t.startswith("Asistente"):
            print(f"\nROMEO> {t.split(': ', 1)[-1]}")
        else:
            print(f"  · {t}")

    warning = error = debug = lambda self, t: print(f"  ! {t}")


class _Node:
    """Lo minimo de rclpy.node.Node que usa OrchestratorNode."""

    def __init__(self, *a, **k):
        self._log = _Log()

    def get_logger(self):
        return self._log

    def create_subscription(self, *a, **k):
        return None

    def create_timer(self, *a, **k):
        return types.SimpleNamespace(cancel=lambda: None)

    def create_client(self, *a, **k):
        return types.SimpleNamespace(service_is_ready=lambda: True)

    def create_publisher(self, *a, **k):
        return types.SimpleNamespace(publish=lambda m: None)

    def destroy_node(self):
        pass


sys.modules["rclpy.node"].Node = _Node


class _Bridge:
    """Hardware falso: imprime lo que el robot haria."""

    def __init__(self, nodo=None):
        self.speak_client = types.SimpleNamespace(service_is_ready=lambda: True)
        self._cara_actual = None

    def hablar(self, texto):
        return True            # ya se imprimio por el logger

    def mostrar_cara(self, nombre):
        # Mismo criterio que RosBridge: no repetir la cara que ya esta
        # puesta, para que la prueba refleje lo que hace el robot.
        if nombre == self._cara_actual:
            return True
        self._cara_actual = nombre
        print(f"  [cara: {nombre}]")
        return True

    def reproducir_musica(self, *a, **k):
        print("  [musica]")
        return True

    def mover(self, *a, **k):
        print("  [motores]")
        return True

    def detectar_emocion(self):
        # La camara no existe aca: se pide por teclado.
        print("  [camara] que cara pusiste? (feliz/triste/enojado/neutral, enter=no te vi)")
        r = input("  cara> ").strip().capitalize()
        if r in ("Feliz", "Triste", "Enojado", "Neutral"):
            return r, 0.9, True
        return None, 0.0, False


import romeo_brain._ros_bridge as rb                      # noqa: E402
rb.RosBridge = _Bridge
from romeo_brain import orchestrator_node as orq          # noqa: E402
orq.RosBridge = _Bridge
orq.time.sleep = lambda s: None                           # sin pausas de cara

nodo = orq.OrchestratorNode()
nodo._iniciar_sesion()

print("\n  (escribe como si le hablaras. /estado  /salir)\n")
while True:
    try:
        entrada = input("tu> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not entrada:
        continue
    if entrada == "/salir":
        break
    if entrada == "/estado":
        e = nodo.estado
        print(f"  nombre={e['nombre']} en_trivia={e['en_trivia']} "
              f"esperando_tema={e['esperando_tema']} "
              f"aciertos={e['aciertos']}/{e['total']}")
        continue
    try:
        nodo._procesar_turno(entrada)
    except Exception as e:  # noqa: BLE001
        print(f"  [error] {type(e).__name__}: {e}")
    print()

print("  hasta luego")
