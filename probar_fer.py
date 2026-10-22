# -*- coding: utf-8 -*-
"""Pruebas de la ventana de percepcion con FER y SER, sin camara, modelo ni ROS.

Cubre lo que agrega la integracion de FER (tesis de Rocha et al.):
  - el resumen de la ventana: mediana por emocion, normalizada (4.1.4.1)
  - PULSE: confianza por entropia (4.1-4.2) y CAT -> VAD (4.3)
  - la fusion TER + FER (4.4) y su efecto en el animo de Romeo (ROSE)

    python probar_fer.py
"""
import os
import sys
import types

RAIZ = os.path.dirname(os.path.abspath(__file__))
for _p in ("romeo_brain", "romeo_drivers"):
    sys.path.insert(0, os.path.join(RAIZ, "ros2_ws", "src", _p))

# --- Stubs de ROS 2 (igual que probar_flujo.py) --------------------------
for mod, attrs in (
    ("rclpy", {"init": lambda **k: None, "shutdown": lambda: None}),
    ("rclpy.callback_groups", {"ReentrantCallbackGroup": lambda: None}),
    ("rclpy.executors", {"MultiThreadedExecutor": object}),
    ("rclpy.node", {}),
    ("romeo_interfaces", {}),
    ("romeo_interfaces.msg", {n: type(n, (), {}) for n in
                              ("UserInput", "FaceCommand", "MotionCommand")}),
    ("romeo_interfaces.srv", {n: type(n, (), {"Request": type("R", (), {})}) for n in
                              ("DetectEmotion", "PerceiveFace", "PlayMusic", "Speak")}),
):
    m = types.ModuleType(mod)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[mod] = m
_amb = types.ModuleType("ament_index_python")
_pk = types.ModuleType("ament_index_python.packages")
_pk.get_package_share_directory = lambda n: os.path.join(RAIZ, "ros2_ws", "src", n, n)
_amb.packages = _pk
sys.modules["ament_index_python"] = _amb
sys.modules["ament_index_python.packages"] = _pk
os.environ.setdefault("TRIVIA_CORRECTOR_MODEL", "")

from romeo_brain import estado_afectivo as af  # noqa: E402
from romeo_drivers._emotion_detector import ETIQUETAS, DetectorEmocion  # noqa: E402

fallos = []


def comprobar(descripcion, condicion, detalle=""):
    print(f"  [{'ok  ' if condicion else 'FALLA'}] {descripcion}"
          f"{('  ' + detalle) if detalle else ''}")
    if not condicion:
        fallos.append(descripcion)


def distribucion(**pesos):
    """8 probabilidades en el orden de ETIQUETAS a partir de nombres."""
    p = [pesos.get(e, 0.0) for e in ETIQUETAS]
    t = sum(p)
    return [x / t for x in p]


CARA_FELIZ = distribucion(felicidad=0.85, neutral=0.10, sorpresa=0.05)
CARA_TRISTE = distribucion(tristeza=0.80, neutral=0.15, enojo=0.05)
CARA_NEUTRA = distribucion(neutral=0.90, felicidad=0.05, tristeza=0.05)

print("=" * 70)
print("1. Resumen de la ventana: mediana por emocion, normalizada")
print("=" * 70)
r = DetectorEmocion.resumir([CARA_FELIZ, CARA_FELIZ, CARA_TRISTE])
comprobar("suma 1", abs(sum(r) - 1.0) < 1e-9, f"-> {sum(r):.6f}")
comprobar("un cuadro raro no gana (mediana, no media)",
          ETIQUETAS[r.index(max(r))] == "felicidad",
          f"-> {ETIQUETAS[r.index(max(r))]}")
comprobar("sin cuadros -> None", DetectorEmocion.resumir([]) is None)

det = DetectorEmocion.__new__(DetectorEmocion)
import collections, threading  # noqa: E401,E402
det._observaciones = collections.deque([(10.0, CARA_TRISTE), (11.0, CARA_FELIZ),
                                        (12.0, CARA_FELIZ), (20.0, CARA_TRISTE)])
det._lock = threading.Lock()
probs, n = det.resumen(10.5, 12.5)
comprobar("solo entran los cuadros de la ventana", n == 2, f"-> {n} cuadros")
probs, n = det.resumen(30.0, 31.0)
comprobar("ventana sin cara -> (None, 0)", probs is None and n == 0)

print()
print("=" * 70)
print("2. PULSE: entropia (4.1-4.2) y CAT -> VAD (4.3)")
print("=" * 70)
v, c, dom = af.percibir_categorias(CARA_FELIZ, ETIQUETAS)
comprobar("cara feliz -> valencia positiva", v[0] > 0.5, f"-> v={v[0]:+.2f}")
comprobar("dominante felicidad", dom == "felicidad")
_, c_seg, _ = af.percibir_categorias(distribucion(tristeza=0.97, neutral=0.03), ETIQUETAS)
_, c_dud, _ = af.percibir_categorias(distribucion(tristeza=0.4, enojo=0.35, neutral=0.25), ETIQUETAS)
comprobar("segura -> mas confianza que dudosa", c_seg > c_dud,
          f"{c_seg:.2f} > {c_dud:.2f}")
vm, _, _ = af.percibir_categorias(distribucion(tristeza=0.6, neutral=0.4), ETIQUETAS)
vt, _, _ = af.percibir_categorias(distribucion(tristeza=1.0), ETIQUETAS)
comprobar("60 % triste cae entre neutral y triste (usa toda la distribucion)",
          vt[0] < vm[0] < 0, f"{vt[0]:+.2f} < {vm[0]:+.2f} < 0")
comprobar("distribucion vacia -> sin lectura",
          af.percibir_categorias([], ETIQUETAS)[0] is None)

print()
print("=" * 70)
print("3. Turno completo del orquestador: TER + FER -> fusion -> ROSE")
print("=" * 70)


class _Log:
    def __init__(self):
        self.lineas = []

    def info(self, t):
        self.lineas.append(t)

    warning = error = debug = info


class _Node:
    def __init__(self, *a, **k):
        self._log = _Log()

    def get_logger(self):
        return self._log

    def declare_parameter(self, n, d=None):
        self._p = getattr(self, "_p", {})
        self._p[n] = d

    def get_parameter(self, n):
        return types.SimpleNamespace(value=self._p[n])

    def create_subscription(self, *a, **k):
        return None

    def create_timer(self, *a, **k):
        return types.SimpleNamespace(cancel=lambda: None)


sys.modules["rclpy.node"].Node = _Node


class _Bridge:
    """Hardware falso: la camara devuelve lo que se le diga."""
    def __init__(self, nodo=None):
        self.speak_client = types.SimpleNamespace(service_is_ready=lambda: True)
        self._cara_actual = None
        self.cara_fer = None
        self.ventanas = []

    def percibir_cara(self, t0, t1):
        self.ventanas.append((t0, t1))
        if self.cara_fer is None:
            return list(ETIQUETAS), [], 0
        return list(ETIQUETAS), list(self.cara_fer), 6

    def hablar(self, t):
        return True

    def mostrar_cara(self, n):
        self._cara_actual = n

    def mover(self, *a):
        pass

    def reproducir_musica(self, *a, **k):
        return True

    def detectar_emocion(self):
        return None, 0.0, False


import romeo_brain._ros_bridge as rb  # noqa: E402
rb.RosBridge = _Bridge
from romeo_brain import llama_client, orchestrator_node as orq  # noqa: E402
orq.RosBridge = _Bridge
orq.time.sleep = lambda s: None
llama_client.generar_respuesta = lambda *a, **k: "respuesta simulada"
llama_client.precargar = lambda *a, **k: None


def turno(texto, cara, con_ventana=True, voz=None, fuentes=None):
    """Un turno de chat libre en un nodo nuevo; devuelve (nodo, valencia).
    `voz` es una distribucion SER (dict emocion -> probabilidad)."""
    nodo = orq.OrchestratorNode()
    if fuentes is not None:
        nodo._fuentes = set(fuentes)
    nodo._iniciar_sesion()
    nodo.estado["esperando_nombre"] = False
    nodo._bridge.cara_fer = cara
    t = (100.0, 103.0) if con_ventana else (0.0, 0.0)
    ser = (list(voz), list(voz.values())) if voz else None
    nodo._procesar_turno(texto, *t, ser)
    return nodo, nodo._afecto.vector[0]


base = af.ANIMO_BASE[0]
n, v = turno("hola que tal", CARA_FELIZ)
comprobar("se pide la ventana de la frase", n._bridge.ventanas == [(100.0, 103.0)])
comprobar("solo la cara feliz (sin texto emocional) SUBE el animo", v > base,
          f"{base:+.2f} -> {v:+.2f}")
n, v = turno("hola que tal", CARA_TRISTE)
comprobar("solo la cara triste lo BAJA", v < base, f"{base:+.2f} -> {v:+.2f}")
n, v = turno("hola que tal", CARA_NEUTRA)
comprobar("cara neutra sola NO mueve el animo", v == base, f"-> {v:+.2f}")
_, v_txt = turno("estoy muy triste", None)
_, v_neu = turno("estoy muy triste", CARA_NEUTRA)
_, v_tri = turno("estoy muy triste", CARA_TRISTE)
comprobar("texto triste + cara neutra: la cara lo modera",
          v_txt < v_neu, f"{v_txt:+.2f} < {v_neu:+.2f}")
# La fusion (4.4) es un PROMEDIO ponderado: con las dos tristes el
# resultado queda entre ambas, no mas triste que el texto solo. Lo que se
# comprueba es que la cara triste lo deja mucho mas triste que la neutra.
comprobar("texto triste + cara triste: queda triste (no lo modera)",
          v_tri < v_neu - 0.1, f"{v_tri:+.2f} < {v_neu:+.2f}")
n, v = turno("estoy muy triste", CARA_FELIZ)
comprobar("hay linea de fusion en el log",
          any("fusion de TER+FER" in l for l in n._log.lineas))
n, v = turno("hola", CARA_FELIZ, con_ventana=False)
comprobar("por teclado (sin ventana) no se mira la camara",
          n._bridge.ventanas == [] and v == base)

print()
print("=" * 70)
print("4. SER: etiquetas del modelo y fusion con TER y FER")
print("=" * 70)
from romeo_drivers import _ser_engine as ser  # noqa: E402
salida_mesd = [{"label": "Sadness", "score": 0.7}, {"label": "Neutral", "score": 0.2},
               {"label": "Anger", "score": 0.1}]
etq, prb = ser.agrupar(salida_mesd)
comprobar("etiquetas del modelo (ingles) -> claves de PULSE",
          set(etq) == {"tristeza", "neutral", "enojo"}, f"-> {etq}")
comprobar("probabilidades suman 1", abs(sum(prb) - 1) < 1e-9)
comprobar("formato emotion2vec ('生气/angry') tambien se entiende",
          ser.traducir_etiqueta("生气/angry") == "enojo")
etq, prb = ser.agrupar([{"label": "<unk>", "score": 0.5}, {"label": "Happiness", "score": 0.5}])
comprobar("etiquetas desconocidas se descartan y se renormaliza",
          etq == ["felicidad"] and prb == [1.0])
comprobar("sin emociones conocidas -> None", ser.agrupar([{"label": "other", "score": 1}]) is None)
clas = ser.ClasificadorVoz()
comprobar("sin modelo cargado, clasificar no falla (None)",
          clas.clasificar([0.1] * 16000) is None)

VOZ_TRISTE = {"tristeza": 0.75, "neutral": 0.2, "enojo": 0.05}
VOZ_NEUTRA = {"neutral": 0.9, "tristeza": 0.05, "felicidad": 0.05}
_, v = turno("hola que tal", None, voz=VOZ_TRISTE)
comprobar("solo la voz triste baja el animo", v < base, f"{base:+.2f} -> {v:+.2f}")
_, v = turno("hola que tal", CARA_NEUTRA, voz=VOZ_NEUTRA)
comprobar("cara y voz neutras (sin texto) NO mueven el animo", v == base, f"-> {v:+.2f}")
n, v3 = turno("estoy muy triste", CARA_TRISTE, voz=VOZ_TRISTE)
comprobar("las tres fuentes se fusionan",
          any("fusion de TER+FER+SER" in l for l in n._log.lineas))
comprobar("texto, cara y voz tristes -> animo triste", v3 < 0.15, f"-> {v3:+.2f}")
_, v_solo_ter = turno("estoy muy triste", CARA_FELIZ, voz=VOZ_NEUTRA, fuentes={"TER"})
comprobar("fuentes_percepcion=[TER] ignora cara y voz", abs(v_solo_ter - v_txt) < 1e-9,
          f"{v_solo_ter:+.2f} == {v_txt:+.2f}")

print()
print("=" * 70)
if fallos:
    print(f"  {len(fallos)} FALLOS:")
    for f in fallos:
        print(f"    - {f}")
    sys.exit(1)
print("  todo correcto")
print("=" * 70)
