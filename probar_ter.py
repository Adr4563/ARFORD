# -*- coding: utf-8 -*-
"""Pruebas del TER y de su integracion con PULSE/ROSE. Sin ROS.

    python probar_ter.py
"""
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "ros2_ws", "src", "romeo_brain"))

from romeo_brain import estado_afectivo as af  # noqa: E402
from romeo_brain import ter  # noqa: E402

fallos = []


def comprobar(descripcion, condicion, detalle=""):
    estado = "ok  " if condicion else "FALLA"
    print(f"  [{estado}] {descripcion}{('  ' + detalle) if detalle else ''}")
    if not condicion:
        fallos.append(descripcion)


print("=" * 70)
print("1. Detecta lo que el texto expresa")
print("=" * 70)
for texto, esperada in [
    ("me siento muy triste hoy", "tristeza"),
    ("estoy feliz, me encanta esto", "felicidad"),
    ("estoy harto, es injusto", "enojo"),
    ("tengo miedo, estoy muy nervioso", "miedo"),
]:
    emo, conf = ter.analizar(texto)
    comprobar(f'"{texto}"', emo == esperada, f"-> {emo} ({conf})")

print()
print("=" * 70)
print("2. Negaciones -- no invertir, descartar")
print("=" * 70)
for texto in ["no estoy triste", "ya no me siento solo", "nunca estoy enojado"]:
    emo, conf = ter.analizar(texto)
    comprobar(f'"{texto}" no detecta la emocion negada',
              emo is None, f"-> {emo}")
emo, _ = ter.analizar("no estoy triste")
comprobar("negar tristeza NO afirma felicidad", emo != "felicidad")

print()
print("=" * 70)
print("3. Sin evidencia -- no forzar una emocion")
print("=" * 70)
for texto in ["la capital de peru es lima", "si", "no se", "la siguiente",
              "cuanto es dos mas dos", ""]:
    emo, conf = ter.analizar(texto)
    comprobar(f'"{texto}" -> sin evidencia',
              emo is None and conf == 0.0, f"-> {emo}, {conf}")

print()
print("=" * 70)
print("4. Ambiguedad -- callarse antes que elegir a cara o cruz")
print("=" * 70)
emo, conf = ter.analizar("estoy contento pero muy triste a la vez")
comprobar("señales opuestas -> sin evidencia", emo is None, f"-> {emo}")

print()
print("=" * 70)
print("5. La confianza esta acotada")
print("=" * 70)
_, conf = ter.analizar("triste tristeza deprimido llorando destrozado "
                       "sufriendo dolor pena vacio")
comprobar(f"nueve terminos fuertes no pasan de {ter.CONFIANZA_MAXIMA}",
          conf <= ter.CONFIANZA_MAXIMA, f"-> {conf}")
_, c1 = ter.analizar("estoy mal")
_, c2 = ter.analizar("estoy destrozado y sufriendo mucho dolor")
comprobar("mas evidencia -> mas confianza", c2 > c1, f"{c1} < {c2}")

print()
print("=" * 70)
print("6. Integracion con PULSE")
print("=" * 70)
v, c = af.percibir_del_texto(*ter.analizar("estoy muy triste"))
comprobar("tristeza da valencia negativa", v is not None and v[0] < 0,
          f"-> {v}")
v, c = af.percibir_del_texto(*ter.analizar("la capital es lima"))
comprobar("sin evidencia -> sin vector", v is None and c == 0.0)

print()
print("=" * 70)
print("7. fusionar() con una sola modalidad (no se inventa interfaz)")
print("=" * 70)
v1, c1 = af.percibir_del_texto("tristeza", 0.5)
solo = af.fusionar([(v1, c1)])
comprobar("una lectura -> esa misma", solo == v1, f"-> {solo}")
comprobar("lista vacia -> None", af.fusionar([]) is None)
comprobar("lecturas nulas se filtran", af.fusionar([(None, 0.0)]) is None)
dos = af.fusionar([(v1, c1), (af.ANCLAJES["felicidad"], 0.9)])
comprobar("dos lecturas -> media ponderada",
          dos is not None and v1[0] < dos[0], f"-> {dos}")

print()
print("=" * 70)
print("8. Hace falta insistir para que cambie la cara")
print("=" * 70)
e = af.EstadoAfectivo()
v, c = af.percibir_del_texto(*ter.analizar("estoy triste"))
e.empatizar(v)
comprobar("un mensaje triste NO cambia la cara", e.cara() == "content",
          f"-> {e.cara()}")
for _ in range(4):
    v, c = af.percibir_del_texto(*ter.analizar("estoy muy triste y solo"))
    e.empatizar(v)
comprobar("insistir SI la cambia", e.cara() == "sad", f"-> {e.cara()}")

print()
print("=" * 70)
if fallos:
    print(f"  {len(fallos)} FALLOS:")
    for f in fallos:
        print(f"    - {f}")
    sys.exit(1)
print("  todo correcto")
print("=" * 70)
