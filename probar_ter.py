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
print("8. El perfil mueve el animo de verdad (tesis, ec. 4.5)")
print("=" * 70)
# La empatia se aplica ENTERA: OPTIMISTIC mueve el 40 % hacia el usuario,
# no 0.4 x la confianza del TER (~12 %), que no sacaba nunca al animo de
# la zona neutra.
e = af.EstadoAfectivo(perfil="OPTIMISTIC")
v, c = af.percibir_del_texto(*ter.analizar("estoy triste"))
antes = e.vector
e.empatizar(v, c)
esperado = tuple(0.4 * v[i] + 0.6 * antes[i] for i in range(3))
comprobar("OPTIMISTIC mueve el 40 % del camino",
          all(abs(e.vector[i] - esperado[i]) < 1e-9 for i in range(3)),
          f"-> {tuple(round(x, 2) for x in e.vector)}")
comprobar("un mensaje triste ya se nota en la cara", e.cara() == "sad",
          f"-> {e.cara()}")
emp = af.EstadoAfectivo(perfil="EMPATHETIC")
emp.empatizar(v, c)
comprobar("EMPATHETIC se mueve mas que OPTIMISTIC",
          emp.vector[0] < e.vector[0], f"{emp.vector[0]:+.2f} < {e.vector[0]:+.2f}")
comprobar("sin confianza no mueve nada",
          af.EstadoAfectivo().empatizar(v, 0.0) == af.ANIMO_BASE)

# El caso del log del robot: enojo, 24 s, enojo otra vez. Antes el animo
# volvia entero a la base entre turnos y nunca salia de neutral_L0.
from romeo_brain import aeb  # noqa: E402
e = af.EstadoAfectivo(perfil="OPTIMISTIC")
t = 1000.0
estados = []
for texto in ["muy enojado porque ayer me mintieron",
              "soy muy enojado con la mentira de mi amigo, es frustrante"]:
    e.decaer(ahora=t)
    v, c = af.percibir_del_texto(*ter.analizar(texto))
    e.empatizar(v, c)
    e._ultimo_estimulo = t
    estados.append(aeb.mapear(e)["estado"])
    t += 24
comprobar("conversacion enojada real -> enojo, no neutral",
          all(x.startswith("enojo") for x in estados), f"-> {estados}")

# Un texto alegre SUBE el animo (regresion: antes lo bajaba, porque el
# anclaje se escalaba por la confianza y quedaba cerca del origen, que es
# neutral).
e = af.EstadoAfectivo()
antes = e.vector[0]
v, c = af.percibir_del_texto(*ter.analizar("estoy feliz, que alegria"))
e.empatizar(v, c)
comprobar("un texto alegre SUBE la valencia", e.vector[0] > antes,
          f"{antes:+.3f} -> {e.vector[0]:+.3f}")

print()
print("=" * 70)
print("9. Regresiones del TER (ñ, 'solo', 'basta')")
print("=" * 70)
emo, _ = ter.analizar("te extraño mucho")
comprobar('"te extraño" casa (la ñ ya no la pierde)', emo == "tristeza", f"-> {emo}")
for texto in ["solo quiero jugar", "tan solo una pregunta mas", "solo dime la respuesta"]:
    emo, _ = ter.analizar(texto)
    comprobar(f'"{texto}" -> "solo" adverbio, sin evidencia', emo is None, f"-> {emo}")
for texto in ["me siento solo", "estoy muy solo"]:
    emo, _ = ter.analizar(texto)
    comprobar(f'"{texto}" -> tristeza', emo == "tristeza", f"-> {emo}")
emo, _ = ter.analizar("no me siento solo")
comprobar('"no me siento solo" -> negado', emo is None, f"-> {emo}")
emo, _ = ter.analizar("basta")
comprobar('"basta" ya no es enojo', emo is None, f"-> {emo}")
emo, _ = ter.analizar("me mintio, es frustrante")
comprobar('"frustrante" -> enojo', emo == "enojo", f"-> {emo}")
comprobar('"molesta" pesa 0.9, no 0.8',
          ter._TERMINOS["enojo"]["molesta"] == 0.9)
sin_normalizar = [t for terms in ter._TERMINOS.values() for t in terms
                  if t != ter._normalizar(t)]
comprobar("todas las claves normalizadas", not sin_normalizar, f"{sin_normalizar}")

print()
print("=" * 70)
print("10. Decaimiento segun el tiempo real")
print("=" * 70)
e = af.EstadoAfectivo()
e.empatizar(af.ANCLAJES["tristeza"])
t0 = e._ultimo_estimulo
tras = e.vector
e.decaer(ahora=t0 + e.latencia - 0.1)
comprobar("durante la latencia no decae", e.vector == tras)
e.decaer(ahora=t0 + e.latencia)
uno = e.vector
comprobar("al terminar la latencia, un paso", uno != tras)
e.decaer(ahora=t0 + e.latencia + 0.5)
comprobar("llamar dos veces en el mismo tramo no descuenta dos veces", e.vector == uno)
e2 = af.EstadoAfectivo()
e2.empatizar(af.ANCLAJES["tristeza"])
e2.decaer(ahora=e2._ultimo_estimulo + e2.latencia + 3 * e2.paso)
e3 = af.EstadoAfectivo()
e3.empatizar(af.ANCLAJES["tristeza"])
for k in range(4):   # el mismo tiempo, en cuatro llamadas
    e3.decaer(ahora=e3._ultimo_estimulo + e3.latencia + k * e3.paso)
comprobar("de una vez o en varias llamadas da lo mismo",
          all(abs(e2.vector[i] - e3.vector[i]) < 1e-9 for i in range(3)))
e4 = af.EstadoAfectivo()
e4.empatizar(af.ANCLAJES["tristeza"])
e4.decaer(ahora=e4._ultimo_estimulo + 600)
comprobar("10 min de silencio -> vuelve a la base", e4.vector == e4.animo_base,
          f"-> {e4.vector}")

print()
print("=" * 70)
print("11. Perfiles afectivos")
print("=" * 70)
for nombre, p in af.PERFILES.items():
    est = af.EstadoAfectivo(perfil=nombre)
    comprobar(f"{nombre} se carga", est.perfil == nombre and est.empatia == p["empatia"])
est = af.EstadoAfectivo(perfil="no-existe")
comprobar("perfil desconocido -> OPTIMISTIC", est.perfil == "OPTIMISTIC")
comprobar("solo OPTIMISTIC y EMPATHETIC",
          set(af.PERFILES) == {"OPTIMISTIC", "EMPATHETIC"})
espejo = af.EstadoAfectivo(perfil="EMPATHETIC")
espejo.empatizar(af.ANCLAJES["tristeza"], 1.0)
comprobar("EMPATHETIC copia al usuario", espejo.vector == af.ANCLAJES["tristeza"])
defecto = af.EstadoAfectivo()
comprobar("sin perfil -> mismos valores que antes",
          defecto.empatia == af.EMPATIA and defecto.animo_base == af.ANIMO_BASE)

print()
print("=" * 70)
print("12. AEB")
print("=" * 70)
from romeo_brain import aeb  # noqa: E402

e = af.EstadoAfectivo()
p = aeb.mapear(e)
comprobar("en la base -> neutral, sin tono ni movimiento",
          p["estado"] == "neutral_L0" and p["tono"] is None and p["movimiento"] is None,
          f"-> {p['estado']}")
claves = {"estado", "emocion", "nivel", "cara", "tono", "movimiento", "musica"}
comprobar("devuelve todas las claves del contrato", set(p) == claves)
for _ in range(4):
    v, c = af.percibir_del_texto(*ter.analizar("estoy muy triste y destrozado"))
    e.empatizar(v, c)
p = aeb.mapear(e)
comprobar("usuario triste insistente -> tristeza", p["emocion"] == "tristeza"
          and p["cara"] == "sad" and p["tono"], f"-> {p['estado']}")
comprobar("tristeza NUNCA mueve el carrito", p["movimiento"] is None)
comprobar("la cara del AEB coincide con estado.cara()", p["cara"] == e.cara())
niveles = []
e = af.EstadoAfectivo()
for _ in range(8):
    e.empatizar(af.ANCLAJES["felicidad"], 1.0)
    niveles.append(aeb.mapear(e)["nivel"])
comprobar("alegria insistente sube de nivel sin bajar",
          niveles == sorted(niveles) and niveles[-1] >= 3, f"-> {niveles}")
p = aeb.mapear(e)
comprobar("alegria alta -> gesto fisico", p["movimiento"] in ("RR", "ROTATE360"),
          f"-> {p['estado']} {p['movimiento']}")
e = af.EstadoAfectivo()
for _ in range(6):
    e.empatizar(af.ANCLAJES["enojo"], 1.0)
p = aeb.mapear(e)
comprobar("enojo -> cara angry, tono calmado, sin movimiento",
          p["cara"] == "angry" and p["movimiento"] is None and "calma" in p["tono"],
          f"-> {p['estado']}")

print()
print("=" * 70)
if fallos:
    print(f"  {len(fallos)} FALLOS:")
    for f in fallos:
        print(f"    - {f}")
    sys.exit(1)
print("  todo correcto")
print("=" * 70)
