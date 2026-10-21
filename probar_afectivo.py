# -*- coding: utf-8 -*-
"""Prueba la capa afectiva (PULSE / ROSE / AEB) sin robot ni camara.

Reproduce el caso de estudio del paper de Rocha et al. con el robot FRED:
los MISMOS estimulos contra dos perfiles distintos, para ver que el factor
de empatia cambia de verdad como reacciona el robot.

    python probar_afectivo.py
"""
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "ros2_ws", "src", "romeo_brain"))

from romeo_brain import estado_afectivo as af  # noqa: E402

ETIQUETAS = ["neutral", "felicidad", "sorpresa", "tristeza",
             "enojo", "asco", "miedo", "desprecio"]


def fmt(v):
    return f"[{v[0]:+.2f}, {v[1]:+.2f}, {v[2]:+.2f}]"


# ══════════════════════════════════════════════════════════════════
print("=" * 68)
print("1. PULSE -- confianza por entropia de Shannon")
print("=" * 68)
casos = [
    ("FER segurisimo (95% tristeza)",
     [0.01, 0.01, 0.01, 0.95, 0.01, 0.005, 0.005, 0.0]),
    ("FER dudando (40/35/25)",
     [0.25, 0.0, 0.0, 0.40, 0.35, 0.0, 0.0, 0.0]),
    ("FER perdido (todo igual)",
     [0.125] * 8),
]
for nombre, probs in casos:
    c = af.confianza(probs)
    vad = af.a_vad(probs, ETIQUETAS)
    print(f"  {nombre:32s} confianza={c:.3f}  VAD={fmt(vad)}")

print()
print("  Lo que importa: la lectura dudosa pesa menos en la fusion que la")
print("  segura, sin que nadie tenga que ponerle un umbral a mano.")

# ══════════════════════════════════════════════════════════════════
print()
print("=" * 68)
print("2. PULSE -- la distribucion ENTERA, no solo la ganadora")
print("=" * 68)
mezcla = [0.40, 0.0, 0.0, 0.60, 0.0, 0.0, 0.0, 0.0]
solo_gana = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
print(f"  60% triste + 40% neutral -> VAD={fmt(af.a_vad(mezcla, ETIQUETAS))}")
print(f"  solo 'tristeza' (argmax)  -> VAD={fmt(af.a_vad(solo_gana, ETIQUETAS))}")
print()
print("  La mezcla cae a medio camino: 'algo triste' y 'muy triste' dejan")
print("  de ser lo mismo. Con solo el argmax los dos casos serian iguales.")

# ══════════════════════════════════════════════════════════════════
print()
print("=" * 68)
print("3. ROSE -- el caso de estudio de FRED: dos perfiles, mismo estimulo")
print("=" * 68)

# Usuario triste, como en el paper.
usuario_triste = af.ANCLAJES["tristeza"]
print(f"  Usuario triste: VAD={fmt(usuario_triste)}")
print()

for nombre, empatia, base in [
    ("EMPATICO PURO  (e=1.0, base neutra)", 1.0, (0.0, 0.0, 0.0)),
    ("OPTIMISTA      (e=0.4, base positiva)", 0.4, (0.40, 0.30, 0.20)),
]:
    est = af.EstadoAfectivo(empatia=empatia, animo_base=base)
    est.empatizar(usuario_triste)
    zona = ("negativa" if est.vector[0] < -0.05
            else "neutra" if est.vector[0] < 0.05 else "positiva")
    print(f"  {nombre}")
    print(f"      Romeo queda en {fmt(est.vector)}   (valencia {zona})")

print()
print("  El empatico se hunde con el usuario; el optimista se entristece")
print("  un poco pero se queda arriba, y desde ahi puede animar. Para un")
print("  robot de acompañamiento, el segundo es el que sirve.")

# ══════════════════════════════════════════════════════════════════
print()
print("=" * 68)
print("4. ROSE -- el estado PERSISTE y decae solo")
print("=" * 68)
est = af.EstadoAfectivo()          # perfil por defecto: e=0.4
print(f"  arranque            {fmt(est.vector)}")
est.empatizar(af.ANCLAJES["tristeza"])
print(f"  usuario triste      {fmt(est.vector)}")
est.empatizar(af.ANCLAJES["tristeza"])
print(f"  sigue triste        {fmt(est.vector)}"
      f"   <- se acumula")
# Decaimiento: se simula el paso del tiempo pasandole el instante a
# decaer(). El primer paso cae al terminar la latencia, y despues uno cada
# `paso` segundos.
t0 = est._ultimo_estimulo
for i in range(1, 4):
    ahora = t0 + est.latencia + (i - 1) * est.paso
    est.decaer(ahora=ahora)
    print(f"  pasa el tiempo ({i})  {fmt(est.vector)}   t={ahora - t0:.0f}s")
est2 = af.EstadoAfectivo()
est2.empatizar(af.ANCLAJES["tristeza"])
est2.decaer(ahora=est2._ultimo_estimulo + 600)
print(f"  10 min de silencio  {fmt(est2.vector)}   <- vuelve a la base")
print()
print("  Hoy cada turno empieza de cero. Con esto, dos mensajes tristes")
print("  seguidos pesan mas que uno, y el efecto se diluye solo.")

# ══════════════════════════════════════════════════════════════════
print()
print()
