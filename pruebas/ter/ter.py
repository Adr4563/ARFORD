#!/usr/bin/env python3
"""TER: emocion de un TEXTO en espanol con un clasificador neuronal.

    python3 ter.py                      # escribes frases, Enter, Ctrl+C para salir
    python3 ter.py "hoy me siento triste" "gane el partido"

Modelo: pysentimiento/robertuito-emotion-analysis (RoBERTuito, entrenado en
espanol). Es el equivalente en espanol al de la tesis de Rocha
(michellejieli/emotion_text_classifier, solo ingles): 7 clases, con
"others" haciendo de neutral.

Corre con onnxruntime + tokenizers, que ya estan en la Pi por apt: sin torch,
sin transformers y sin venv. El modelo se exporta una vez en la PC con
exportar_ter.py y vive fuera del repo en ~/.romeo/modelos/ter/.

Si encuentra el ter.py del robot (palabras clave), muestra tambien lo que
dice ese, para comparar los dos.
"""

import json
import math
import os
import sys
import time

import numpy as np

DIR = os.environ.get("ROMEO_TER_DIR", os.path.expanduser("~/.romeo/modelos/ter"))
# Etiquetas del modelo -> claves de estado_afectivo.ANCLAJES.
TRADUCCION = {"others": "neutral", "joy": "felicidad", "sadness": "tristeza",
              "anger": "enojo", "surprise": "sorpresa", "disgust": "asco",
              "fear": "miedo"}
# Anclajes VAD de De Bruyne et al. (2020), los de la tesis.
ANCLAJES = {
    "neutral": (0.00, 0.00, 0.00), "felicidad": (0.76, 0.48, 0.35),
    "sorpresa": (0.40, 0.67, -0.13), "tristeza": (-0.63, -0.27, -0.33),
    "enojo": (-0.51, 0.59, 0.25), "asco": (-0.60, 0.35, 0.11),
    "miedo": (-0.64, 0.60, -0.43),
}
AQUI = os.path.dirname(os.path.abspath(__file__))


class Ter:

    def __init__(self, directorio=DIR, hilos=2):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        with open(os.path.join(directorio, "ter.json"), encoding="utf-8") as f:
            id2label = json.load(f)["id2label"]
        self.etiquetas = [TRADUCCION[id2label[str(i)]] for i in range(len(id2label))]
        self._tok = Tokenizer.from_file(os.path.join(directorio, "tokenizer.json"))
        self._tok.enable_truncation(128)  # model_max_length del modelo
        op = ort.SessionOptions()
        op.intra_op_num_threads = hilos
        self._sesion = ort.InferenceSession(os.path.join(directorio, "ter.onnx"), op)

    def clasificar(self, texto):
        """Probabilidades en el orden de self.etiquetas."""
        # RoBERTuito es "uncased": se entreno con todo en minusculas.
        enc = self._tok.encode(texto.lower())
        ids = np.array([enc.ids], dtype=np.int64)
        mask = np.array([enc.attention_mask], dtype=np.int64)
        z = self._sesion.run(None, {"input_ids": ids, "attention_mask": mask})[0][0]
        e = np.exp(z - z.max())
        return e / e.sum()


def confianza(p):
    h = -sum(x * math.log(x) for x in p if x > 0) / math.log(len(p))
    return 1.0 - h


def a_vad(p, etiquetas):
    return tuple(sum(pi * ANCLAJES[e][k] for pi, e in zip(p, etiquetas)) for k in range(3))


def _ter_del_robot():
    """El ter.py de palabras clave de romeo_brain, si esta (repo o Pi)."""
    for rel in ("../../ros2_ws/src/romeo_brain", "../../src/romeo_brain"):
        ruta = os.path.normpath(os.path.join(AQUI, rel))
        if os.path.exists(os.path.join(ruta, "romeo_brain", "ter.py")):
            sys.path.insert(0, ruta)
            try:
                from romeo_brain import ter as ter_robot
                return ter_robot
            except Exception:  # noqa: BLE001 - la comparacion es opcional
                return None
    return None


def mostrar(ter, texto, robot):
    t0 = time.perf_counter()
    p = ter.clasificar(texto)
    ms = (time.perf_counter() - t0) * 1000
    orden = sorted(zip(ter.etiquetas, p), key=lambda kv: -kv[1])[:3]
    v, a, d = a_vad(p, ter.etiquetas)
    print(f"  {', '.join(f'{e} {x:.0%}' for e, x in orden)}\n"
          f"      confianza {confianza(p):.2f}, VAD ({v:+.2f}, {a:+.2f}, {d:+.2f}), {ms:.0f} ms")
    if robot is not None:
        emo, c = robot.analizar(texto)
        print(f"      palabras clave del robot: {emo or 'sin evidencia'} ({c:.2f})")


def main():
    ter = Ter()
    robot = _ter_del_robot()
    frases = sys.argv[1:]
    if frases:
        for f in frases:
            print(f"> {f}")
            mostrar(ter, f, robot)
        return 0
    print("Escribe una frase y Enter (Ctrl+C para salir).", flush=True)
    try:
        while True:
            f = input("> ").strip()
            if f:
                mostrar(ter, f, robot)
    except (KeyboardInterrupt, EOFError):
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
