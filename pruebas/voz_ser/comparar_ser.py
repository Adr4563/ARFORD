#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""comparar_ser.py -- compara modelos de emocion en la VOZ (SER) para ROMEO.

Corre cada modelo en un PROCESO APARTE (asi la RAM de uno no contamina la
del siguiente) sobre los mismos audios, y reporta por modelo:
    - tiempo de carga y RAM (despues de cargar y pico durante la inferencia)
    - latencia por audio (mediana de N repeticiones) y su razon con la
      duracion del audio (RTF; < 1 = mas rapido que tiempo real)
    - clase predicha y probabilidad por audio, ya reducida a las 4 clases de
      ROMEO (neutral, tristeza, enojo, felicidad) con la MISMA funcion
      `agrupar` del repo
    - aciertos, si el nombre del archivo dice la emocion esperada
      (triste, feliz/alegre, enojado/enfadado, neutral)

USO (desde cualquier carpeta, en la PC de inferencia):
    pip install transformers torch soundfile psutil
    pip install funasr            # solo para los modelos emotion2vec

    python comparar_ser.py --repo C:\\ruta\\ARFORD --audios C:\\ruta\\audios
    python comparar_ser.py --repo ... --audios ... --modelos umuteam dusha
    python comparar_ser.py --repo ... --audios ... --hilos 4 --repeticiones 5

--repo     raiz del repo ARFORD (de ahi se importan _ser_engine y
           _ser_wavlm, para medir EXACTAMENTE el mismo codigo del robot).
--audios   carpeta con .wav (cualquier frecuencia; se pasa a 16 kHz).
           Pon los MISMOS audios que ya usaste con el microfono de la placa.

NOTAS
- La primera vez cada modelo se DESCARGA (de 0,4 a ~2,4 GB). No cuentes ese
  tiempo: el script mide la carga con el modelo ya en cache si lo corres
  dos veces.
- Cierra otros programas pesados para que la RAM medida sea fiable.
- Los nombres de modelos en Hugging Face y la API de FunASR se tomaron de las
  tarjetas publicas; si alguno falla, el script lo dice y sigue con el resto.
"""
import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# nombre corto -> (tipo, id del modelo)
MODELOS = {
    "dusha":     ("hf",    "xbgoose/hubert-large-speech-emotion-recognition-russian-dusha-finetuned"),
    "umuteam":   ("hf",    "UMUTeam/w2v-bert-emotion-es"),
    "wavlm":     ("wavlm", "3loi/SER-Odyssey-Baseline-WavLM-Categorical"),
    "e2v_base":  ("e2v",   "emotion2vec/emotion2vec_plus_base"),
    "e2v_large": ("e2v",   "emotion2vec/emotion2vec_plus_large"),
}
POR_DEFECTO = ["dusha", "umuteam", "e2v_base", "e2v_large", "wavlm"]

# Pista en el nombre del archivo -> clase esperada (claves de ROMEO).
PISTAS = [("triste", "tristeza"), ("sad", "tristeza"),
          ("feliz", "felicidad"), ("alegre", "felicidad"), ("happy", "felicidad"),
          ("enoj", "enojo"), ("enfad", "enojo"), ("angry", "enojo"),
          ("neutral", "neutral"), ("neutro", "neutral")]

SR = 16000


# ----------------------------------------------------------------- audio
def cargar_audio(ruta):
    """(muestras float32 mono a 16 kHz, duracion en s)."""
    import numpy as np
    import soundfile as sf
    x, sr = sf.read(str(ruta), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        n = int(x.size * SR / sr)
        x = np.interp(np.linspace(0, x.size - 1, n), np.arange(x.size), x)
    x = x.astype("float32")
    return x, x.size / SR


def esperado_de(nombre):
    n = nombre.lower()
    for pista, clase in PISTAS:
        if pista in n:
            return clase
    return None


# ------------------------------------------------------ envoltorio emotion2vec
class ClasificadorE2V:
    """emotion2vec+ via FunASR, con la misma interfaz que ClasificadorVoz."""

    def __init__(self, modelo):
        self.modelo = modelo
        self._m = None
        self.error = None

    def cargar(self):
        try:
            from funasr import AutoModel
            try:
                self._m = AutoModel(model=self.modelo, hub="hf", disable_update=True)
            except Exception:
                # Alternativa: el mismo modelo en ModelScope ("iic/...")
                self._m = AutoModel(model=self.modelo.replace("emotion2vec/", "iic/"),
                                    hub="ms", disable_update=True)
            return True
        except Exception as e:  # noqa: BLE001
            self.error = str(e)[:300]
            return False

    def clasificar(self, samples, sample_rate=SR):
        import soundfile as sf
        from _ser_engine import agrupar
        with tempfile.TemporaryDirectory() as d:
            ruta = os.path.join(d, "a.wav")
            sf.write(ruta, samples, SR)
            res = self._m.generate(ruta, granularity="utterance",
                                   extract_embedding=False)
        r = res[0]
        salida = [{"label": l, "score": float(s)}
                  for l, s in zip(r["labels"], r["scores"])]
        return agrupar(salida)


# --------------------------------------------------------------- trabajador
def trabajador(nombre, carpeta, repo, hilos, repeticiones):
    """Corre UN modelo y escribe una linea 'RESULTADO:{json}' en stdout."""
    import numpy as np
    import psutil

    sys.path.insert(0, str(Path(repo) / "ros2_ws" / "src" / "romeo_drivers" / "romeo_drivers"))
    os.environ.setdefault("ROMEO_SER", "1")
    os.environ["ROMEO_SER_HILOS"] = str(hilos)
    try:
        import torch
        torch.set_num_threads(hilos)
    except Exception:  # noqa: BLE001
        pass

    proc = psutil.Process()
    mb = lambda: proc.memory_info().rss / 1e6  # noqa: E731

    tipo, ident = MODELOS[nombre]
    ram0 = mb()
    t0 = time.perf_counter()
    if tipo == "hf":
        from _ser_engine import ClasificadorVoz
        clf = ClasificadorVoz(ident)
    elif tipo == "wavlm":
        from _ser_wavlm import ClasificadorWavLM
        clf = ClasificadorWavLM(ident, hilos)
    else:
        clf = ClasificadorE2V(ident)
    ok = clf.cargar()
    t_carga = time.perf_counter() - t0
    if not ok:
        print("RESULTADO:" + json.dumps({"modelo": nombre, "id": ident,
              "error": getattr(clf, "error", "no cargo")}))
        return
    ram_carga = mb()

    audios = sorted(Path(carpeta).glob("*.wav"))
    filas = []
    ram_pico = ram_carga
    # calentamiento (la primera inferencia siempre es mas lenta)
    if audios:
        x, _ = cargar_audio(audios[0])
        clf.clasificar(x, SR)
    for ruta in audios:
        x, dur = cargar_audio(ruta)
        tiempos, res = [], None
        for i in range(repeticiones):
            t = time.perf_counter()
            r = clf.clasificar(x, SR)
            tiempos.append(time.perf_counter() - t)
            if i == 0:
                res = r
            ram_pico = max(ram_pico, mb())
        lat = statistics.median(tiempos)
        if res is None:
            pred, prob, dist = None, None, {}
        else:
            etiquetas, probs = res
            dist = {e: round(p, 3) for e, p in zip(etiquetas, probs)}
            k = int(np.argmax(probs))
            pred, prob = etiquetas[k], float(probs[k])
        filas.append({"audio": ruta.name, "dur_s": round(dur, 2),
                      "esperado": esperado_de(ruta.name), "pred": pred,
                      "prob": None if prob is None else round(prob, 3),
                      "dist": dist, "lat_s": round(lat, 3),
                      "rtf": round(lat / dur, 3) if dur else None})
    print("RESULTADO:" + json.dumps({
        "modelo": nombre, "id": ident, "carga_s": round(t_carga, 1),
        "ram_base_mb": round(ram0), "ram_carga_mb": round(ram_carga),
        "ram_pico_mb": round(ram_pico), "filas": filas}))


# ------------------------------------------------------------- orquestador
def principal():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="raiz del repo ARFORD")
    ap.add_argument("--audios", required=True, help="carpeta con .wav")
    ap.add_argument("--modelos", nargs="+", default=POR_DEFECTO, choices=list(MODELOS))
    ap.add_argument("--hilos", type=int, default=4, help="hilos de CPU (default 4)")
    ap.add_argument("--repeticiones", type=int, default=3)
    ap.add_argument("--salida", default="resultados_ser", help="prefijo de los archivos")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.worker:
        trabajador(a.worker, a.audios, a.repo, a.hilos, a.repeticiones)
        return

    if not list(Path(a.audios).glob("*.wav")):
        sys.exit(f"No hay .wav en {a.audios}")

    resultados = []
    for nombre in a.modelos:
        print(f"\n=== {nombre}: {MODELOS[nombre][1]} ===", flush=True)
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", nombre,
               "--repo", a.repo, "--audios", a.audios,
               "--hilos", str(a.hilos), "--repeticiones", str(a.repeticiones)]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        linea = next((l for l in p.stdout.splitlines() if l.startswith("RESULTADO:")), None)
        if linea is None:
            print("  (fallo el proceso; revisa el error de arriba)")
            resultados.append({"modelo": nombre, "id": MODELOS[nombre][1],
                               "error": "el proceso termino sin resultado"})
            continue
        r = json.loads(linea[len("RESULTADO:"):])
        resultados.append(r)
        if "error" in r:
            print("  ERROR al cargar:", r["error"])
            continue
        print(f"  carga {r['carga_s']} s | RAM tras cargar {r['ram_carga_mb']} MB"
              f" | RAM pico {r['ram_pico_mb']} MB")
        for f in r["filas"]:
            print(f"  {f['audio'][:34]:34} esperado={str(f['esperado']):10}"
                  f" pred={str(f['pred']):10} p={f['prob']}  {f['lat_s']} s"
                  f" (RTF {f['rtf']})")
        if r["filas"] and all(f["pred"] is None for f in r["filas"]):
            print("  AVISO: ninguna etiqueta del modelo se reconocio. Revisa el"
                  " id2label de su config.json y agregalo a _ETIQUETAS de"
                  " _ser_engine.py.")

    # ---- resumen
    print("\n\n================ RESUMEN ================")
    cab = f"{'modelo':10} {'carga s':>8} {'RAM MB':>8} {'pico MB':>8} {'lat med s':>10} {'RTF':>6} {'aciertos':>9}"
    print(cab)
    for r in resultados:
        if "error" in r:
            print(f"{r['modelo']:10} ERROR: {str(r['error'])[:80]}")
            continue
        fs = r["filas"]
        lat = statistics.median([f["lat_s"] for f in fs]) if fs else float("nan")
        rtf = statistics.median([f["rtf"] for f in fs if f["rtf"] is not None]) if fs else float("nan")
        con = [f for f in fs if f["esperado"]]
        ac = sum(1 for f in con if f["pred"] == f["esperado"])
        print(f"{r['modelo']:10} {r['carga_s']:>8} {r['ram_carga_mb']:>8} {r['ram_pico_mb']:>8}"
              f" {lat:>10.3f} {rtf:>6.2f} {ac:>4}/{len(con):<4}")

    with open(a.salida + ".json", "w", encoding="utf-8") as fh:
        json.dump(resultados, fh, ensure_ascii=False, indent=2)
    with open(a.salida + ".csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["modelo", "audio", "dur_s", "esperado", "pred", "prob",
                    "lat_s", "rtf", "dist"])
        for r in resultados:
            for f in r.get("filas", []):
                w.writerow([r["modelo"], f["audio"], f["dur_s"], f["esperado"],
                            f["pred"], f["prob"], f["lat_s"], f["rtf"],
                            json.dumps(f["dist"], ensure_ascii=False)])
    print(f"\nGuardado: {a.salida}.csv y {a.salida}.json")


if __name__ == "__main__":
    principal()
