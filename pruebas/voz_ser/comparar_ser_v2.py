#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""comparar_ser_v2.py -- ronda 2: candidatos para REEMPLAZAR al SER actual de
ROMEO (dusha, HuBERT afinado en ruso), todos en las MISMAS condiciones.

Igual que comparar_ser.py: un PROCESO por modelo (la RAM de uno no contamina
la del otro) y las salidas se reducen a las 4 clases de ROMEO (neutral,
tristeza, enojo, felicidad) con `agrupar` de _ser_engine.py.

CAMBIOS RESPECTO A LA RONDA 1
- umuteam: su config declara `CustomAudioClassification` (sin auto_map), asi
  que pipeline() le ponia una cabeza AL AZAR. Aqui se reimplementa la cabeza
  de su toolkit (NLP-UMUTeam/umuteam-speech-emotion, wav2vec2_bert_es.py):
  projector -> media temporal -> dense -> tanh -> out_proj, y se cargan sus
  pesos. Sin trust_remote_code.
- wavlm: _ser_wavlm.py usa `from ._ser_engine import ...`, que solo funciona
  importado como paquete. Aqui se importa `romeo_drivers._ser_wavlm`.
- --normalizar: variantes de volumen aplicadas IGUAL a todos los modelos, en
  la misma ejecucion (el modelo se carga una vez y corre cada variante).
- Metricas: exactitud con IC de Wilson 95 %, UAR, F1 macro, F1 y recall por
  clase, matriz de confusion 4x4 y McNemar exacta (pareada) contra el modelo
  de referencia (--referencia, por defecto e2v_base) y contra dusha.

USO
    python comparar_ser_v2.py --repo C:\\ruta\\ARFORD --audios audios_r2
    python comparar_ser_v2.py --repo ... --audios ... --normalizar ninguna rms
    python comparar_ser_v2.py --repo ... --audios ... --modelos dusha umuteam

--audios  carpeta con subcarpetas actuados/ y conversacion/ (o .wav sueltos).
          La clase esperada sale del NOMBRE del archivo (triste, feliz,
          enojado, neutral...). Los que no la dicen solo reportan prediccion.
"""
import argparse
import csv
import json
import math
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
    "e2v_base":  ("e2v",   "emotion2vec/emotion2vec_plus_base"),
    "umuteam":   ("umu",   "UMUTeam/w2v-bert-emotion-es"),
    "wavlm":     ("wavlm", "3loi/SER-Odyssey-Baseline-WavLM-Categorical"),
    "e2v_large": ("e2v",   "emotion2vec/emotion2vec_plus_large"),   # opcional
    "pollito_v5": ("pollito", "pollitoconpapass/superb-ser-finetuned-spanish-v5"),
    # pollito_v7 (pollitoconpapass/ser-spanish-model-v7) EXCLUIDO: su ficha no
    # declara licencia (revisado 19-11-2026). Regla: licencia no clara = fuera.
}
# pollito_v5: config.json solo trae LABEL_0..LABEL_5. La traduccion sale de
# EMOTIONS_DICT de su README (miedo, triste, neutral, enojo, disgusto, feliz),
# escrita con sinonimos que _ETIQUETAS ya conoce (triste/feliz/disgusto no
# estan). Mismo criterio que ROMEO: miedo -> tristeza, asco -> enojo.
POLLITO_ETIQUETAS = {"LABEL_0": "miedo", "LABEL_1": "tristeza", "LABEL_2": "neutral",
                     "LABEL_3": "enojo", "LABEL_4": "asco", "LABEL_5": "felicidad"}
POR_DEFECTO = ["dusha", "e2v_base", "umuteam", "wavlm", "pollito_v5"]
ACTUAL = "dusha"                      # el modelo que se quiere reemplazar

CLASES = ["neutral", "tristeza", "enojo", "felicidad"]
PISTAS = [("triste", "tristeza"), ("sad", "tristeza"),
          ("feliz", "felicidad"), ("alegre", "felicidad"), ("happy", "felicidad"),
          ("enoj", "enojo"), ("enfad", "enojo"), ("angry", "enojo"),
          ("neutral", "neutral"), ("neutro", "neutral")]
SR = 16000
N_CHICO = 60
DURACIONES = (2, 5, 10)               # cubetas de latencia (+-1 s)
RMS_OBJETIVO_DB = -20.0               # para --normalizar rms
PICO_OBJETIVO = 0.9                   # para --normalizar pico (~ -1 dBFS)


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


def normalizar(x, modo):
    """Misma transformacion de volumen para todos los modelos."""
    import numpy as np
    if modo == "ninguna":
        return x
    if modo == "pico":
        p = float(np.max(np.abs(x)))
        return x if p < 1e-6 else (x * (PICO_OBJETIVO / p)).astype("float32")
    if modo == "rms":
        r = float(np.sqrt(np.mean(x ** 2)))
        if r < 1e-6:
            return x
        y = x * (10 ** (RMS_OBJETIVO_DB / 20) / r)
        return np.clip(y, -1.0, 1.0).astype("float32")   # recorta picos
    raise ValueError(modo)


def esperado_de(nombre):
    n = nombre.lower()
    for pista, clase in PISTAS:
        if pista in n:
            return clase
    return None


def listar_audios(carpeta):
    """[(ruta, subconjunto)] ordenados. subconjunto = nombre de la subcarpeta."""
    raiz = Path(carpeta)
    out = []
    for r in sorted(raiz.rglob("*.wav")):
        sub = r.parent.relative_to(raiz).as_posix() or "."
        out.append((r, sub))
    return out


# ------------------------------------------------------ envoltorio emotion2vec
class ClasificadorE2V:
    """emotion2vec+ via FunASR, con la misma interfaz que ClasificadorVoz."""

    def __init__(self, modelo, hilos):
        self.modelo, self.hilos = modelo, hilos
        self._m = None
        self.error = None

    def cargar(self):
        try:
            from funasr import AutoModel
            kw = dict(disable_update=True, ncpu=self.hilos, device="cpu")
            try:
                self._m = AutoModel(model=self.modelo, hub="hf", **kw)
            except Exception:
                self._m = AutoModel(model=self.modelo.replace("emotion2vec/", "iic/"),
                                    hub="ms", **kw)
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
        return agrupar([{"label": l, "score": float(s)}
                        for l, s in zip(r["labels"], r["scores"])])


# ------------------------------------------------------ envoltorio UMUTeam
class ClasificadorUMU:
    """UMUTeam/w2v-bert-emotion-es con SU cabeza (CustomAudioClassification).

    Reimplementa ClassificationHead + forward de
    NLP-UMUTeam/umuteam-speech-emotion/src/speech_emotion/models/wav2vec2_bert_es.py
    y replica el preprocesado de su inference.get_w2vbert_emotion().
    """

    def __init__(self, modelo, hilos):
        self.modelo, self.hilos = modelo, hilos
        self._p = None
        self.error = None

    def cargar(self):
        try:
            import torch
            import torch.nn as nn
            from huggingface_hub import hf_hub_download
            from safetensors import safe_open
            from transformers import AutoConfig, AutoFeatureExtractor, Wav2Vec2BertModel

            cfg = AutoConfig.from_pretrained(self.modelo)
            if cfg.use_weighted_layer_sum:
                raise RuntimeError("use_weighted_layer_sum=True no implementado")
            fe = AutoFeatureExtractor.from_pretrained(self.modelo)
            base, info = Wav2Vec2BertModel.from_pretrained(self.modelo,
                                                           output_loading_info=True)
            if info.get("missing_keys"):
                raise RuntimeError(f"pesos faltantes en el backbone: {info['missing_keys'][:5]}")
            base.eval()

            n_lab = len(cfg.id2label)
            proj = nn.Linear(cfg.hidden_size, cfg.classifier_proj_size)
            dense = nn.Linear(768, 768)
            out = nn.Linear(768, n_lab)
            ruta = hf_hub_download(self.modelo, "model.safetensors")
            with safe_open(ruta, "pt") as f:
                proj.load_state_dict({"weight": f.get_tensor("projector.weight"),
                                      "bias": f.get_tensor("projector.bias")})
                dense.load_state_dict({"weight": f.get_tensor("classifier.dense.weight"),
                                       "bias": f.get_tensor("classifier.dense.bias")})
                out.load_state_dict({"weight": f.get_tensor("classifier.out_proj.weight"),
                                     "bias": f.get_tensor("classifier.out_proj.bias")})
            etiquetas = [cfg.id2label[i] for i in range(n_lab)]
            self._p = (torch, fe, base, proj.eval(), dense.eval(), out.eval(), etiquetas)
            return True
        except Exception as e:  # noqa: BLE001
            self.error = str(e)[:300]
            return False

    def clasificar(self, samples, sample_rate=SR):
        from _ser_engine import agrupar
        torch, fe, base, proj, dense, out, etiquetas = self._p
        ent = fe(samples, sampling_rate=SR, max_length=16000, truncation=True,
                 return_tensors="pt")
        with torch.no_grad():
            h = base(ent["input_features"],
                     attention_mask=ent.get("attention_mask")).last_hidden_state
            h = proj(h).mean(dim=1)                  # una sola frase: sin relleno
            logits = out(torch.tanh(dense(h)))[0]    # dropout = identidad en eval
            p = torch.softmax(logits, dim=-1).numpy()
        return agrupar([{"label": e, "score": float(v)} for e, v in zip(etiquetas, p)])


# ----------------------------------------------------------------- metricas
class ClasificadorPollito:
    """pollitoconpapass/superb-ser-finetuned-spanish-v5, como en su ficha:
    Wav2Vec2FeatureExtractor + Wav2Vec2ForSequenceClassification, softmax."""

    def __init__(self, modelo, hilos):
        self.modelo, self.hilos, self.error = modelo, hilos, None

    def cargar(self):
        try:
            import torch
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
            torch.set_num_threads(self.hilos)
            self.fe = Wav2Vec2FeatureExtractor.from_pretrained(self.modelo)
            self.m = Wav2Vec2ForSequenceClassification.from_pretrained(self.modelo).eval()
            self.id2label = self.m.config.id2label
            return True
        except Exception as e:  # noqa: BLE001
            self.error = str(e)[:300]
            return False

    def clasificar(self, x, sr):
        import torch
        from _ser_engine import agrupar
        with torch.inference_mode():
            ent = self.fe(x, sampling_rate=sr, padding=True, return_tensors="pt")
            p = torch.softmax(self.m(**ent).logits, dim=-1)[0].tolist()
        return agrupar([{"label": POLLITO_ETIQUETAS[self.id2label[i]], "score": float(v)}
                        for i, v in enumerate(p)])


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - m), min(1.0, c + m))


def mcnemar_exacta(b, c):
    """p bilateral exacto (binomial) con b, c = discordantes."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def metricas(filas):
    """Sobre filas con 'esperado'. pred None cuenta como error."""
    con = [f for f in filas if f["esperado"]]
    n = len(con)
    conf = {e: {p: 0 for p in CLASES + ["(ninguna)"]} for e in CLASES}
    for f in con:
        conf[f["esperado"]][f["pred"] or "(ninguna)"] += 1
    k = sum(conf[c][c] for c in CLASES)
    por_clase = {}
    for c in CLASES:
        tp = conf[c][c]
        sop = sum(conf[c].values())
        predc = sum(conf[e][c] for e in CLASES)
        rec = tp / sop if sop else None
        pre = tp / predc if predc else 0.0
        f1 = (2 * pre * rec / (pre + rec)) if rec and pre else 0.0
        por_clase[c] = {"soporte": sop, "recall": rec, "precision": pre,
                        "f1": f1 if sop else None}
    presentes = [c for c in CLASES if por_clase[c]["soporte"]]
    uar = statistics.mean(por_clase[c]["recall"] for c in presentes) if presentes else None
    f1m = statistics.mean(por_clase[c]["f1"] for c in presentes) if presentes else None
    lo, hi = wilson(k, n)
    return {"n": n, "aciertos": k, "exactitud": k / n if n else None,
            "ic95": [lo, hi], "uar": uar, "f1_macro": f1m,
            "por_clase": por_clase, "confusion": conf}


def comparar_pareado(fa, fb):
    """McNemar entre dos modelos sobre los MISMOS audios con etiqueta."""
    ia = {f["ruta"]: f for f in fa if f["esperado"]}
    ib = {f["ruta"]: f for f in fb if f["esperado"]}
    b = c = 0
    for r in ia.keys() & ib.keys():
        oa = ia[r]["pred"] == ia[r]["esperado"]
        ob = ib[r]["pred"] == ib[r]["esperado"]
        b += oa and not ob
        c += ob and not oa
    return {"solo_a": b, "solo_b": c, "p": mcnemar_exacta(b, c)}


# --------------------------------------------------------------- trabajador
def trabajador(nombre, carpeta, repo, hilos, repeticiones, variantes):
    """Corre UN modelo (todas las variantes) y escribe 'RESULTADO:{json}'."""
    import numpy as np
    import psutil

    pkg = Path(repo) / "ros2_ws" / "src" / "romeo_drivers"
    sys.path.insert(0, str(pkg / "romeo_drivers"))   # _ser_engine suelto
    sys.path.insert(0, str(pkg))                     # paquete romeo_drivers
    os.environ.setdefault("ROMEO_SER", "1")
    os.environ["ROMEO_SER_HILOS"] = str(hilos)
    import torch
    torch.set_num_threads(hilos)

    proc = psutil.Process()
    mb = lambda: proc.memory_info().rss / 1e6  # noqa: E731

    tipo, ident = MODELOS[nombre]
    ram0 = mb()
    t0 = time.perf_counter()
    if tipo == "hf":
        from _ser_engine import ClasificadorVoz
        clf = ClasificadorVoz(ident)
    elif tipo == "wavlm":
        from romeo_drivers._ser_wavlm import ClasificadorWavLM
        clf = ClasificadorWavLM(ident, hilos)
    elif tipo == "umu":
        clf = ClasificadorUMU(ident, hilos)
    elif tipo == "pollito":
        clf = ClasificadorPollito(ident, hilos)
    else:
        clf = ClasificadorE2V(ident, hilos)
    ok = clf.cargar()
    t_carga = time.perf_counter() - t0
    if not ok:
        print("RESULTADO:" + json.dumps({"modelo": nombre, "id": ident,
              "error": getattr(clf, "error", "no cargo")}))
        return
    torch.set_num_threads(hilos)          # por si la carga lo cambio
    ram_carga = mb()

    audios = listar_audios(carpeta)
    ram_pico = ram_carga
    if audios:                            # calentamiento
        clf.clasificar(cargar_audio(audios[0][0])[0], SR)
    filas = []
    for var in variantes:
        for ruta, sub in audios:
            x, dur = cargar_audio(ruta)
            x = normalizar(x, var)
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
            filas.append({"variante": var, "subconjunto": sub,
                          "ruta": Path(ruta).relative_to(carpeta).as_posix(),
                          "audio": Path(ruta).name, "dur_s": round(dur, 2),
                          "esperado": esperado_de(Path(ruta).name), "pred": pred,
                          "prob": None if prob is None else round(prob, 3),
                          "dist": dist, "lat_s": round(lat, 3),
                          "rtf": round(lat / dur, 3) if dur else None})
    print("RESULTADO:" + json.dumps({
        "modelo": nombre, "id": ident, "hilos": torch.get_num_threads(),
        "carga_s": round(t_carga, 1), "ram_base_mb": round(ram0),
        "ram_carga_mb": round(ram_carga), "ram_pico_mb": round(ram_pico),
        "filas": filas}))


# ------------------------------------------------------------- impresion
def pct(v):
    return "  -  " if v is None else f"{100 * v:5.1f}"


def imprimir_resumen(resultados, variantes, referencia):
    ok = [r for r in resultados if "error" not in r]
    print("\n\n======================= RESUMEN =======================")
    for r in resultados:
        if "error" in r:
            print(f"  {r['modelo']}: ERROR -> {str(r['error'])[:120]}")

    print("\n-- Recursos (una carga por modelo) --")
    print(f"{'modelo':10} {'carga s':>8} {'RAM MB':>8} {'pico MB':>8} {'hilos':>6}")
    for r in ok:
        print(f"{r['modelo']:10} {r['carga_s']:>8} {r['ram_carga_mb']:>8}"
              f" {r['ram_pico_mb']:>8} {r['hilos']:>6}")

    print("\n-- Latencia mediana (s) y RTF por duracion del audio (variante "
          f"'{variantes[0]}') --")
    cab = f"{'modelo':10} {'todas':>12}" + "".join(f" {str(d) + ' s':>12}" for d in DURACIONES)
    print(cab)
    for r in ok:
        fs = [f for f in r["filas"] if f["variante"] == variantes[0]]

        def celda(sel):
            if not sel:
                return f"{'-':>12}"
            return f"{statistics.median(f['lat_s'] for f in sel):>6.3f}/{statistics.median(f['rtf'] for f in sel):<5.2f}"
        linea = f"{r['modelo']:10} {celda(fs):>12}"
        for d in DURACIONES:
            linea += " " + celda([f for f in fs if abs(f["dur_s"] - d) <= 1.0])
        print(linea)

    for var in variantes:
        print(f"\n================ VARIANTE: normalizar={var} ================")
        por_mod = {r["modelo"]: [f for f in r["filas"] if f["variante"] == var] for r in ok}
        n = max((metricas(fs)["n"] for fs in por_mod.values()), default=0)
        if n < N_CHICO:
            print(f"  AVISO: solo {n} audios con etiqueta (< {N_CHICO}). Resultado ORIENTATIVO;"
                  " los IC son anchos.")
        print(f"{'modelo':10} {'aciertos':>9} {'exact %':>8} {'IC95 %':>13} {'UAR %':>6}"
              f" {'F1m %':>6} | " + " ".join(f"{'R ' + c[:4]:>6}" for c in CLASES)
              + f" | {'vs ' + ACTUAL:>14} | {'vs ' + referencia:>14}")
        for nom, fs in por_mod.items():
            m = metricas(fs)
            lo, hi = m["ic95"]
            ic = f"{100 * lo:4.0f}-{100 * hi:<4.0f}" if m["n"] else "-"

            def vs(otro):
                if otro == nom or otro not in por_mod:
                    return f"{'(ref)' if otro == nom else '-':>14}"
                c = comparar_pareado(fs, por_mod[otro])
                return f"{c['solo_a']:>2}/{c['solo_b']:<2} p={c['p']:<5.3f}"
            print(f"{nom:10} {m['aciertos']:>4}/{m['n']:<4} {pct(m['exactitud']):>8} {ic:>13}"
                  f" {pct(m['uar']):>6} {pct(m['f1_macro']):>6} | "
                  + " ".join(f"{pct(m['por_clase'][c]['recall']):>6}" for c in CLASES)
                  + f" | {vs(ACTUAL)} | {vs(referencia)}")
        print("  R xxxx = recall por clase. 'vs X: a/b p' = audios que SOLO acierta este"
              " modelo / que SOLO acierta X, y p de McNemar exacta (p<0,05 = diferencia"
              " significativa).")
        for nom, fs in por_mod.items():
            m = metricas(fs)
            if not m["n"]:
                continue
            print(f"\n  {nom} - matriz de confusion (filas = actuado, columnas = predicho)"
                  f"  F1: " + ", ".join(f"{c} {pct(m['por_clase'][c]['f1']).strip()}" for c in CLASES))
            print("    " + f"{'':10}" + "".join(f"{c[:9]:>10}" for c in CLASES + ["(ninguna)"]))
            for e in CLASES:
                if sum(m["confusion"][e].values()):
                    print("    " + f"{e:10}" + "".join(f"{m['confusion'][e][p]:>10}"
                                                       for p in CLASES + ["(ninguna)"]))
        sin = {nom: [f for f in fs if not f["esperado"]] for nom, fs in por_mod.items()}
        if any(sin.values()):
            print("\n  Audios SIN etiqueta (solo prediccion):")
            for nom, fs in sin.items():
                cuenta = {}
                for f in fs:
                    cuenta[f["pred"]] = cuenta.get(f["pred"], 0) + 1
                print(f"    {nom:10} " + ", ".join(f"{k}: {v}" for k, v in sorted(cuenta.items(), key=str)))


# ------------------------------------------------------------- orquestador
def principal():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="raiz del repo ARFORD")
    ap.add_argument("--audios", required=True, help="carpeta (actuados/, conversacion/)")
    ap.add_argument("--modelos", nargs="+", default=POR_DEFECTO, choices=list(MODELOS))
    ap.add_argument("--normalizar", nargs="+", default=["ninguna", "rms"],
                    choices=["ninguna", "pico", "rms"],
                    help="una o varias variantes, se reportan por separado")
    ap.add_argument("--referencia", default="e2v_base",
                    help="modelo contra el que se hace McNemar (ademas de dusha)")
    ap.add_argument("--hilos", type=int, default=4)
    ap.add_argument("--repeticiones", type=int, default=3)
    ap.add_argument("--salida", default="resultados_ser_v2", help="prefijo de los archivos")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.worker:
        trabajador(a.worker, a.audios, a.repo, a.hilos, a.repeticiones, a.normalizar)
        return

    audios = listar_audios(a.audios)
    if not audios:
        sys.exit(f"No hay .wav en {a.audios}")
    subs = {}
    for _, s in audios:
        subs[s] = subs.get(s, 0) + 1
    print(f"Audios: {len(audios)} ({', '.join(f'{k}: {v}' for k, v in subs.items())})"
          f" | variantes: {a.normalizar} | hilos: {a.hilos} | repeticiones: {a.repeticiones}")

    env = dict(os.environ, OMP_NUM_THREADS=str(a.hilos), MKL_NUM_THREADS=str(a.hilos),
               PYTHONIOENCODING="utf-8")
    resultados = []
    for nombre in a.modelos:
        print(f"\n=== {nombre}: {MODELOS[nombre][1]} ===", flush=True)
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", nombre,
               "--repo", a.repo, "--audios", a.audios, "--hilos", str(a.hilos),
               "--repeticiones", str(a.repeticiones), "--normalizar", *a.normalizar]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, encoding="utf-8", env=env)
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
              f" | RAM pico {r['ram_pico_mb']} MB | hilos {r['hilos']}")
        for f in r["filas"]:
            print(f"  [{f['variante']:7}] {f['ruta'][:40]:40} esperado={str(f['esperado']):10}"
                  f" pred={str(f['pred']):10} p={f['prob']}  {f['lat_s']} s")
        if r["filas"] and all(f["pred"] is None for f in r["filas"]):
            print("  AVISO: ninguna etiqueta del modelo se reconocio (revisa _ETIQUETAS).")

    imprimir_resumen(resultados, a.normalizar, a.referencia)

    # ---- salidas
    ok = [r for r in resultados if "error" not in r]
    resumen = {}
    for var in a.normalizar:
        por_mod = {r["modelo"]: [f for f in r["filas"] if f["variante"] == var] for r in ok}
        resumen[var] = {}
        for nom, fs in por_mod.items():
            m = metricas(fs)
            m["mcnemar"] = {o: comparar_pareado(fs, por_mod[o])
                            for o in {ACTUAL, a.referencia} if o in por_mod and o != nom}
            resumen[var][nom] = m
    with open(a.salida + ".json", "w", encoding="utf-8") as fh:
        json.dump({"config": {"modelos": a.modelos, "variantes": a.normalizar,
                              "hilos": a.hilos, "repeticiones": a.repeticiones,
                              "audios": a.audios, "n_audios": len(audios)},
                   "resultados": resultados, "metricas": resumen},
                  fh, ensure_ascii=False, indent=2)
    with open(a.salida + ".csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["modelo", "variante", "subconjunto", "audio", "dur_s", "esperado",
                    "pred", "acierto", "prob", "lat_s", "rtf", "dist"])
        for r in ok:
            for f in r["filas"]:
                w.writerow([r["modelo"], f["variante"], f["subconjunto"], f["audio"],
                            f["dur_s"], f["esperado"], f["pred"],
                            "" if not f["esperado"] else int(f["pred"] == f["esperado"]),
                            f["prob"], f["lat_s"], f["rtf"],
                            json.dumps(f["dist"], ensure_ascii=False)])
    print(f"\nGuardado: {a.salida}.csv y {a.salida}.json")


if __name__ == "__main__":
    principal()
