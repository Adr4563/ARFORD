# -*- coding: utf-8 -*-
"""Deja la PC lista para Romeo con un solo comando:

    python servidores_pc.py

1. Si faltan, baja los modelos a ~/.romeo/modelos (una sola vez):
     - STT: NeMo FastConformer espanol  (asr/sherpa-onnx-nemo-...-es-1424-int8)
     - TTS: voz es_MX-claude-high       (vits-piper-es_MX-claude-high)
2. Arranca el servidor de STT (8082) y el de TTS (8083).
3. Comprueba Ollama, STT y TTS, y dice que IP poner en la Pi.

Ctrl+C apaga los dos servidores.
"""
import os
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request

RAIZ = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(RAIZ, "ros2_ws", "src", "romeo_drivers", "scripts")
MODELOS = os.path.expanduser("~/.romeo/modelos")
BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download"

# (carpeta donde va, archivo que prueba que ya esta, url del .tar.bz2)
DESCARGAS = [
    (os.path.join(MODELOS, "asr"),
     "sherpa-onnx-nemo-fast-conformer-transducer-es-1424-int8/encoder.int8.onnx",
     f"{BASE}/asr-models/sherpa-onnx-nemo-fast-conformer-transducer-es-1424-int8.tar.bz2"),
    (MODELOS,
     "vits-piper-es_MX-claude-high/es_MX-claude-high.onnx",
     f"{BASE}/tts-models/vits-piper-es_MX-claude-high.tar.bz2"),
]


def bajar_modelos():
    for carpeta, prueba, url in DESCARGAS:
        if os.path.exists(os.path.join(carpeta, prueba)):
            print(f"[ok] ya esta: {prueba.split('/')[0]}")
            continue
        os.makedirs(carpeta, exist_ok=True)
        archivo = os.path.join(carpeta, url.rsplit("/", 1)[1])
        print(f"[..] bajando {url.rsplit('/', 1)[1]} (una sola vez)...")
        urllib.request.urlretrieve(url, archivo)
        with tarfile.open(archivo, "r:bz2") as tar:
            tar.extractall(carpeta)
        os.remove(archivo)
        print(f"[ok] listo: {prueba.split('/')[0]}")


def responde(url):
    try:
        urllib.request.urlopen(url, timeout=3)
        return True
    except Exception:
        return False


def ip_para_la_pi():
    """La IP de esta PC en la red de la Pi (no la de VirtualBox ni WSL)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.143.0.1", 80))   # no envia nada: solo elige la interfaz
        return s.getsockname()[0]
    except OSError:
        return "<ip-de-esta-pc>"


def main():
    bajar_modelos()

    print("\n[..] arrancando servidores...")
    procesos = [subprocess.Popen([sys.executable, os.path.join(SCRIPTS, s)])
                for s in ("stt_server.py", "tts_server.py")]

    servicios = [("Ollama", "http://localhost:11434/api/tags"),
                 ("STT", "http://localhost:8082/salud"),
                 ("TTS", "http://localhost:8083/salud")]
    for _ in range(60):          # hasta 60 s cargando modelos
        if all(responde(u) for _, u in servicios):
            break
        time.sleep(1)

    print()
    for nombre, url in servicios:
        print(f"  {'OK   ' if responde(url) else 'FALLA'}  {nombre}")
    print(f"\nEn la Pi:\n  ROMEO_PC={ip_para_la_pi()} ros2 launch romeo_bringup romeo_bringup.launch.py")
    print("\nCtrl+C para apagar los servidores.\n")

    try:
        while all(p.poll() is None for p in procesos):
            time.sleep(1)
        print("[!] un servidor se cerro; mira el error arriba")
    except KeyboardInterrupt:
        pass
    finally:
        for p in procesos:
            p.terminate()


if __name__ == "__main__":
    main()
