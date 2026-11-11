#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproduce la sesion de trivia alimentando las respuestas ya dadas por el
usuario, pasando por el flujo COMPLETO (STT -> afectiva -> LLM -> TTS) para
medir recursos y tiempos de inicio a fin.

Para cada respuesta:
  1) genera un WAV con el TTS de Romeo (voz del robot; el timbre no importa
     para el STT, solo que sea habla real)
  2) envia ese WAV al servidor STT del PC -> mide STT, SER y TER
  3) publica el texto transcrito en /romeo/user_input -> Romeo hace la capa
     afectiva, el LLM y responde por TTS

Corre EN LA PI (necesita ROS2 para publicar el topico).

    python3 reproducir_sesion.py respuestas_usuario.json <IP_DEL_PC>

El recolector de recursos debe estar corriendo aparte, de inicio a fin.
"""
import json
import sys
import time
import urllib.request
import urllib.parse

import rclpy
from rclpy.node import Node
from romeo_interfaces.msg import UserInput

PC = sys.argv[2] if len(sys.argv) > 2 else "10.143.105.48"
TTS = f"http://{PC}:8083/hablar"
STT = f"http://{PC}:8082/transcribir"
PAUSA = 10.0  # segundos entre respuestas (deja que Romeo complete el turno)


def generar_wav(texto):
    """Pide al TTS el audio del texto; devuelve los bytes WAV."""
    url = TTS + "?" + urllib.parse.urlencode({"texto": texto})
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read()


def transcribir(wav):
    """Envia el WAV al STT (mide STT/SER/TER); devuelve el dict completo
    (texto, ser, ter, tiempos)."""
    req = urllib.request.Request(STT, data=wav,
                                 headers={"Content-Type": "audio/wav"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def main():
    respuestas = json.load(open(sys.argv[1], encoding="utf-8"))
    rclpy.init()
    nodo = Node("reproductor_sesion")
    pub = nodo.create_publisher(UserInput, "/romeo/user_input", 10)
    time.sleep(1.0)

    print(f"Reproduciendo {len(respuestas)} respuestas | PC={PC}")
    for i, texto in enumerate(respuestas, 1):
        texto = texto.strip()
        if not texto:
            continue
        print(f"\n[{i}/{len(respuestas)}] {texto[:50]}")
        try:
            t0 = time.time()
            wav = generar_wav(texto)
            print(f"   TTS->WAV {len(wav)} bytes ({time.time()-t0:.1f}s)")
            # 2) STT mide STT/SER/TER sobre ese audio
            t1 = time.time()
            res = transcribir(wav)
            trans = res.get("texto") or texto
            print(f"   STT:'{trans[:35]}' SER={bool(res.get('ser'))} "
                  f"TER={bool(res.get('ter'))} ({time.time()-t1:.1f}s)")
        except Exception as e:  # noqa: BLE001
            print(f"   (fallo audio, se publica el texto directo: {e})")
            trans, res = texto, {}
        # 3) publicar el texto + SER + TER al orquestador -> afectiva+LLM+TTS
        msg = UserInput()
        msg.text = trans or texto
        try:
            msg.source = "replay"
            msg.t_inicio = time.time()
            msg.t_fin = time.time()
            ser = res.get("ser") or {}
            msg.ser_etiquetas = list(ser.get("etiquetas", []))
            msg.ser_probabilidades = [float(x) for x in ser.get("probabilidades", [])]
            ter = res.get("ter") or {}
            msg.ter_etiquetas = list(ter.get("etiquetas", []))
        except Exception:  # noqa: BLE001
            pass
        pub.publish(msg)
        print(f"   -> publicado a Romeo; espera {PAUSA:g}s")
        time.sleep(PAUSA)

    print("\nFIN de la reproduccion.")
    nodo.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
