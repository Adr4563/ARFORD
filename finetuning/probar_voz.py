# -*- coding: utf-8 -*-
"""Mide si el microfono de la Lora Audio Board capta la voz, y la transcribe.

Lo ejecuta el USUARIO en su terminal, para que vea la cuenta atras y sepa
cuando hablar -- las mediciones hechas desde otra sesion capturaban silencio
porque el aviso no llegaba a quien tenia que hablar.

    python probar_voz.py          # 8 s con la placa en el PC
    python probar_voz.py 12       # 12 s
"""
import sys
import time
import wave

import numpy as np
import sounddevice as sd

SEGUNDOS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
SR = 16000

dispositivo = next(
    (i for i, d in enumerate(sd.query_devices())
     if "usb uac" in d["name"].lower() and d["max_input_channels"] > 0), None)
if dispositivo is None:
    print("No encuentro la placa ('usb uac'). Dispositivos de entrada:")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"  [{i}] {d['name']}")
    sys.exit(1)

print(f"Microfono: [{dispositivo}] {sd.query_devices(dispositivo)['name']}")
print()
for n in (3, 2, 1):
    print(f"  {n}...", flush=True)
    time.sleep(1)
print(f"  >>> HABLA AHORA <<<  ({SEGUNDOS} segundos)", flush=True)

audio = sd.rec(SEGUNDOS * SR, samplerate=SR, channels=1, dtype="float32",
               device=dispositivo)
sd.wait()
print("  listo.")
print()

x = audio.reshape(-1)
ventana = SR // 4
picos = [float(abs(x[i:i + ventana]).max())
         for i in range(0, len(x) - ventana + 1, ventana)]
pico = max(picos)
variacion = pico / max(min(picos), 1e-4)

print(f"  pico maximo : {100 * pico:.1f}% de la escala")
print(f"  por 1/4 seg : {' '.join('%.0f' % (100 * p) for p in picos)}")
print(f"  variacion   : {variacion:.0f}x")
print()
if variacion < 3:
    print("  >> NO hay voz: el nivel es plano, solo ruido de fondo.")
elif pico < 0.15:
    print(f"  >> HAY voz, pero floja ({100 * pico:.0f}%). whisper quiere 26-31%:")
    print("     habla mas cerca (5-10 cm) y mas fuerte.")
else:
    print("  >> HAY voz con buen nivel.")

ganancia = min(0.9 / pico, 20.0)
y = (np.clip(x * ganancia, -1, 1) * 32767).astype(np.int16)
with wave.open("voz_real.wav", "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(y.tobytes())
print()
print(f"  guardado voz_real.wav (amplificado x{ganancia:.1f})")
print("  avisame y lo transcribo con el whisper de la Pi.")
