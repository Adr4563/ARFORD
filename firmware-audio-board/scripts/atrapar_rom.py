#!/usr/bin/env python3
"""Atrapa la ventana del cargador de ROM del ESP32-S3 y graba en ella.

PROBLEMA QUE RESUELVE
---------------------
Cuando la placa corre el firmware UAC, se apodera del USB y deja de exponer
puerto serie: `esptool` ya no puede alcanzarla y no hay forma de regrabarla
por software. Lo normal sería entrar en modo descarga con el botón BOOT.

Pero el registro del núcleo demuestra que, en CADA reinicio, el cargador de
ROM enumera igualmente y el sistema llega a crear `/dev/ttyACM0`::

    9566.109  usb 1-1.3: idProduct=1001  (USB JTAG/serial debug unit)
    9566.197  cdc_acm 1-1.3:1.0: ttyACM0: USB ACM device   <-- el puerto existe
    9566.455  usb 1-1.3: USB disconnect                    <-- 258 ms despues

Esa ventana dura entre 258 y 512 ms. Es suficiente para que esptool sincronice
y retenga el chip mediante DTR/RTS -- pero NO si hay que arrancar un proceso
nuevo: cargar el intérprete de Python cuesta del orden de medio segundo, o sea
más que la ventana entera. Por eso fallaban los intentos con `esptool` en
bucle o con `idf.py flash`: llegaban tarde siempre.

SOLUCIÓN
--------
Un único proceso, con esptool YA importado en memoria, que provoca él mismo el
reinicio (cortando la corriente del puerto USB con uhubctl) y vigila el fichero
cada milisegundo. Cuando aparece, graba en el acto: cero arranques de proceso
entre la detección y la escritura.

Así no hace falta pulsar ningún botón.

USO
---
    ~/.lora/venv/bin/python atrapar_rom.py restaurar   # vuelve a Xiaozhi
    ~/.lora/venv/bin/python atrapar_rom.py grabar      # firmware corregido
"""

import os
import subprocess
import sys
import time

import esptool

PUERTO = "/dev/ttyACM0"
HUB = "1-1"
PUERTO_HUB = "3"

RESPALDO = os.path.expanduser("~/xiaozhi-firmware-backup.bin")
CONSTRUCCION = os.path.expanduser("~/firmware-audio-board/build")

# Cuántas veces reintentar el ciclo completo. Cada reinicio abre una ventana
# nueva, así que insistir sale casi gratis y sube mucho la probabilidad.
INTENTOS = 6
ESPERA_MAX = 25.0


def ciclo_de_corriente():
    """Corta y restaura la alimentación del puerto USB de la placa."""
    subprocess.run(
        ["sudo", "uhubctl", "-l", HUB, "-p", PUERTO_HUB, "-a", "cycle", "-d", "2"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def esperar_puerto(limite):
    """Vigila el nodo cada milisegundo. Devuelve cuánto tardó, o None."""
    t0 = time.time()
    while time.time() - t0 < limite:
        if os.path.exists(PUERTO):
            return time.time() - t0
        time.sleep(0.001)
    return None


def argumentos(modo):
    comun = [
        "--chip", "esp32s3",
        "--port", PUERTO,
        "--baud", "921600",
        "--before", "default-reset",
        "--after", "hard-reset",
        "--connect-attempts", "10",
        "write-flash",
        "--flash-mode", "dio",
        "--flash-freq", "80m",
        "--flash-size", "16MB",
    ]
    if modo == "restaurar":
        return comun + ["0x0", RESPALDO]
    return comun + [
        "0x0", os.path.join(CONSTRUCCION, "bootloader", "bootloader.bin"),
        "0x8000", os.path.join(CONSTRUCCION, "partition_table", "partition-table.bin"),
        "0x10000", os.path.join(CONSTRUCCION, "lora_audio_board.bin"),
    ]


def main():
    modo = sys.argv[1] if len(sys.argv) > 1 else "restaurar"
    if modo not in ("restaurar", "grabar"):
        print("Uso: atrapar_rom.py [restaurar|grabar]")
        return 2

    destino = RESPALDO if modo == "restaurar" else CONSTRUCCION
    print(f"Modo: {modo}  ->  {destino}")
    print("esptool ya cargado en memoria; no se arrancará ningún proceso nuevo.")

    # MODO VIGILAR: no se corta la corriente, solo se espera a que la persona
    # pulse los botones.
    #
    # Está medido que hace falta: los seis cortes de corriente de una tanda
    # anterior enumeraron TODOS como idProduct=8000 (la aplicación), sin una
    # sola ventana de ROM. Al arrancar en frío el chip lee GPIO0, lo ve alto y
    # salta directo a la aplicación. En cambio las 15 ventanas de ROM
    # registradas (idProduct=1001, con su ttyACM0 creado) coincidieron todas
    # con pulsaciones manuales. O sea: el botón funciona y el corte de
    # corriente no. Cortar la corriente además ESTORBA, porque reinicia la
    # placa justo cuando la persona estaba a punto de abrir su ventana.
    vigilar = len(sys.argv) > 2 and sys.argv[2] == "vigilar"

    intentos = 1 if vigilar else INTENTOS
    espera = 120.0 if vigilar else ESPERA_MAX

    if vigilar:
        print("\n>>> PULSA LOS BOTONES AHORA, repetidamente, durante 2 minutos.")
        print(">>> Mantener uno y pulsar otro, una y otra vez. No pares.")

    for intento in range(1, intentos + 1):
        if not vigilar:
            print(f"\n--- Intento {intento}/{intentos}: cortando corriente ---")
            ciclo_de_corriente()

        tardanza = esperar_puerto(espera)
        if tardanza is None:
            print("   el puerto no llegó a aparecer")
            continue

        print(f"   puerto detectado en {tardanza:.3f}s -- grabando YA")
        try:
            esptool.main(argumentos(modo))
            print("\nGRABACIÓN COMPLETADA")
            return 0
        except SystemExit as e:
            # esptool llama a sys.exit() al terminar, también cuando va bien.
            if e.code in (0, None):
                print("\nGRABACIÓN COMPLETADA")
                return 0
            print(f"   esptool falló (código {e.code}); se reintenta")
        except Exception as e:  # noqa: BLE001 - se reintenta con otra ventana
            print(f"   no se pudo sincronizar: {e}")

    print("\nNo se logró atrapar la ventana en ningún intento.")
    print("Queda el camino manual: mantener BOOT pulsado durante un reinicio.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
