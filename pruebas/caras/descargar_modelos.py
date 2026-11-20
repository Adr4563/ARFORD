#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""descargar_modelos.py -- baja los detectores de comparar_caras.py a
pruebas/caras/modelos/. Pide confirmacion antes de descargar.

Enlaces verificados con la API de GitHub (2026-11-18):
  yolov8n-face.onnx   12,7 MB  akanametov/yolo-face, release 1.0.0   GPL-3.0
  yolov11n-face.onnx  10,6 MB  akanametov/yolo-face, release 1.0.0   GPL-3.0
  det_2.5g.onnx        3,3 MB  dentro de buffalo_m.zip (276 MB) de
                               deepinsight/insightface, release v0.7.
                               Solo se baja ESE archivo leyendo el zip por
                               rangos HTTP, no el paquete entero.
                               Licencia: investigacion NO comercial.
YuNet no se baja: ya esta en el repo.
"""
import hashlib
import struct
import sys
import urllib.request
import zlib
from pathlib import Path

DESTINO = Path(__file__).resolve().parent / "modelos"
YOLO = {
    "yolov8n-face.onnx": "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.onnx",
    "yolov11n-face.onnx": "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11n-face.onnx",
}
BUFFALO_M = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_m.zip"
DENTRO = "buffalo_m/det_2.5g.onnx"


def rango(url, ini, fin):
    req = urllib.request.Request(url, headers={"Range": f"bytes={ini}-{fin}"})
    return urllib.request.urlopen(req, timeout=60).read()


def extraer_de_zip(url, nombre):
    """Un archivo de un zip remoto, sin bajar el zip completo."""
    total = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD"),
                                       timeout=60).headers["Content-Length"])
    cola = rango(url, max(0, total - 65536), total - 1)
    i = cola.rfind(b"PK\x05\x06")
    tam_cd, off_cd = struct.unpack("<II", cola[i + 12:i + 20])
    cd = rango(url, off_cd, off_cd + tam_cd - 1)
    p = 0
    while cd[p:p + 4] == b"PK\x01\x02":
        metodo = struct.unpack("<H", cd[p + 10:p + 12])[0]
        crc, csize, usize = struct.unpack("<III", cd[p + 16:p + 28])
        fl, el, cl = struct.unpack("<HHH", cd[p + 28:p + 34])
        off_local = struct.unpack("<I", cd[p + 42:p + 46])[0]
        n = cd[p + 46:p + 46 + fl].decode()
        if n == nombre:
            cab = rango(url, off_local, off_local + 29)
            fl2, el2 = struct.unpack("<HH", cab[26:30])
            ini = off_local + 30 + fl2 + el2
            datos = rango(url, ini, ini + csize - 1)
            if metodo == 8:
                datos = zlib.decompressobj(-15).decompress(datos)
            elif metodo != 0:
                raise RuntimeError(f"compresion {metodo} no soportada")
            if len(datos) != usize or (zlib.crc32(datos) & 0xFFFFFFFF) != crc:
                raise RuntimeError("tamano o CRC no coinciden")
            return datos
        p += 46 + fl + el + cl
    raise RuntimeError(f"{nombre} no esta en el zip")


def guardar(nombre, datos):
    DESTINO.mkdir(parents=True, exist_ok=True)
    (DESTINO / nombre).write_bytes(datos)
    print(f"  {nombre}: {len(datos) / 1e6:.1f} MB  sha256 {hashlib.sha256(datos).hexdigest()}")


def main():
    print(__doc__)
    if input("¿Descargar? [s/N] ").strip().lower() != "s":
        sys.exit("Cancelado.")
    for nombre, url in YOLO.items():
        if (DESTINO / nombre).exists():
            print(f"  {nombre}: ya existe")
            continue
        guardar(nombre, urllib.request.urlopen(url, timeout=120).read())
    if (DESTINO / "det_2.5g.onnx").exists():
        print("  det_2.5g.onnx: ya existe")
    else:
        guardar("det_2.5g.onnx", extraer_de_zip(BUFFALO_M, DENTRO))


if __name__ == "__main__":
    main()
