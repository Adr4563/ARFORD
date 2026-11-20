# Comparación de detectores de cara

Compara YuNet (el detector actual de ROMEO, en CPU y dentro del sensor
IMX500) con SCRFD-2.5GF, YOLOv8n-face y YOLO11n-face **sobre los mismos
cuadros**. Todo corre en la Raspberry Pi, desde la raíz del repo, y no cambia
nada de ROMEO. El método, la regla de decisión y los resultados están en
[INFORME_CARAS.md](INFORME_CARAS.md).

## 1. Preparar (una vez)

```bash
python3 -c "import cv2, onnxruntime, psutil, picamera2; print('ok')"
python3 pruebas/caras/descargar_modelos.py   # pide confirmación (~27 MB)
```

## 2. Grabar

**Primero detén ROMEO**: la cámara solo admite un proceso a la vez.

```bash
# un escenario tras otro, 100 cuadros cada uno (640x480 y 320x240 del mismo instante)
python3 pruebas/caras/grabar_caras.py --escenarios dist_1m sin_cara
# modo servidor: lee "escenario [cuadros]" por stdin, sin recargar el firmware del IMX500
python3 pruebas/caras/grabar_caras.py --servidor
```

Todo queda en `pruebas/caras/datos/`:

| Carpeta | Contenido |
|---|---|
| `640x480/<escenario>/`, `320x240/<escenario>/` | los cuadros (PNG en BGR correcto) |
| `sensor/<escenario>/` | tensores crudos del YuNet del IMX500 (solo en los cuadros donde los entregó) |
| `meta/<escenario>.jsonl`, `meta/<escenario>.info.json` | metadatos de la cámara por cuadro y resumen del escenario |

## 3. Comparar

```bash
python3 pruebas/caras/comparar_caras.py --prueba    # 5 cuadros por carpeta
python3 pruebas/caras/comparar_caras.py             # todo; aplica la regla de decisión
python3 pruebas/caras/efecto_color.py               # YuNet y FER con el color correcto o invertido
```

Las salidas van a `pruebas/caras/resultados/`:
- `caras.csv`: una fila por cuadro y detector.
- `caras.json`: las vistas 1 y 2 y la decisión.
- `mosaicos/`: 20 cuadros con las cajas de cada detector.
- `efecto_color.json`.

## Licencias

| Detector | Licencia |
|---|---|
| YuNet | MIT (OpenCV Zoo) |
| SCRFD-2.5GF (`det_2.5g.onnx`, InsightFace) | **solo investigación no comercial** |
| YOLOv8n-face / YOLO11n-face (akanametov/yolo-face) | repo: GPL-3.0; **metadatos de los ONNX: AGPL-3.0** (Ultralytics) |

Las carpetas `datos/`, `resultados/` y `modelos/` no se suben al repo
(`.gitignore`): tienen rostros del equipo y modelos de terceros.
