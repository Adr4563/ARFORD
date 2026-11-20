# Comparación de detectores de cara para ROMEO

**Objetivo.** Decidir si YuNet, el detector de cara actual de ROMEO, se queda
o si algún detector ligero lo supera en la Raspberry Pi 4B con la AI Camera
(IMX500). Se comparan todos sobre los mismos cuadros grabados.

**Etiquetas.** Cada afirmación lleva **[MEDIDO]** (sale de una medición o del
código, con su fuente) o **[HIPÓTESIS]** (sin verificar todavía).

**Estado:** regla de decisión escrita el 19-11-2026, **antes de grabar y de
ver resultados** (sección 2, sin cambios desde entonces). Pruebas completadas
el mismo día; resultados en las secciones 5 a 7.

**Resumen.**
- **YuNet se queda.** Empata con SCRFD, YOLOv8n-face y YOLO11n-face (todos
  al 100 % en una prueba que resultó demasiado fácil) y es 16 a 22 veces más
  rápido.
- El problema real está en cómo ROMEO usa YuNet **dentro del sensor**:
  - un umbral más estricto de lo que parece;
  - pérdida de caras de perfil;
  - cuadros descartados;
  - y colores invertidos en las etapas de CPU.

---

## 1. Montaje

| Elemento | Detalle |
|---|---|
| Equipo | Raspberry Pi 4B, 4 núcleos, 7,6 GB de RAM, Ubuntu 26.04 LTS, kernel 7.0.0-1020-raspi, Python 3.14.4 [MEDIDO] |
| Cámara | Raspberry Pi AI Camera, sensor IMX500 (4056x3040; modo de 2028x1520 a 30 fps) [MEDIDO, `rpicam-hello --list-cameras`] |
| Librerías | OpenCV 4.10.0, onnxruntime 1.29.0, numpy 2.3.5, psutil 7.1.0, picamera2 [MEDIDO] |
| Hilos | 2 por detector, igual que ROMEO (`HILOS_ONNX=2`) |
| ROMEO | detenido durante las pruebas; no se cambió su código ni sus parámetros |

### Detectores

| Nombre | Modelo | Dónde corre | Umbral de operación | Licencia |
|---|---|---|---|---|
| yunet_cpu | `face_detection_yunet_2023mar.onnx` (el del repo) | CPU de la Pi | 0,7 | MIT (OpenCV Zoo) |
| yunet_sensor | el mismo modelo, convertido a `.rpk` int8 | dentro del IMX500 | 0,7 | MIT (convertido por el equipo) |
| scrfd_2g5 | `det_2.5g.onnx` (InsightFace, paquete `buffalo_m`, release v0.7) | CPU | 0,5 | **Solo investigación no comercial** |
| yolov8n_face | `yolov8n-face.onnx` (akanametov/yolo-face, release 1.0.0) | CPU | 0,25 | repo: GPL-3.0; **metadatos del modelo: AGPL-3.0** |
| yolo11n_face | `yolov11n-face.onnx` (mismo origen) | CPU | 0,25 | repo: GPL-3.0; **metadatos del modelo: AGPL-3.0** |

**[MEDIDO, metadatos de los ONNX]** Los dos YOLO dicen
`license: AGPL-3.0 License (https://ultralytics.com/license)`. Se entrenaron
con Ultralytics; la licencia del repo de origen dice GPL-3.0. Si un YOLO-face
se integrara, hay que asumir **AGPL-3.0**, que además obliga a publicar el
código si el sistema se usa a través de una red.

**[MEDIDO, metadatos]** `yolov8n-face.onnx` es en realidad un modelo
**YOLOv8n-pose** (cara más 5 puntos; Ultralytics 8.3.241), exportado con el
NMS incluido. Su salida es `(1, 300, 21)`, con cajas en formato x1,y1,x2,y2.
Por eso:
- el puntaje crudo de la vista 2 no baja de su umbral interno; el menor valor
  no cero visto fue 0,77;
- en la vista 2 su criterio quedó degenerado (0 FP a cualquier umbral).

`yolov11n-face.onnx` es un detector normal: `task=detect`, salida `(1, 5, 8400)`.

Si SCRFD o un YOLO-face ganara y se integrara a ROMEO, **hay que revisar si
su licencia es compatible con la del repositorio**. SCRFD no admite uso
comercial.

Sumas sha256 de los modelos descargados [MEDIDO, salida de `descargar_modelos.py`]:
- `yolov8n-face.onnx`: `06b941fd5792be624ad18f2df9ede0a021c4df165dd418204d978c20fd555928`
- `yolov11n-face.onnx`: `61d8cf127ff377939cf009565457aaddf216f4d9fdd9ac24f5aab3cb10ee67b5`
- `det_2.5g.onnx`: `041f73f47371333d1d17a6fee6c8ab4e6aecabefe398ff32cca4e2d5eaee0af9`

## 2. Regla de decisión (escrita antes de ver los resultados)

Se aplica sobre la **vista 2** ("igualdad de condiciones"). **Nunca se declara
ganador solo con la vista 1.**

**Escenarios de uso real:** `dist_0.5m`, `dist_1m`, `dist_1.5m`, `dist_2m`
y `ang_30`. Sus cuadros se agrupan por resolución.

**Qué se compara:** la tasa de detección de YuNet (yunet_cpu) contra la del
mejor competidor (el de tasa más alta entre scrfd_2g5, yolov8n_face y
yolo11n_face). Cada uno usa el umbral que deja como máximo 1 % (y, aparte,
5 %) de los cuadros de `sin_cara` con una caja.

**Intervalos:** Wilson del 95 % calculado con la **muestra efectiva**
`n_ef = n(1−r)/(1+r)`, donde `r` es la autocorrelación de lag 1 de la serie
de cuadros. Es más conservador que usar n, porque los cuadros seguidos no
son independientes.

| Caso | Resultado |
|---|---|
| El intervalo de YuNet se solapa con el del mejor competidor (o YuNet tiene la tasa más alta) **y** la latencia de YuNet es menor o igual | **YuNet se queda** |
| El mejor competidor supera a YuNet **sin solaparse** | Se reporta con su costo en latencia, CPU, temperatura y licencia. **No se decide el cambio.** |
| Empate y YuNet tiene más latencia | Se desempata por CPU, después temperatura, después licencia |
| Diferencias dentro del solape | Se reportan como **EMPATE**, nunca como victoria |
| La tasa mejora sin solaparse al pasar de 320x240 a 640x480 | Hallazgo aparte: **el límite son los píxeles**, no el modelo |

**Resolución principal:** 320x240, la de ROMEO (`fer_resolucion`). La de
640x480 se reporta aparte.

**Datos del sensor:** yunet_sensor se evalúa en las dos vistas, porque se
guardan sus tensores crudos sin umbral. En la regla, a YuNet lo representa
yunet_cpu.

**Si a todos los umbrales hay 0 falsos positivos,** el criterio de 1 % o 5 %
queda degenerado. Se dirá así y se reportará la tasa al umbral mínimo.

`comparar_caras.py` aplica esta regla automáticamente (función `decision`).

## 3. Método

**Grabación** (`grabar_caras.py --servidor`):
- Cada pedido a la cámara da dos flujos del **mismo instante**: el principal
  a 640x480 (BGR888) y el de baja resolución a 320x240 (YUV420, convertido a
  BGR). Así las dos resoluciones tienen la misma pose.
- 5 cuadros por segundo, igual que `fer_fps` de ROMEO.
- Al menos 100 cuadros por escenario y resolución; `sin_cara`, 200, con fondos
  variados.
- Con cada cuadro se guardan:
  - los 12 tensores crudos del YuNet del sensor;
  - los metadatos de la cámara: ExposureTime, AnalogueGain, DigitalGain,
    ColourGains, ColourTemperature, Lux, FrameDuration y SensorTimestamp.

**Escenarios.** Cambia una sola variable a la vez; la base es 1 m, de frente
y con luz normal.
- Distancia: 0,5 m, 1 m, 1,5 m, 2 m y 3 m.
- Cara girada: unos 30° y unos 60°.
- Luz baja.
- Con lentes y sin lentes.
- Sin cara.
- Un objeto rojo, para la prueba del orden de color.

Durante cada toma la persona hace movimientos pequeños y naturales (gira
apenas la cabeza, parpadea, cambia el gesto). Aun así, los cuadros seguidos
están correlacionados, y por eso se reporta `n_ef`.

**Comparación** (`comparar_caras.py`):
- Un proceso por detector.
- Latencia mediana y p95 por cuadro.
- CPU medida solo durante la inferencia, sumando todos los hilos; 100 %
  equivale a un núcleo.
- RAM (RSS) y temperatura (`thermal_zone0`).
- Por cada cuadro se guarda también el **puntaje máximo sin umbral**. Para
  yunet_cpu sale de una segunda instancia con umbral 0,01 que corre fuera del
  tiempo medido.
- "Detección" significa que hay al menos una caja (o que el puntaje máximo
  pasa el umbral). **No hay cajas de referencia**: la tasa no distingue una
  caja bien puesta de una mal puesta. Por eso se guardan 20 mosaicos con las
  cajas de todos los detectores, para revisarlos a ojo.

**Datos.** Los cuadros son rostros del equipo. Se quedan en la Pi
(`pruebas/caras/datos/`, incluido en `.gitignore`) y no se suben al repo. Son
pruebas técnicas, no un estudio con participantes.

## 4. Hallazgos previos sobre ROMEO (pruebas del 18-11-2026)

1. **[MEDIDO] Orden de color.** `_emotion_detector.py` pide
   `FORMATO_CAPTURA = "BGR888"` y trata el cuadro como BGR sin convertirlo
   (`_a_bgr`). picamera2 sigue los nombres de libcamera: "BGR888" deja los
   bytes en orden **R,G,B**. Prueba: un cuadro guardado tal cual con
   `cv2.imwrite` mostró la piel azul. Por tanto, el YuNet de CPU y el FER de
   ROMEO reciben los canales rojo y azul intercambiados. La prueba con un
   objeto rojo y el efecto en el FER están en la sección 6.
2. **[MEDIDO] El YuNet del sensor entrega salida en ~1 de cada 2 cuadros.**
   Con capturas seguidas durante 12 s, 151 de 302 cuadros a 320x240
   (alternando exactamente 0/1) y 88 de 354 a 640x480. Al ritmo de 5 fps,
   18 de 57.
3. **[MEDIDO, en el código] Esos cuadros se descartan.** En
   `_emotion_detector.py:509-513`, si el cuadro no trae salida del sensor,
   `caja` queda en `None` y el cuadro no se clasifica: se pierde como si no
   hubiera cara.
   **[HIPÓTESIS]** Por eso el FER clasifica bastante por debajo de los 5 fps
   configurados. Se contrasta con el log de "fps reales" del nodo en la
   sección 6.
4. **Los umbrales de 0,7 no miden lo mismo. Verificado el 19-11, ver 5.4.**

## 5. Resultados (grabación y comparación del 19-11-2026) [MEDIDO]

### 5.1 Datos

| Escenario | Cuadros por resolución | Observación |
|---|---|---|
| dist 0,5 / 1 / 1,5 / 2 / 3 m | 100 c/u | En 3 m aparece al fondo una persona de la oficina (lo detectan varios detectores a 640x480); no cambia la tasa de "al menos una caja". |
| ang_30, ang_60 | 100 c/u | En los mosaicos, el giro de 60° es casi de perfil. |
| lentes_si, lentes_no | 100 c/u | |
| sin_cara | 200 | **Poca variedad de fondo**: 195 de 199 pares de cuadros seguidos casi iguales (cámara casi quieta). La primera toma se descartó porque tenía 2 personas reales al fondo. |
| **luz_baja** | **no medido** | No se pudo bajar la luz del ambiente. |
| rojo | 30 | Celular con pantalla roja, para la prueba de color (sección 6). |

- Total: 2200 cuadros, 640x480 y 320x240 del mismo instante, a 5 fps.
- El periodo real entre cuadros fue de 0,200 s.
- La temperatura de la Pi durante la grabación fue de 49 a 58 °C.

### 5.2 Costo (2 hilos, Pi 4B; mediana / p95 por cuadro)

| Detector | 320x240 | 640x480 | CPU | RAM | Temperatura al correr |
|---|---|---|---|---|---|
| yunet_cpu | **31 / 34 ms** | 105 / 113 ms | 184 % | 156 MB | 56 → 65 °C |
| yunet_sensor | 11 / 12 ms (solo decodificar) | 10 / 11 ms | 100 % de un núcleo, solo al decodificar | 152 MB | — |
| scrfd_2g5 | 502 / 511 ms | 503 / 511 ms | 201 % | 188 MB | 62 → 71 °C |
| yolov8n_face | 694 / 706 ms | 696 / 708 ms | 201 % | 200 MB | 69 → 76 °C |
| yolo11n_face | 603 / 616 ms | 611 / 623 ms | 201 % | 196 MB | 73 → 77 °C |

Los tres competidores trabajan siempre a 640x640, así que su costo no baja a
320x240. A 5 fps, SCRFD y los YOLO **no alcanzan el tiempo real**: necesitan
de 0,5 a 0,7 s por cuadro, y el periodo es de 0,2 s.

### 5.3 Vista 1: cada detector con su umbral de operación (320x240; 640x480 da lo mismo)

| Escenario | yunet_cpu | yunet_sensor | scrfd_2g5 | yolov8n_face | yolo11n_face |
|---|---|---|---|---|---|
| dist_0.5m | 100 | 100 | 100 | 100 | 100 |
| dist_1m | 100 | 96 | 100 | 100 | 100 |
| dist_1.5m | 100 | **86** | 100 | 100 | 100 |
| dist_2m | 100 | 97 | 100 | 100 | 100 |
| dist_3m | 100 | 97 | 100 | 100 | 100 |
| ang_30 | 100 | 96 | 100 | 100 | 100 |
| **ang_60** | 100 | **3** | 100 | 100 | 100 |
| lentes_si | 100 | 93 | 100 | 100 | 100 |
| lentes_no | 100 | 97 | 100 | 100 | 100 |
| sin_cara (FP) | 0 | 0 | 0 | 0 | 0 |

- Tasas en %, n = 100 por celda (200 en sin_cara).
- Con n = 100 y 100 %, el intervalo de Wilson va de 96 a 100 %. Con la
  muestra efectiva (n_ef de 3 a 68 según la celda) el límite inferior baja
  hasta 44–93 %.
- El ancho mediano de la caja coincide entre detectores en ±2 px:
  - a 320x240: 73 px a 0,5 m, 47 px a 1 m, 34 px a 1,5 m, 29 px a 2 m y
    22 px a 3 m;
  - a 640x480, el doble.

### 5.4 El umbral del sensor [MEDIDO]

- **El puntaje de `FaceDetectorYN` es exactamente √(cls·obj)** (diferencia
  0,0000 en 60 cuadros, calculada con la red cruda en `cv2.dnn`).
  `_yunet_camara.decodificar`, en cambio, compara **cls·obj sin raíz**
  contra 0,7. Por eso **el 0,7 del sensor equivale a 0,84 en la CPU**.
- Con el umbral equivalente (0,49 sobre el producto), yunet_sensor queda
  así, todavía con 0 % de FP en sin_cara:

  | Escenario | Umbral 0,7 | Umbral 0,49 |
  |---|---|---|
  | dist_1.5m | 86 % | 99 % |
  | lentes_si | 93 % | 94 % |
  | ang_60 | 3 % | 49 % |
  | resto | 96–97 % | 100 % |

- **Además, el modelo int8 del sensor puntúa mucho más bajo con la cara de
  perfil.** En ang_60, la mediana de cls·obj es 0,79 en la CPU y 0,33 en el
  sensor; en dist_1m es 0,85 contra 0,77. A 60° el problema no es solo el
  umbral.
  - **[HIPÓTESIS]** La causa sería la cuantización a int8 o el cambio de los
    Resize hecho al convertir.
  - **[HIPÓTESIS]** También podría influir que la red recibe el campo
    completo 4:3 comprimido a 640x640, lo que deforma la cara (sección 6a).

### 5.5 Vista 2: igualdad de condiciones

Umbral de cada detector fijado para un máximo de 1 % y de 5 % de FP en
sin_cara (200 cuadros):

| Detector | Umbral (FP ≤ 1 %, 320x240) | Uso real agrupado (0,5–2 m + 30°) | IC con n | IC con n_ef |
|---|---|---|---|---|
| yunet_cpu | 0,0185 | 100 % (500/500) | 99–100 | 96–100 |
| yunet_sensor | 0,002 | 100 % | 99–100 | 97–100 |
| scrfd_2g5 | 0,041 | 100 % | 99–100 | 97–100 |
| yolov8n_face | sin piso (0 FP a cualquier umbral; **criterio degenerado**) | 100 % | 99–100 | 97–100 |
| yolo11n_face | 0,0144 | 100 % | 99–100 | 97–100 |

Con FP ≤ 5 % y a 640x480 el resultado es igual: 100 % en todos los
escenarios, incluido ang_60.

**Efecto techo [MEDIDO]:** el puntaje más alto en sin_cara es tan bajo que
el umbral baja casi a 0 y todos llegan al 100 %. **Estos escenarios (una sola
persona cooperando y con buena luz) no separan a los detectores.** La poca
variedad de fondo en sin_cara también hace que el umbral sea poco
representativo.

## 6. Preguntas (a) y (b)

### (a) Resoluciones y fps

- **Entrada del YuNet del sensor: 640x640** [MEDIDO, `IMX500.get_input_size()`].
  - Toma el campo completo del sensor (`ScalerCrop` 4052x3040, 4:3)
    [MEDIDO, metadatos].
  - `_yunet_camara.py` no llama a `set_auto_aspect_ratio()`, y
    `decodificar` escala 640→ancho y 640→alto por separado [MEDIDO, en el
    código]. Por tanto, la red ve la escena **comprimida en horizontal** a
    un cuadrado [deducido del código; no hay forma de verla, porque el .rpk
    no entrega su tensor de entrada: `CnnInputTensor` ausente].
- **Recorte para el FER:** sale del cuadro principal de ROMEO a 320x240
  (`fer_resolucion` [320, 240] en `romeo_params.yaml` de la Pi). La cara mide
  73 px a 0,5 m, 47 px a 1 m, 34 px a 1,5 m, 29 px a 2 m y 22 px a 3 m
  [MEDIDO, 5.3]. El FER de serie (FER+) la lleva a **64x64 en gris**
  (`_emotion_detector.py:345` de la Pi). Desde unos 0,7 m **se amplía**, es
  decir, se inventan píxeles.
- **fps real del FER:**
  - El log de ROMEO dice "5.0 fps reales", **pero cuenta intentos y no
    clasificaciones** [MEDIDO, en el código: `ventana_cuadros += 1` en la
    línea 510, antes de saber si hubo cara; la observación se guarda solo si
    `probs` no es None, en la línea 530].
  - El sensor dio salida al primer pedido en **617 de 1100 cuadros (56 %)**,
    con entre 24 % y 100 % según la toma [MEDIDO en la grabación, que usa
    dos flujos]. En la prueba del 18-11 con un solo flujo, como ROMEO, fue
    32 %.
  - Con eso y la tasa del sensor a 0,7 (86–100 % de frente), **ROMEO
    clasificaría de 1,2 a 5 fps, unos 2,7 de media, en vez de 5**
    [HIPÓTESIS: estimación a partir de mediciones parciales, no medida
    dentro de ROMEO].

### (b) Colores y luz

**Qué pide y qué espera cada etapa** [MEDIDO, código de la Pi]:
- `_emotion_detector.py` pide `FORMATO_CAPTURA = "BGR888"` (línea 145).
  `_a_bgr` (líneas 265–278) lo devuelve sin convertir, porque supone que ya
  es BGR.
- Lo que espera cada etapa:
  - YuNet CPU (OpenCV): espera **BGR**.
  - FER+ y DeepFace: hacen `COLOR_BGR2GRAY`, o sea, suponen BGR.
  - HSEmotion y MobileFaceNet: aplican `[:, :, ::-1]` ("BGR→RGB") y esperan
    **RGB**.

**Orden real de la cámara [MEDIDO]:**
- El manual de picamera2 dice que "BGR888" entrega [R,G,B].
- Prueba con la pantalla roja de un celular, 30 cuadros:
  - en el cuadro crudo de `make_array("main")`, el canal 0 es el rojo
    (46,9 contra 40,7 del canal 2);
  - el flujo de 320x240 va por YUV420, independiente de ese nombre, y también
    da el rojo como canal más alto (35,1 contra 27,7).
  - La prueba confirma la piel azul del 18-11.
- **Conclusión: todas las etapas de CPU de ROMEO reciben R y B
  intercambiados.**
  - YuNet CPU recibe RGB y no BGR.
  - FER+ y DeepFace calculan el gris con los pesos de R y B cambiados.
  - HSEmotion y MobileFaceNet terminan recibiendo BGR donde esperan RGB.

**Sensor IMX500:**
- El .rpk **no entrega su tensor de entrada**, así que no se puede leer su
  orden de color [MEDIDO].
- Su imagen sale del procesamiento interno del sensor, no del flujo `main`.
  Por eso **el intercambio de ROMEO no lo afecta** [HIPÓTESIS, por la
  arquitectura de picamera2 e IMX500].
- **Prueba pendiente para decidir su orden:** mostrar en un monitor una foto
  pública de un rostro con los colores normales y otra con los canales
  intercambiados, y comparar el puntaje del sensor.

**Efecto medido** (`efecto_color.py`: los mismos 900 cuadros con cara, con
color correcto e invertido; la caja de la cara sale de la versión correcta):

| Etapa | 320x240 | 640x480 |
|---|---|---|
| YuNet CPU, detección ≥ 0,7 | **100 % → 89 %** (puntaje mediano 0,919 → 0,884) | 100 % → 100 % (0,925 → 0,908) |
| FER+ (el de serie), cambio de la emoción principal | 0 % | 0 % |
| DeepFace | 25,8 % | 20,4 % |
| HSEmotion | 12,7 % (neutral 86 → 77 %, miedo 3 → 7 %) | 20,9 % (miedo 5 → 15 %) |
| MobileFaceNet | 23,7 % (con el color invertido, neutral 81 → 95 %) | 31,4 % (neutral 67 → 98 %) |

- **FER+ no cambia, porque da "neutral" al 100 % en los 900 cuadros, con
  color correcto o invertido** [MEDIDO], aunque durante las tomas hubo
  gestos (sonreír, ponerse serio). **[HIPÓTESIS]** FER+ casi no responde con
  estas caras; es un punto central para la fase 2.
- **Para los FER en color, el intercambio es grave.** Con MobileFaceNet,
  casi todo se vuelve neutral.
- **Con el FER+ de serie y el YuNet del sensor, el efecto actual en ROMEO es
  pequeño.** Si ROMEO pasa a la CPU (sin sensor) o a un FER en color, el
  efecto deja de ser despreciable: 11 puntos menos de detección a 320x240 y
  13–31 % de emociones cambiadas.

**Luz [MEDIDO, metadatos por cuadro]:**
- La exposición fue casi constante: 10 ms, con ganancia de 1,7 a 2,0 y unos
  1100 lux.
- Subió en dist_0.5m (13,8 ms, 712 lux), donde la persona tapa más luz, y en
  sin_cara (15,4 ms).
- Las ColourGains fueron (1,78–1,86, 1,99–2,10).
- **No se sabe cómo cambian con poca luz**, porque luz_baja no se pudo grabar.

## 7. Conclusión

1. **Aplicando la regla sobre la vista 2: YuNet SE QUEDA.**
   - En los escenarios de uso real, los intervalos de yunet_cpu y del mejor
     competidor se solapan: EMPATE al 100 %.
   - La latencia de YuNet es mucho menor: 31 ms contra 502–694 ms.
   - No apareció el hallazgo de "el límite son los píxeles": 320x240 y
     640x480 dan lo mismo.
2. **La prueba no separó a los detectores de CPU (efecto techo).** Para
   distinguirlos harían falta escenarios difíciles: varias personas,
   contraluz, oclusión (mano o mascarilla), luz baja, rostros a más de 3 m y
   sin_cara con muchos fondos. De todos modos, SCRFD y los YOLO **no caben
   en tiempo real en la Pi** (0,5–0,7 s por cuadro) y tienen licencias
   restrictivas (no comercial o AGPL).
3. **Lo que sí falla es el YuNet del sensor, que es el que usa ROMEO hoy.**
   Pierde la cara de perfil (3 % a 60°) y a veces de frente (86 % a 1,5 m).
   Hay dos causas medidas:
   - un umbral más estricto por la fórmula (producto contra raíz);
   - puntajes más bajos del modelo int8 con la cara girada.
4. **Hallazgos sobre ROMEO para corregir** (el equipo decide; no se tocó
   nada):

   | Hallazgo | Estado | Corrección posible |
   |---|---|---|
   | Colores invertidos | MEDIDO | pedir "RGB888", o convertir en `_a_bgr` |
   | Umbral del sensor | MEDIDO | comparar √(cls·obj), o usar 0,49 |
   | Log de fps que cuenta intentos | MEDIDO | contar solo los cuadros con `probs` |
   | Cuadros sin salida del sensor descartados | MEDIDO | reintentar el pedido o caer a YuNet CPU |
   | FER+ que da siempre neutral | MEDIDO | evaluarlo en la fase 2 |

5. **Fase 2** (no iniciada; espera la orden del equipo): clasificador de
   emoción facial, HSEmotion contra alternativas. Los puntos centrales son el
   sesgo al hablar, el orden de colores y que FER+ responde neutral a todo.

## 8. Pruebas en vivo, modelo por modelo (20-11-2026)

**Para qué.** Mostrar en tiempo real cómo cada detector interactúa con la
persona y cómo se recolectan los datos. Es una demostración que complementa
la comparación oficial de las secciones 5 a 7, y **no la reemplaza**.

**Cómo se hizo** (`ver_en_vivo.py camara --detectores <uno> --umbral-todos 0.7`):
- **Un modelo a la vez**, para no sobrecargar la Pi. Corre sobre la imagen en
  vivo de la cámara y procesa tantos cuadros como alcanza.
- **Umbral 0,70 para los cinco**, decidido con el equipo, cada uno sobre su
  propio puntaje. Para SCRFD y los YOLO es más exigente que su valor por
  defecto (0,50 y 0,25), el que se usó en las secciones 5 a 7.
- **Una ventana de medición por escenario**, que se abre con el "listo" de
  la persona: 20 s, y 30 s en sin_cara. Se cuentan los cuadros procesados en
  los que el modelo encontró al menos una cara.
- La persona estuvo **sin lentes** en las distancias y los ángulos; aparte
  se midió 1 m con lentes.
- **No se guardó ninguna imagen**, solo los resúmenes en texto
  (`resultados/vivo/vivo_*.log` y `vivo_resumen.csv`).
- La Pi tenía la cámara derecha y el cable reemplazado (ver incidencias).

**Resultados [MEDIDO, `resultados/vivo/vivo_resumen.csv`]** (k/n, % e IC95 de Wilson):

| Escenario | yunet_cpu | **yunet_sensor** | scrfd_2g5 | yolov8n_face | yolo11n_face |
|---|---|---|---|---|---|
| 0,5 m (sin lentes) | 286/286 (100 %; IC 99–100) | **257/301 (85 %; IC 81–89)** | 20/34 (59 %; IC 42–74) | 25/25 (100 %; IC 87–100) | 30/30 (100 %; IC 89–100) |
| 1 m (sin lentes) | 296/296 (100 %; IC 99–100) | **279/301 (93 %; IC 89–95)** | 35/35 (100 %; IC 90–100) | 25/25 (100 %; IC 87–100) | 31/31 (100 %; IC 89–100) |
| 1,5 m (sin lentes) | 298/298 (100 %; IC 99–100) | 301/301 (100 %; IC 99–100) | 35/35 (100 %; IC 90–100) | 25/25 (100 %; IC 87–100) | 31/31 (100 %; IC 89–100) |
| 2 m (sin lentes) | 299/299 (100 %; IC 99–100) | **212/301 (70 %; IC 65–75)** | 34/34 (100 %; IC 90–100) | 25/25 (100 %; IC 87–100) | 30/30 (100 %; IC 89–100) |
| 3 m (sin lentes) | **267/299 (89 %; IC 85–92)** | **273/301 (91 %; IC 87–94)** | 35/35 (100 %; IC 90–100) | 25/25 (100 %; IC 87–100) | 30/30 (100 %; IC 89–100) |
| 30° (sin lentes) | 296/296 (100 %; IC 99–100) | **156/301 (52 %; IC 46–57)** | **16/34 (47 %; IC 32–63)** | 25/25 (100 %; IC 87–100) | 30/30 (100 %; IC 89–100) |
| 60° (sin lentes) | 297/297 (100 %; IC 99–100) | **82/301 (27 %; IC 22–32)** | **25/35 (71 %; IC 55–84)** | 25/25 (100 %; IC 87–100) | 31/31 (100 %; IC 89–100) |
| 1 m con lentes | 298/298 (100 %; IC 99–100) | **265/301 (88 %; IC 84–91)** | 35/35 (100 %; IC 90–100) | 25/25 (100 %; IC 87–100) | 30/30 (100 %; IC 89–100) |
| Sin cara (FP) | 0/448 (0 %; IC 0–1) | 0/451 (0 %; IC 0–1) | 0/52 (0 %; IC 0–7) | 0/37 (0 %; IC 0–9) | 0/45 (0 %; IC 0–8) |

En yunet_sensor, n son los cuadros **con** salida del sensor. En cada
ventana hubo además tantos cuadros **sin** salida como con ella (por
ejemplo, 301 con salida y 301 sin, "sd301" en el registro), lo que confirma
en vivo que el sensor responde en 1 de cada 2 cuadros (hallazgo 2 de la
sección 4).

**Agregado por grupo de escenarios [MEDIDO, `resultados/vivo/vivo_agregado.csv`]:**

| Grupo | yunet_cpu | yunet_sensor | scrfd_2g5 | yolov8n_face | yolo11n_face |
|---|---|---|---|---|---|
| Uso real (0,5–2 m de frente + 30°) | 1475/1475 (100 %; IC 100–100) | **1205/1505 (80 %; IC 78–82)** | 140/172 (81 %; IC 75–87) | 125/125 (100 %; IC 97–100) | 152/152 (100 %; IC 98–100) |
| Difíciles (3 m + 60°) | 564/596 (95 %; IC 93–96) | **355/602 (59 %; IC 55–63)** | 60/70 (86 %; IC 76–92) | 50/50 (100 %; IC 93–100) | 61/61 (100 %; IC 94–100) |
| Todos con cara (8 escenarios) | 2337/2369 (99 %; IC 98–99) | **1825/2408 (76 %; IC 74–77)** | 235/277 (85 %; IC 80–89) | 200/200 (100 %; IC 98–100) | 243/243 (100 %; IC 98–100) |
| Falsas alarmas (sin_cara) | 0/448 (0 %) | 0/451 (0 %) | 0/52 (0 %) | 0/37 (0 %) | 0/45 (0 %) |

**La regla de la sección 2, aplicada a esta demostración** (orientativo: los
cuadros son consecutivos y los IC con n son optimistas):
- **Uso real, yunet_cpu contra el mejor competidor (yolo11n_face):** 100 %
  [100–100] contra 100 % [98–100]. Los intervalos se solapan: **EMPATE**.
  YuNet es unas 10 veces más rápido (~14 contra ~1,5 cuadros por segundo),
  así que **YuNet se queda**. Coincide con la sección 7.
- **yunet_sensor contra yunet_cpu (mismo modelo):** 80 % [78–82] contra
  100 % [100–100] en uso real. Los intervalos están muy separados, y en los
  difíciles la diferencia es aún mayor (59 % contra 95 %). Aunque la muestra
  efectiva sea menor que n, una diferencia de 20 a 36 puntos no se explica
  por la correlación entre cuadros [HIPÓTESIS razonable, no calculada con
  n_ef]. **El sensor es el punto débil de ROMEO.**
- **Los modelos nuevos no aportan nada que justifique su costo**, porque
  YuNet CPU ya está en ~100 % en uso real. Lo único en que superaron a YuNet
  CPU fue a 3 m (100 % contra 89 %).

**Costo en vivo [MEDIDO]:**

| Modelo | Cuadros por segundo | Temperatura máxima de la Pi |
|---|---|---|
| yunet_cpu | ~14 | normal |
| yunet_sensor | ~15 con salida (de ~30 de la cámara) | 61 °C: la red corre en el IMX500 |
| scrfd_2g5 | ~1,7 | **81 °C**: se detuvo para enfriar |
| yolov8n_face | ~1,25 | 74 °C |
| yolo11n_face | ~1,5 | 74 °C |

**Lectura:**
- **[MEDIDO]** Los dos YOLO detectaron en todos los cuadros de los 8
  escenarios con cara, y ninguno de los cuatro dio falsas alarmas.
- **[MEDIDO]** YuNet CPU solo perdió cuadros a 3 m (89 %), donde la cara
  mide unos 20 px a 320x240. Los otros tres trabajan a 640x640.
- **[MEDIDO]** SCRFD, con umbral 0,70, perdió cuadros muy cerca y con la
  cara girada. En las secciones 5 a 7, con su umbral 0,50, daba 100 % en
  esos escenarios.
  **[HIPÓTESIS]** Con la cara girada, su puntaje cae entre 0,5 y 0,7. No se
  guardó el puntaje crudo cuadro a cuadro para confirmarlo.
- **[MEDIDO]** Los modelos nuevos procesan de 1,25 a 1,7 cuadros por
  segundo, por debajo de los 5 que usa el FER de ROMEO. SCRFD llevó la Pi al
  límite térmico.
- **[MEDIDO]** **El YuNet del sensor, el que usa ROMEO, es el que más
  falla:**
  - 52 % a 30° y 27 % a 60°, con intervalos que no se solapan con el 100 %
    de YuNet CPU;
  - además, irregular de frente: 85 % a 0,5 m, 70 % a 2 m y 100 % a 1,5 m.

  Confirma en vivo lo de la sección 5.4: el mismo modelo, comprimido a int8
  y con un umbral más exigente (cls·obj ≥ 0,70, equivalente a 0,84 en la
  CPU), pierde caras que la versión de CPU sí ve. No dio falsas alarmas
  (0/451).

**Limitaciones (importantes):**
- Los IC suponen cuadros independientes, y aquí son consecutivos (hasta 14
  por segundo en YuNet). **La muestra efectiva es menor y los intervalos
  reales son más anchos** [HIPÓTESIS sobre la magnitud; no se calculó n_ef
  porque no se guardó la serie cuadro a cuadro].
- Cada modelo se midió en una toma distinta: no son los mismos cuadros. Las
  diferencias entre modelos incluyen variaciones de pose, gesto y distancia
  entre tomas.
- Los modelos lentos tienen pocos cuadros por escenario (25 a 35): un 100 %
  con 25 cuadros es menos sólido que con 290.
- **La decisión de la sección 7 no cambia.** YuNet se queda: en detección
  empata con los demás salvo a 3 m, y es unas 10 veces más rápido.

**Incidencias de la sesión [MEDIDO]:**
- Tras mover la Pi, la cámara dejó de detectarse: el RP2040 de la cámara
  daba un error de I/O (-5) en el bus I2C. Se resolvió **cambiando el cable
  plano**. Con un cable intermedio la conexión era intermitente ("Camera
  frontend has timed out").
- Las dos primeras tomas en vivo se hicieron con la **cámara girada 180°**:
  YuNet daba un puntaje mediano de 0,67, y 0,93 al girar la imagen. Se
  descartaron, se borraron y se repitieron con la cámara derecha.
- Una toma se perdió por desconectar la Pi sin apagarla: los archivos
  quedaron en 0 bytes. Desde entonces `ver_en_vivo.py` hace `os.sync()` al
  terminar cada grabación.
- La IP de la Pi cambia en cada arranque (DHCP de la universidad).

## Archivos

| Archivo | Contenido |
|---|---|
| `grabar_caras.py` | graba los escenarios |
| `comparar_caras.py` | corre los detectores, calcula las métricas y aplica la regla |
| `descargar_modelos.py` | baja los 3 modelos nuevos, con verificación de CRC |
| `efecto_color.py` | YuNet y los FER con el color correcto o invertido |
| `ver_en_vivo.py` | dashboard en el navegador, con tres modos: `grabado` (repite la prueba guardada), `camara` (en vivo, por escenario) y `prueba` (graba y procesa a la vista) |
| `resultados/vivo/` | registros, `vivo_resumen.csv` (por escenario) y `vivo_agregado.csv` (por grupo) de las pruebas en vivo (sección 8) |
| `resultados/caras.csv`, `.json`, `mosaicos/`, `efecto_color_*.json` | resultados; se quedan en la Pi y no van al repo |
| `resultados_comparar.log`, `resultados_color.log` | salida completa de las corridas (en la Pi) |
| `datos/meta/` | metadatos por cuadro, `sensor_info.json` y `grabacion.log` |
