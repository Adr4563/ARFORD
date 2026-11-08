# Revisión pendiente

Lista actualizada el 8 de noviembre de 2026.

## Por arreglar

### 1. La cámara no entrega imagen

La Pi detecta la cámara (`imx500` aparece en `rpicam-hello --list-cameras`),
pero no recibe cuadros:

```
ERROR RPI pipeline_base.cpp:1356 Camera frontend has timed out!
```

`rpicam-still` tampoco logra sacar una foto, así que no es un problema del
código de Romeo. El firmware de la IMX500 está instalado y reiniciar la Pi no
lo resolvió. Sin imagen, FER no da lecturas y Romeo usa solo voz y texto.

Qué probar, con la Pi apagada:

1. Dar vuelta el cable flex en el conector de la Pi.
2. Probar con otro flex.
3. Si sigue igual, la cámara está dañada.

Verificar con:

```bash
rpicam-still -n --timeout 1500 --width 640 --height 480 -o /tmp/prueba.jpg
```

### 2. La rueda trasera derecha no gira hacia adelante

Con `T4B` gira hacia atrás y con `T4` no gira. Hacia atrás usa GPIO 4 y hacia
adelante GPIO 5, así que el motor, el driver y la batería responden y el fallo
está en la señal de GPIO 5 → IN3 del segundo L298N.

Qué probar: revisar ese cable en los dos extremos. Si sigue igual, intercambiar
IN3 e IN4 en el driver: si la falla cambia de sentido, es el cable o el pin del
ESP32; si no cambia, es la entrada IN3 del driver.

### 3. Respuesta a «¿cómo te sientes, Romeo?»

Si Romeo no está en Normal, la IA recibe a la vez el tono hacia la persona y
su propio ánimo, y responde mal («Me parece incómodo escuchar sobre tu dolor»).
La pregunta debería llevar solo el ánimo de Romeo (`aeb.tono_para`).

### 4. Comentarios del chat y de la trivia

- Con tristeza, el chat a veces entiende a la persona pero no le da ánimo.
- Se le escapó un «usted» («¿Cómo se siente?»).
- El dato curioso de la trivia puede ser inventado: dijo que en 1939 se
  inventó la radio.

## Por comprobar

No se han vuelto a medir desde que el STT pasó a la PC.

### 5. La placa de audio se desconectaba del USB

El 15 de octubre se midieron 65 desconexiones, cada 1-2 s. Desde entonces el
micrófono ha funcionado en las sesiones, pero no se revisó `dmesg`.

```bash
sudo dmesg -w | grep "1-1"
```

Si reaparece: conectar la placa directo a la Pi sin hub, cambiar el cable USB y
bajar el volumen.

### 6. Nivel del micrófono

El 15 de octubre la placa entregaba un pico del 4,3 % de la escala. El STT
todavía entiende mal algunas frases.

No tocar los parámetros del STT a ciegas: un ajuste de varios a la vez dejó al
VAD sin detectar voz. Grabar un WAV, medir el nivel y cambiar un parámetro por
vez. La ganancia del códec ES7210 solo se cambia en `firmware-audio-board/`.

### 7. Eco de Romeo en su propio micrófono

Ya no apareció en los logs: el audio se descarta mientras Romeo suena y el STT
responde rápido. `lanzar_romeo.sh` usa `ROMEO_GRACIA_ECO=0`. Si vuelve, subir
ese valor o activar el módulo `echo-cancel` de PipeWire.

## Por decidir o instalar

- **YuNet en el sensor IMX500:** falta dar permiso a debugfs en la Pi:
  ```bash
  echo "debugfs /sys/kernel/debug debugfs defaults,mode=0755 0 0" | sudo tee -a /etc/fstab
  sudo mount -o remount,mode=0755 /sys/kernel/debug
  ```
- **`ROMEO_NEUTRAL_ENCOGE`:** está activo en `lanzar_romeo.sh` y no es parte de
  la metodología de Rocha. Decidir si se deja o se usa solo la fórmula original.
- **Modelo SER:** se usa wav2vec2 MESD, aunque WavLM clasificó mejor los audios
  de prueba (`ROMEO_SER_MODELO`).
- **Medición de FER:** comparar CPU, RAM y temperatura con
  `fer_bajo_demanda` en `true` y `false` cuando la cámara dé imagen.
- **Pi sin actualizar:** falta `git pull`, compilar `romeo_brain` y copiar
  `pruebas/panel_robot/panel.py` a `~/romeo_panel.py`.
- **Scripts que solo están en la Pi:** `lanzar_tunel.sh`, `reiniciar_romeo.sh`,
  `reiniciar_panel.sh`, `apagar_romeo.sh` y `carrito_prueba.sh`.
