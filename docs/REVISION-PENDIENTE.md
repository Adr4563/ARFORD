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

### 2. Respuesta a «¿cómo te sientes, Romeo?»

Si Romeo no está en Normal, la IA recibe a la vez el tono hacia la persona y
su propio ánimo, y responde mal («Me parece incómodo escuchar sobre tu dolor»).
La pregunta debería llevar solo el ánimo de Romeo (`aeb.tono_para`).

### 3. Comentarios del chat y de la trivia

- Con tristeza, el chat a veces entiende a la persona pero no le da ánimo.
- Se le escapó un «usted» («¿Cómo se siente?»).
- El dato curioso de la trivia puede ser inventado: dijo que en 1939 se
  inventó la radio.

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
