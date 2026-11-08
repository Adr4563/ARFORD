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

### 2. Ánimo con tristeza profunda (L3-L4)

Con gemma3:4b, 8 de 12 respuestas dieron ánimo sin preguntar. Las demás
cerraron con una pregunta o sin animar, y a veces el ánimo suena poético
(«la belleza de un atardecer»).

## Por decidir o instalar

- **YuNet en el sensor IMX500:** falta dar permiso a debugfs en la Pi:
  ```bash
  echo "debugfs /sys/kernel/debug debugfs defaults,mode=0755 0 0" | sudo tee -a /etc/fstab
  sudo mount -o remount,mode=0755 /sys/kernel/debug
  ```
- **Medición de FER:** comparar CPU, RAM y temperatura con
  `fer_bajo_demanda` en `true` y `false` cuando la cámara dé imagen.
- **Scripts que solo están en la Pi:** `lanzar_tunel.sh`, `reiniciar_romeo.sh`,
  `reiniciar_panel.sh`, `apagar_romeo.sh` y `carrito_prueba.sh`.
