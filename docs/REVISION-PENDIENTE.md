# Revisión pendiente

Dos problemas detectados interactuando con el robot el 15 de octubre de 2026.
Ninguno está resuelto.

## 1. La placa de audio se desconecta del USB

Medido en la Raspberry: **65 desconexiones**, reconectando cada 1-2 segundos.

```
[ 3841.536561] usb 1-1.3: USB disconnect
[ 3841.706034] usb 1-1.3: new full-speed USB device    <- 0.2 s despues
[ 3843.152678] usb 1-1.3: USB disconnect               <- 1.4 s despues
[ 3844.297085] usb 1-1.3: new full-speed USB device
[ 3845.097171] usb 1-1.3: USB disconnect
[ 3845.272091] usb 1-1.3: new full-speed USB device
```

Justo antes de cada caída aparece:

```
usb 1-1.3: cannot get connectors status: req = 0x81, wValue = 0x200, ...
```

Es el driver UAC perdiendo respuesta de la placa.

Cuando se cae, el micrófono muere a mitad de frase y whisper recibe audio
cortado. De ahí transcripciones como `'9. 3.'`, `'que se les en'` o
`'el otro de la coda'`.

### Qué se descartó

- No es software ni configuración: el puerto `1-1.3` desaparece de
  `/sys/bus/usb/devices/` y `lsusb` deja de verla.
- No es el hub entero: el `1-1.4` (CH340 del carrito) sigue vivo en el mismo
  hub mientras el `1-1.3` se cae.
- No es el nombre del dispositivo: `mic_dispositivo: "Board: USB Audio"` es
  correcto y coincide con un único dispositivo cuando la placa está conectada.

### Qué probar

1. **Conectar la placa directo a un puerto de la Pi**, sin el hub VIA Labs.
   El ESP32-S3 con el códec de audio pide corriente que un hub sin
   alimentación propia no sostiene.
2. **Cambiar el cable USB.** Los de solo carga dan exactamente este patrón.
3. **Bajar el volumen.** El comentario de `romeo_params.yaml` ya avisa de que
   con la música fuerte "la placa pedía demasiada corriente y se
   desconectaba". Está en 60: probar a 40.

Verificar con:

```bash
sudo dmesg -w | grep "1-1"
```

Si pasa 2 minutos sin `USB disconnect`, era el hub o el cable.

## 2. El robot se transcribe a sí mismo

El micrófono y el altavoz están en la misma placa, a centímetros. Cuando Romeo
habla, su propio micrófono lo oye, whisper lo transcribe y el orquestador lo
procesa como si fuera el usuario. Romeo acaba conversando consigo mismo.

Pares reales del log:

```
Asistente: "Me alegra escuchar que te sientes bien. ¿Cómo te sientes con
            respecto a la conexión con los..."
[mic]:     'de alegras, escucha que de cientes bien, como de cientes con
            respecto la conexión con'

Asistente: "Parece que estás experimentando un sentimiento de desesperanza..."
[mic]:     'parece que está se experimentando en sentimiento de desesperanza.'

Asistente: "¿Cómo te sientes en este momento?"
[mic]:     'que si antes en este momento'
```

También tomó **"Lora"** como nombre del usuario, captado de su propia voz
diciendo "Lora Audio Board".

### Por qué el filtro actual no basta

`communication_node` ya tiene anti-realimentación: `_mientras_suena()` marca
el altavoz como ocupado y `_on_texto_voz()` descarta lo que llegue mientras
`_sonando()` sea True.

El problema es el momento. whisper-tiny tarda **~4 s** en transcribir en la
Pi, así que:

1. Romeo habla -> `_reproduciendo = 1` -> el micrófono capta su voz
2. Romeo termina -> `_reproduciendo = 0`
3. **4 s después**, whisper acaba de transcribir ese audio
4. `_on_texto_voz` pregunta `_sonando()` -> ya es False -> lo publica

El filtro llega tarde. Es un bug real, no un problema de configuración.

### Qué probar

Una **ventana de gracia**: que `_sonando()` siga devolviendo True unos
segundos después de que el altavoz calle, cubriendo la latencia de whisper.

Se intentó con 5 s y funcionó para el eco, pero se desplegó junto a un timeout
de inactividad de 6 s y la combinación dejó a Romeo hablando solo sin dejar
hueco al usuario: cada frase suya bloqueaba el micrófono 5 s, y a los 6 s
volvía a hablar. **La ventana de gracia sola era correcta; el error fue
meterla junto al timeout.** Ambos se revirtieron.

La solución de fondo sería **cancelación de eco acústico (AEC)**. PipeWire
trae un módulo `echo-cancel` que no está activado.

## 3. Nota sobre el nivel del micrófono

Medido: la placa entrega un pico del **4,3%** de la escala (RMS 207). El
comentario de `_stt_engine.py` da por bueno un 26-31% hablándole de cerca, así
que estamos **6 veces por debajo**.

Con ese nivel whisper alucina: devuelve la misma palabra repetida
(`'nuevo nuevo nuevo...'`, `'escucha, escucha, escucha...'`,
`'bien, bien, bien, bien'`).

**No tocar los parámetros del STT a ciegas.** Se intentó ajustar
`PICO_MINIMO`, `GANANCIA_MAX`, `min_silence_duration`, el `threshold` del VAD
y `min_speech_duration`, más un filtro de repeticiones. Resultado: el VAD dejó
de detectar voz y el robot pasó de 13-14 transcripciones por sesión a **0**.
Todo revertido con git.

Si se retoma: grabar la voz del usuario a un WAV, medir el nivel real y
comparar la transcripción antes y después de cada cambio. Un parámetro a la
vez, midiendo, no varios a la vez.

El nivel bajo se origina en la ganancia del códec ES7210, que solo se cambia
en el firmware del ESP32-S3 (`firmware-audio-board/`), no desde la Pi.
