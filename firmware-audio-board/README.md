# Firmware de la placa de audio — ESP32-S3-AUDIO-Board

Firmware que convierte la **Waveshare ESP32-S3-AUDIO-Board** en una **tarjeta de sonido USB** (USB Audio Class 2.0) para la Raspberry Pi de ARFORD.

Con esto, la placa aporta el hardware que a ARFORD le faltaba —micrófonos y altavoz— y la Pi la ve como un dispositivo de audio más, sin protocolos propios ni nada por red.

> **Estado: compilado, grabado y funcionando a medias.**
>
> - ✅ La Raspberry Pi la reconoce como tarjeta de sonido: `card 3: Lora Audio Board`, visible en `arecord -l` y en `aplay -l`. El USB cambió de identidad, de `303a:1001` (puerto serie) a `303a:8000` (audio).

> **El nombre USB sigue siendo `Lora Audio Board`**, no ARFORD: lo fija `CONFIG_UAC_TUSB_PRODUCT` en `sdkconfig.defaults` y es lo que la placa ya flasheada reporta. Los comandos de este README usan `plughw:CARD=Lora` por eso. Cambiarlo requiere editar ese valor y **reflashear**, así que se deja como está hasta que haya otro motivo para volver a grabar la placa.
> - ⚠️ **Altavoz:** `aplay` reproduce sin errores, pero no está confirmado de oído.
> - ❌ **Micrófono:** entrega **silencio digital exacto** (RMS 0.0, pico 0 sobre 80.000 muestras). No es ruido bajo: son ceros.
>
> Ver [Estado y riesgos](#estado-y-riesgos) para el diagnóstico en curso.

---

## Por qué así

La placa venía con el firmware **Xiaozhi**, un asistente de voz chino que manda el audio a sus propios servidores. Servía para comprobar que el hardware funciona, pero no encaja en ARFORD: habla chino, depende de servidores externos y usa un protocolo propio.

Se evaluaron tres caminos:

| Camino | Por qué se descartó o eligió |
|---|---|
| Servidor `xiaozhi-esp32-server` en la Pi | ❌ Pide 4-8 GB de RAM y duplica el LLM y el TTS que ARFORD ya tiene |
| Protocolo propio por puerto serie | ❌ Hay que inventar y mantener el protocolo a los dos lados |
| **Tarjeta de sonido USB (UAC)** | ✅ **Elegido.** La Pi la trata como audio estándar: `arecord` y `aplay` funcionan sin escribir nada |

La ventaja decisiva del UAC es que el STT que ya funciona en la Pi (`lora_drivers/_stt_engine.py`, whisper-tiny a 2.45 s por frase) recibe el audio **sin ningún cambio**: para él es un micrófono normal del sistema.

---

## Cómo queda el flujo

```
Micrófonos ──> ES7210 ──I2S──> ESP32-S3 ──USB (UAC)──> Raspberry Pi
                                                          │
                                                          ├─> STT (whisper-tiny, español)
                                                          ├─> orchestrator_node
                                                          └─> Piper (voz es_MX)
                                                          │
 Altavoz <── ES8311 <──I2S── ESP32-S3 <──USB (UAC)────────┘
```

La placa **no** hace reconocimiento de voz ni síntesis: solo capta y reproduce. Todo el procesamiento sigue en la Pi. Ver el [README del workspace](../ros2_ws/README.md) para esa parte.

---

## Hardware

Datos tomados del diagrama oficial de pines de Waveshare.

### Códecs

| Chip | Función | Dirección I2C |
|---|---|---|
| **ES8311** | Códec de salida (altavoz), mono | `0x18` |
| **ES7210** | ADC de entrada (micrófonos), 4 canales TDM | `0x40` |
| **TCA9555** | Expansor de E/S | `0x20` |
| PCF85063 | Reloj de tiempo real (sin usar aquí) | `0x51` |

### Pines

| Señal | GPIO |
|---|---|
| I2C SDA | 11 |
| I2C SCL | 10 |
| I2S MCLK | 12 |
| I2S BCLK (SCLK) | 13 |
| I2S LRCK (WS) | 14 |
| I2S DOUT (hacia el ES8311, altavoz) | **16** |
| I2S DIN (desde el ES7210, micrófonos) | **15** |

> ⚠️ **El diagrama oficial de Waveshare da estos dos pines al revés** (`DOUT=15`, `DIN=16`). Dos proyectos independientes de la comunidad para esta misma placa coinciden en que el correcto es `DIN=15` / `DOUT=16`, y el síntoma lo confirma: con el mapeo oficial el micrófono devolvía **ceros exactos sin un solo error en el log**, que es justo lo que ocurre al leer el pin del altavoz esperando encontrar micrófono. Los demás pines del diagrama oficial sí son correctos.
| Habilitación del amplificador (PA_EN) | **EXIO8** del TCA9555, no un GPIO |

El amplificador NS4150B **no se enciende con un GPIO del chip**: cuelga del expansor TCA9555, en el bit 0 del puerto 1. Sin activarlo, el altavoz no suena aunque el ES8311 esté bien configurado. Es el fallo más probable al primer intento.

---

## Compilar y grabar

Requiere **ESP-IDF v5.4.2** o superior. En la Raspberry Pi (arquitectura arm64):

```bash
# Una sola vez
git clone -b v5.4.2 --depth 1 --recursive --shallow-submodules \
    https://github.com/espressif/esp-idf.git ~/esp-idf
cd ~/esp-idf && ./install.sh esp32s3

# En cada sesión
. ~/esp-idf/export.sh

# Compilar y grabar
cd firmware-audio-board
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/ttyACM0 flash monitor
```

La placa aparece como `/dev/ttyACM0` en la Pi. Para confirmar cuál es, con ruta estable:

```bash
ls -l /dev/serial/by-id/
# usb-Espressif_USB_JTAG_serial_debug_unit_28:84:85:B2:BF:BC-if00 -> ../../ttyACM0
```

---

## Comprobar que funciona

Tras grabar y reconectar, la Pi debería listarla como dispositivo de audio:

```bash
arecord -l      # tiene que aparecer la placa como captura
aplay -l        # y como reproducción

# Grabar 5 segundos y transcribir con el STT que ya está montado
arecord -D plughw:CARD=Lora -f S16_LE -r 16000 -c 1 -d 5 /tmp/prueba.wav
~/.lora/venv/bin/python ~/lora-stt/scripts/bench_stt.py --wav /tmp/prueba.wav
```

Si `arecord -l` no muestra nada, el firmware no está enumerando como UAC: mirar la salida de `idf.py monitor`.

---

## Volver atrás

El firmware Xiaozhi original está respaldado **completo** (los 16 MB de flash):

```
Raspberry Pi: ~/xiaozhi-firmware-backup.bin
md5: a217bc9cecac925846bd26ca81574b86
```

Para restaurarlo:

```bash
~/.lora/venv/bin/esptool --port /dev/ttyACM0 --baud 921600 \
    write-flash 0x0 ~/xiaozhi-firmware-backup.bin
```

Con eso la placa vuelve a ser el asistente Xiaozhi tal como vino de fábrica.

---

## Estado y riesgos

### Errores reales encontrados al ponerlo en marcha

Los cuatro se descubrieron compilando y grabando, no leyendo documentación. Quedan aquí porque son trampas repetibles con estos componentes:

1. **`bus_handle` obligatorio en I2C.** `audio_codec_i2c_cfg_t` exige el manejador del bus desde IDF 5.3. Sin él, `audio_codec_new_i2c_ctrl()` devuelve `NULL` **en silencio**, y el códec falla luego con un engañoso `Wrong codec config` que hace pensar que el chip no responde.
2. **`rx_handle` / `tx_handle` obligatorios en I2S.** Mismo patrón: en IDF 5.x el bus I2S no lo crea `esp_codec_dev`, hay que instalarlo antes con `i2s_new_channel()` y pasarle los manejadores. Con `NULL` **compila igual**, pero el códec se queda sin camino de datos.
3. **Las direcciones I2C van en formato de 8 bits.** `audio_codec_ctrl_i2c.c` hace `.device_address = (i2c_cfg->addr >> 1)`. El diagrama de Waveshare las da en 7 bits (`0x18`, `0x40`), así que hay que duplicarlas: **`0x30` y `0x80`**, que es justo lo que definen `ES8311_CODEC_DEFAULT_ADDR` y `ES7210_CODEC_DEFAULT_ADDR`. Pasar el valor de 7 bits produce NACK.
   El TCA9555 es la excepción: se maneja con la API cruda de IDF, que usa 7 bits sin desplazar.
4. **`set_mute_cb` y `set_volume_cb` devuelven `void`**, no `esp_err_t`.

### Lo que sigue abierto: el micrófono da silencio

El altavoz, el I2C, el expansor y los relojes I2S funcionan. El micrófono devuelve ceros exactos, sin ningún error en el log. Dos hipótesis:

- **DIN y DOUT intercambiados.** El diagrama de Waveshare dice `GPIO15 = DOUT` y `GPIO16 = DIN`; una fuente de la comunidad lo dice al revés.
- **TDM contra I2S estándar.** El firmware de fábrica configuraba el ES7210 en **TDM de 4 canales** (`MIC1`–`MIC4`); aquí está en Philips estéreo con 2 micrófonos.

Para decidirlo con datos en vez de a base de grabar a ciegas existe la **compilación de diagnóstico** (`sdkconfig.diag` + `LORA_MODO_DIAG`), que no activa el USB de audio —así conserva el puerto serie y sus logs— y mide el nivel de señal en las cuatro combinaciones posibles.

### Al grabar se pierde el acceso por USB

Cuando TinyUSB toma el bus, la placa deja de exponer puerto serie: las tres interfaces USB pasan a ser de clase Audio, `/dev/ttyACM0` desaparece y `esptool` ya no puede alcanzarla.

**Para volver a grabarla hay que entrar en modo descarga físicamente:** mantener pulsado **BOOT**, desconectar y reconectar el USB, y soltar. No hay alternativa por software.

---

## Archivos

| Archivo | Qué es |
|---|---|
| `main/lora_audio_main.c` | El firmware |
| `main/idf_component.yml` | Dependencias (`usb_device_uac`, `esp_codec_dev`) |
| `main/CMakeLists.txt` | Componente principal |
| `CMakeLists.txt` | Proyecto |
| `sdkconfig.defaults` | Target, flash de 16 MB y PSRAM octal |
