# Firmware del carrito mecanum — ESP32-S3-DevKitC-1

Firmware que convierte un **ESP32-S3-DevKitC-1** en el controlador de motores del carrito mecanum de ARFORD. Recibe comandos de texto por USB desde la Raspberry Pi y los traduce a señales para dos drivers **L298N**.

> **Estado: funcionando.** Verificado en el robot el 13 de octubre de 2026.
>
> - ✅ Las 4 ruedas giran hacia adelante con `F`, comprobado rueda por rueda con `T1`..`T4`.
> - ✅ Las 4 diagonales (`FL`/`FR`/`BL`/`BR`) responden desde la página de control.
> - ⚠️ **Rotaciones sin verificar de oído:** `rotateLeft`/`rotateRight` se intercambiaron al corregir el lado izquierdo (ver [Historia](#historia-de-las-correcciones)). Conviene mandar `RL` y confirmar que gira a la izquierda.

---

## Por qué el firmware vive acá

El ESP32 no decide nada: solo obedece. Toda la lógica (qué movimiento toca en cada momento de la trivia, cuánto dura, cuándo parar) está en la Pi, en `romeo_drivers/moving_control_node.py`. El firmware es deliberadamente tonto.

Esa división tiene una razón práctica: cambiar el comportamiento del robot no debería obligar a reflashear una placa. Reflashear pide parar ROS, liberar el puerto serie y dejar el robot fuera de servicio unos minutos; cambiar un nodo de Python es un `colcon build`.

**Sin WiFi, a propósito.** Una versión anterior abría un Access Point propio y servía su propia página de botones. Se quitó porque la Pi ya sirve esa página (`romeo_drivers/_control_web.py`) y mantener dos páginas que hacen lo mismo es trabajo duplicado. La página de la Pi gana porque está donde está la lógica y puede, además, ver el estado del resto del robot.

---

## Cómo queda el flujo

```
Pagina de control (telefono) ─┐
                              ├─> moving_control_node ──> CartSerial ──USB 115200──> ESP32-S3
orchestrator_node (trivia) ───┘    (_control_web.py)     (_cart_serial.py)              │
                                                                                        │ 8 señales
                                                                                        v
                                                                               2x L298N ──> 4 motores
```

Un comando es **una línea de texto terminada en `\n`**. Nada más: ni binario, ni checksum, ni respuesta. Se lee la línea completa y no byte a byte, para no ejecutar un comando a medio escribir.

---

## Comandos

| Comando | Movimiento |
|---|---|
| `F` / `B` | Adelante / atrás |
| `SL` / `SR` | Desplazamiento lateral (*strafe*) izquierda / derecha |
| `RL` / `RR` | Rotación sobre su eje izquierda / derecha |
| `FL` / `FR` | Diagonal adelante-izquierda / adelante-derecha |
| `BL` / `BR` | Diagonal atrás-izquierda / atrás-derecha |
| `S` | Parar |

Cualquier texto no reconocido también para el carrito: ante la duda, quieto.

### Comandos de diagnóstico

| Comando | Rueda | Pines |
|---|---|---|
| `T1` | Front Left | IN1/IN2 = GPIO 17/18 |
| `T2` | Front Right | IN3/IN4 = GPIO 16/15 |
| `T3` | Back Left | IN5/IN6 = GPIO 6/7 |
| `T4` | Back Right | IN7/IN8 = GPIO 5/4 |

Mueven **una sola rueda** hacia adelante. Existen porque tres fallos muy distintos se ven igual desde afuera —mapeo de pines equivocado, un canal del driver muerto, alimentación insuficiente— y moviendo el carrito entero no se distinguen. Rueda por rueda, sí.

No los usa ROS; son para probar a mano.

---

## El watchdog de 500 ms

Si no llega ningún comando en 500 ms, el firmware **para los motores solo**.

```c
if (millis() - lastCmdTime > CMD_TIMEOUT) {
  stopCar();
}
```

Protege contra el cable desconectado, el proceso de Python colgado y la página de control cerrada a mitad de un movimiento. Sin esto, un `F` dejaría el carrito andando hacia la pared indefinidamente.

**La consecuencia es que quien manda el comando tiene que repetirlo.** Un `F` suelto da un tirón de medio segundo y nada más. Los dos caminos de ARFORD ya lo hacen:

- La página de control repite el comando cada pocos ms mientras se mantiene apretado el botón (`_control_web.py`).
- `CartSerial` tiene su propio watchdog de 3 s que manda `S` si dejan de llegar órdenes (`_cart_serial.py`).

Si se prueba a mano, hay que repetir igual:

```bash
# 1 segundo de avance (5 pulsos a 5 Hz) y parada explicita
for i in $(seq 1 5); do echo "F" > /dev/ttyACM0; sleep 0.2; done
echo "S" > /dev/ttyACM0
```

---

## Mapeo de pines

```
              FRENTE
    ┌───────────────────────┐
  ╔═╧═╗ IN1=17        IN3=16 ╔═╧═╗
  ║FL ║ IN2=18        IN4=15 ║FR ║
  ╚═╤═╝                      ╚═╤═╝
    │      ┌───────────┐       │
    │      │ ESP32-S3  │       │
    │      │  USB→Pi   │       │
    │      └───────────┘       │
  ╔═╧═╗ IN5=6         IN7=5  ╔═╧═╗
  ║BL ║ IN6=7         IN8=4  ║BR ║
  ╚═╤═╝                      ╚═╤═╝
    └───────────────────────┘
               ATRÁS
```

Se evitan los GPIO 0, 3, 19, 20, 26-32 y 43-46: en el ESP32-S3 están reservados o no expuestos. El sketch original (para ESP32 clásico) usaba 26-33, que acá no sirven.

### ⚠️ ENA/ENB del L298N

El firmware **no genera PWM**: solo `digitalWrite` en HIGH/LOW. Nunca toca los pines *enable* del L298N, así que **ENA y ENB tienen que estar puenteados a 5 V** con sus jumpers.

Si están al aire, ningún comando mueve nada y el síntoma es idéntico a un firmware roto. Es el primer sitio donde mirar si el carrito no responde.

Como contrapartida, no hay control de velocidad: los motores van a tope o parados.

---

## Compilar y flashear

Desde la Raspberry Pi, con [`arduino-cli`](https://arduino.github.io/arduino-cli/):

```bash
# Instalacion del core ESP32 (una sola vez)
arduino-cli config init
arduino-cli config add board_manager.additional_urls \
  https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
arduino-cli core update-index
arduino-cli core install esp32:esp32
```

**Antes de flashear hay que liberar el puerto serie**, que `moving_control_node` mantiene abierto mientras está activo. Si no, el upload falla con *port busy*:

```bash
# Parar ROS (deja el robot fuera de servicio)
pkill -f "ros2 launch romeo_bringup"
# Confirmar que el puerto quedo libre
sudo fuser -v /dev/ttyACM0   # sin salida = libre
```

La carpeta del sketch debe llamarse igual que el `.ino`, así que se compila desde una copia:

```bash
mkdir -p /tmp/mecanum_car_esp32s3
cp mecanum_car_esp32s3.ino /tmp/mecanum_car_esp32s3/
arduino-cli compile --fqbn esp32:esp32:esp32s3 /tmp/mecanum_car_esp32s3
arduino-cli upload -p /dev/ttyACM0 --fqbn esp32:esp32:esp32s3 /tmp/mecanum_car_esp32s3
```

Y después relanzar ROS:

```bash
source /opt/ros/lyrical/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch romeo_bringup romeo_bringup.launch.py
```

Si la placa no entra en modo flash sola: mantener **BOOT**, pulsar **RESET**, soltar BOOT, y repetir el upload.

### `credentials.h`

Este firmware **no usa WiFi**, así que no hace falta. El `credentials.example.h` se conserva solo por si alguien retoma la variante con Access Point.

---

## Historia de las correcciones

Tres fallos de mapeo, encontrados en este orden. Vale la pena conservarlos porque los síntomas se parecen mucho entre sí y es fácil confundirlos.

| # | Síntoma | Causa | Arreglo |
|---|---|---|---|
| 1 | Las diagonales llevaban al lado contrario | `forwardLeft`/`forwardRight`/`backLeft`/`backRight` estaban cruzadas izq/der | Corregir los cuerpos de las 4 funciones |
| 2 | Adelante/atrás y diagonales invertidas, **pero la rotación correcta** | Chasis montado 180° respecto a lo que asumía el código | Rotar el mapeo: cada rueda a su opuesta en diagonal |
| 3 | El carrito **rotaba en vez de avanzar**: FL y BL hacia atrás, FR y BR hacia adelante | Polaridad invertida solo del lado izquierdo | Intercambiar el par de pines de FL y BL, **e** intercambiar los cuerpos de `rotateLeft`/`rotateRight` |

El detalle que distingue el #2 del #3 es la rotación:

- **Una inversión de los 4 motores** da vuelta también el giro.
- **Una rotación de 180° del chasis** lo preserva.
- **Invertir un solo lado** da vuelta el giro pero no el avance.

Por eso el #3 obligó a intercambiar `rotateLeft`/`rotateRight`: sin eso, `RL` habría girado a la derecha.

El #3 se diagnosticó con los comandos `T1`..`T4`, que se agregaron para eso. Mover el carrito entero no permitía distinguirlo de un problema de alimentación.

---

## Si el carrito no se mueve

En orden, de más probable a menos:

1. **¿ENA/ENB del L298N tienen los jumpers puestos?** Sin ellos no hay movimiento posible (ver arriba).
2. **¿Se está repitiendo el comando?** Un comando suelto dura 500 ms por el watchdog.
3. **¿Alguien más tiene el puerto?** `sudo fuser -v /dev/ttyACM0`. Si `moving_control_node` está activo, lo tiene abierto y los comandos a mano se mezclan con los suyos. Mejor mandarlos por ROS o por la página.
4. **¿Es el puerto correcto?** El ESP32 lleva un CH34x y aparece como `/dev/ttyACM0` o `/dev/ttyACM1` según el orden de enumeración. `CartSerial._resolver_puerto()` lo busca por `/dev/serial/by-id/` (contiene `1a86`), que sí es estable.
5. **¿Alimentación de los motores?** Los 4 motores a la vez dan un pico de corriente que puede provocar un brownout. Si una rueda sola gira (`T1`..`T4`) pero las cuatro juntas no, es esto.
6. **¿Cada rueda responde?** `T1`..`T4` una por una. Si gira otra de la esperada, el mapeo está cruzado.

---

## Archivos

| Archivo | Qué es |
|---|---|
| `mecanum_car_esp32s3.ino` | El firmware |
| `credentials.example.h` | Plantilla de WiFi, sin uso en esta versión |

---

## Ver también

- [`ros2_ws/src/romeo_drivers/romeo_drivers/_cart_serial.py`](../ros2_ws/src/romeo_drivers/romeo_drivers/_cart_serial.py) — el cliente que manda los comandos
- [`ros2_ws/src/romeo_drivers/romeo_drivers/_control_web.py`](../ros2_ws/src/romeo_drivers/romeo_drivers/_control_web.py) — la página de control manual
- [`README.md`](../README.md) — el robot completo
- [`firmware-audio-board/README.md`](../firmware-audio-board/README.md) — el otro firmware de ARFORD
