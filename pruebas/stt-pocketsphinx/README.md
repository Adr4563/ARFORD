# Prueba: audio → texto con PocketSphinx (sin redes neuronales)

Banco de pruebas **aislado** del robot: no importa ROS ni toca ningún nodo.
Mide si PocketSphinx transcribe en español lo bastante bien, en la Raspberry
Pi y con el micrófono de la audio board.

PocketSphinx usa **modelos estadísticos clásicos** (HMM con mezclas
gaussianas y n-gramas), no redes neuronales. Esos modelos **sí están
entrenados** con horas de voz: "sin IA neuronal" se cumple, "sin ningún
modelo entrenado" no.

| Modelo | Origen | Qué trae |
|---|---|---|
| `ciempiess` (por defecto) | UNAM, español de México, 581 h | 10 000 senones, diccionario de 285 000 palabras, modelo de lenguaje |
| `cmu` | CMU Sphinx 5.2 (VoxForge) | modelo `cd_ptm_4000`, diccionario y modelo de lenguaje de 20 000 palabras |

## Archivos

| Archivo | Qué hace |
|---|---|
| `instalar_modelos.sh` | Instala `pocketsphinx` por apt y baja los dos modelos a `~/.romeo/modelos/sphinx/` |
| `en_vivo.py` | Micro de la placa → texto, frase a frase, con la latencia |
| `modelos.py` | Rutas de los modelos y cómo se lanza el decodificador |
| `gramatica.py`, `comandos.txt` | Modo `--comandos`: solo reconoce las frases del menú |
| `frases.txt` | Frases de prueba para medir WER/CER |

## En la Raspberry Pi

```bash
cd ~/ros2_ws && git pull
cd ~/ros2_ws/pruebas/stt-pocketsphinx
bash instalar_modelos.sh
python3 en_vivo.py                 # habla libre, CIEMPIESS
python3 en_vivo.py --modelo cmu    # el otro modelo
python3 en_vivo.py --comandos      # solo el menú
```

Si el robot está corriendo, páralo antes: tiene ocupado el micrófono.

## Lo que ya se midió (25 oct 2026, en la Pi)

Con una frase sintética de 3.1 s ("hola me llamo adrian y quiero jugar
trivia de ciencia", voz Piper, no voz real):

- Carga del modelo: **~9 s**. El modo `single` de pocketsphinx la repite en
  **cada archivo** (8.9 s aunque sea silencio). Por eso todo usa `live -`.
- `ciempiess` en `live`, misma frase 3 veces seguidas:
  1. "hola me llamo adrián ni quiero tocar te iría de ciencia"
  2. "hola me llamo adrián ni quiero jugar trivia de ciencia"
  3. "hola me llamo adrián ni quiero jugar trivia de ciencia"

  La primera sale peor porque la normalización se va ajustando al audio.
- `cmu`: "hola mike amarilla ni quiero ju cartilla de ciencia".
- Bajar los beams para ir más rápido destrozó el resultado de los dos
  modelos y casi no ganó tiempo: se dejan los valores por defecto.
- RAM: la Pi tiene ~7 GB libres con el robot parado.

**Falta medir con voz real.**

Dos arreglos que hace `instalar_modelos.sh` y sin los que `cmu` no funciona
bien: descomprime `es-20k.lm.gz` (pocketsphinx 5 no lee `.gz` y arranca sin
modelo de lenguaje) y rellena el `feat.params`, que en el paquete viene con
plantillas sin sustituir (`__CFG_NUM_FILT__`...).
