# Romeo empático v2 — 23 de octubre de 2026

Reentrenamiento del modelo de chat de Romeo. Resuelve el defecto principal de la v1
y lo deja corriendo en el Raspberry Pi.

## El problema de la v1

`soup.yaml` pedía `target_modules: auto` y Soup eligió lo mínimo: solo `q_proj` y
`v_proj`. Son capas de atención — controlan el estilo, no el contenido. Las capas MLP
(`gate_proj`, `up_proj`, `down_proj`), que es donde el modelo guarda lo que sabe,
quedaron sin entrenar.

Resultado: el modelo aprendió una plantilla fija (exclamación + pregunta) y la repetía
pase lo que pase. No sabía su propio nombre y respondía "¿Te pones a llorar?" a la
muerte de una mascota.

## Qué se cambió

1. **`target_modules` explícito con las 7 capas.** El adaptador pasó de 4,3 MB a 35 MB.
2. **2 épocas en vez de 3.** En la v1 el `eval_loss` bajó 0,02 entre la época 1 y la 2;
   la tercera solo reforzaba la plantilla.
3. **Dataset ampliado** de 982 a 1.825 pares.

## Dataset

`data/romeo_empatico_v2.jsonl` — 1.825 pares

| Origen | Pares |
|---|---|
| Traducción previa | 982 |
| Traducidos hoy (ids 1001-1280) | 275 |
| Identidad y casos sensibles, escritos a mano (72 × 8) | 576 |

Los 72 de identidad van repetidos 8 veces a propósito: son pocos frente a los 1.257
traducidos, que empujan al modelo a cerrar toda respuesta con "¿cómo te fue?". Con peso
×1 y ×3 el tic seguía apareciendo.

Cubren: nombre y naturaleza de Romeo, saludos y despedidas, duelo, miedo, soledad,
tristeza y alegría.

### Ojo con el archivo fuente

De los tres archivos en Descargas, **`empathetic_4000_en.json` está corrompido**: alguien
le pasó un traductor automático que dejó texto mezclado e inservible ("I perdido 20 lbs",
"Puedo't stand snakes"). Afecta a ~1.000 de sus 4.000 registros.

**Usar `empathetic_4000_en_1.json` o `_2.json`**, que están limpios (3.944 correctos).

Se omitieron pares inadecuados como modelo de empatía (p. ej. el id 1044, un robo a una
anciana) y se suavizaron respuestas que juzgaban en vez de acompañar.

## Resultados

Entrenamiento: 218 pasos, 2 épocas, ~1 min en RTX A5000.
`train_loss` 1.00 · `eval_loss` 1.23 · precisión 0,74.

Lo que antes fallaba y ahora funciona:

| Entrada | Respuesta |
|---|---|
| ¿Cómo te llamas? | Me llamo Romeo. ¿Y tú cómo te llamas? |
| ¿Qué eres tú? | Soy Romeo, un robot. Me gusta conversar y hacerte compañía. |
| Se murió mi perro ayer | Lo siento muchísimo. ¿Cómo fue? |
| Tengo miedo | Es normal sentir eso. ¿Qué te preocupa? |

### Lo que sigue fallando

Qwen2.5-0.5B tiene 494 M de parámetros, de los cuales 136 M son embeddings: solo 358 M
hacen el trabajo conversacional. Con esa capacidad el modelo construye bien las frases
que vio seguido y se desvía en las demás. Aparecen respuestas incoherentes de vez en
cuando ("Sí, la gente está a cada rato" ante "me siento solo").

Más datos ayudan, pero el techo lo pone el tamaño del modelo.

## Despliegue en el Raspberry Pi

Pi: `10.143.99.116` (ARM64, 7,6 GB RAM). El nombre `raspberrypi.local` falló de forma
intermitente — conviene usar la IP.

```bash
ollama run romeo-empatico-v2-q8
```

Para el robot:

```bash
ros2 launch romeo_bringup romeo_bringup.launch.py chat_model:=romeo-empatico-v2-q8
```

### Cuantización sin compilador

`soup export --quant q8_0` falla porque falta el binario `llama-quantize`. No hace falta
compilarlo: el conversor de llama.cpp cuantiza desde Python en unos segundos.

```bash
python ~/.soup/llama.cpp/convert_hf_to_gguf.py ./merged_v2 \
  --outtype q8_0 --outfile romeo-empatico-v2.q8_0.gguf
```

Hacerlo **en la PC**, no en el Pi: compilar llama.cpp allá satura sus 4 núcleos.

### Velocidad

| Versión | Tamaño | Respuesta |
|---|---|---|
| f16 | 994 MB | 6-9 s |
| **q8_0** | **531 MB** | **3,4-4,2 s** |

Sin GPU, el Pi recorre el modelo entero por cada palabra generada, así que el peso del
archivo manda sobre la velocidad. q4_K_M (~350 MB) bajaría a ~2,5 s con pérdida de
calidad apreciable en un modelo tan chico.

## Archivos

| Archivo | Qué es |
|---|---|
| `soup_v2.yaml` | Config de entrenamiento con las 7 capas |
| `data/romeo_empatico_v2.jsonl` | Dataset, 1.825 pares |
| `Modelfile.v2.q8` | Modelfile de la versión cuantizada |
| `romeo-empatico-v2.q8_0.gguf` | Modelo desplegado, 531 MB |
| `output/romeo-empatico-v2/` | Adaptador LoRA |
| `merged_v2/` | Modelo fusionado en fp16 |

## Pendiente

- Limpiar el Pi: el clon de llama.cpp (~500 MB) y el GGUF f16 ya no hacen falta.
- Completar la traducción hasta 2.000 pares (faltan ~740) para reducir las frases rotas.
- Evaluar un modelo de 1B si se acepta subir a ~8 s por respuesta.
