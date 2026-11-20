# Comparación de modelos de emoción en la voz (SER) para ROMEO

**Objetivo.** Elegir un reemplazo para el SER actual de ROMEO
(`xbgoose/hubert-large-speech-emotion-recognition-russian-dusha-finetuned`,
"dusha", HuBERT afinado con habla en ruso), comparando candidatos en las
mismas condiciones.

**Fecha:** 18-11-2026 · **Equipo de medición:** PC Windows 11, sin GPU, 8 GB de RAM, Python 3.14.3

## 1. Material

| Elemento | Detalle |
|---|---|
| Micrófono | Waveshare ESP32-S3-Audio-Board por USB ("Microphone (usb uac)", VID 303A) |
| Dispositivo usado | índice **17 (Windows WDM-KS)** de `sounddevice`. Los índices WASAPI, MME y DirectSound de la misma placa grabaron casi en silencio (−70 a −107 dBFS) y WDM-KS no (prueba en dos órdenes distintos). La causa probable es el mezclador de Windows, que WDM-KS no usa; no se verificó. |
| Formato | 16 kHz, mono, float32, 5 s por toma |
| Audios | 24 actuados: 2 hablantes del equipo × 4 emociones × 3 tomas |
| Frases | neutral «hoy es miércoles y son las dos»; feliz «estoy muy contento, qué buen día»; triste «me siento muy mal hoy»; enojado «estoy harto, esto me molesta mucho» |
| Nivel | hablante 1: −43 a −51 dBFS; hablante 2: −36 a −52 dBFS. Ninguna toma saturó. |

Son grabaciones técnicas del equipo para medir el sistema, no un estudio con
participantes. Los audios no se suben al repositorio.

**Procedimiento de grabación** (`grabar.py --disp 17 --emocion <e> --carpeta <c>`):
un pitido agudo indica "hablar" y dos graves, "fin de toma". Hubo que hacerlo
así porque el hablante no veía la cuenta regresiva en pantalla.

## 2. Scripts

| Script | Qué hace |
|---|---|
| `grabar.py` | Graba tomas por emoción y muestra el nivel (dBFS, pico, saturación) de cada una. |
| `comparar_ser.py` | Ronda 1. Un proceso por modelo; reduce cada salida a 4 clases con `agrupar` de `_ser_engine.py`. |
| `comparar_ser_v2.py` | Ronda 2. Corrige umuteam y wavlm, usa los mismos hilos para todos (4), agrega variantes de normalización y calcula UAR, F1, matriz de confusión, IC de Wilson y McNemar exacta. |

Comando de la ronda 2:
```
python comparar_ser_v2.py --repo <ARFORD> --audios audios_r2 --normalizar ninguna rms
```

## 3. Problemas encontrados y cómo se resolvieron

1. **umuteam respondía al azar** (ronda 1: ~35 % de confianza para todo y una predicción distinta en cada carga).
   El `config.json` declara `CustomAudioClassification` sin `auto_map`, así que
   `pipeline("audio-classification")` (lo que indica su ficha) carga la clase estándar y deja
   la cabeza con pesos aleatorios (transformers 5.23.0: `classifier.weight/bias MISSING`).
   Se reimplementó la cabeza de su toolkit (`NLP-UMUTeam/umuteam-speech-emotion`,
   `models/wav2vec2_bert_es.py`): projector → media temporal → dense → tanh → out_proj.
   Las etiquetas (`anger, disgust, fear, joy, neutral, sadness`) ya estaban en `_ETIQUETAS`.
2. **wavlm fallaba** con `ImportError: attempted relative import` en
   `_ser_wavlm.py:157` (`from ._ser_engine import agrupar`). En el robot funciona porque se importa
   como paquete. En v2 se importa como `romeo_drivers._ser_wavlm`. **No se cambió el repo.**
3. **Normalización de volumen.** No cambió ninguna predicción de dusha, e2v_base ni umuteam, porque
   esos modelos normalizan la entrada por dentro (dusha: `do_normalize=True`). Solo wavlm
   cambió, y empeoró. **La hipótesis de la ronda 1 ("con voz baja todo sale tristeza") queda descartada.**

## 4. Resultados (ronda 2, sin normalizar, 24 audios)

| Puesto | Modelo | Aciertos | IC 95 % | UAR | F1 macro | vs dusha (solo él / solo dusha) | p McNemar | RAM | Latencia (5 s, 4 hilos) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **e2v_base** (`emotion2vec/emotion2vec_plus_base`) | **13/24 (54 %)** | 35–72 % | 54 % | 48 % | 7 / 1 | 0,070 | 2,0 GB | 0,47 s |
| 2 | wavlm (`3loi/SER-Odyssey-Baseline-WavLM-Categorical`) | 11/24 (46 %) | 28–65 % | 46 % | 32 % | 7 / 3 | 0,344 | 1,7 GB | 1,11 s |
| 3 | umuteam (`UMUTeam/w2v-bert-emotion-es`) | 8/24 (33 %) | 18–53 % | 33 % | 30 % | 5 / 4 | 1,000 | 2,7 GB | 1,67 s |
| 4 | dusha (actual) | 7/24 (29 %) | 15–49 % | 29 % | 21 % | — | — | 1,7 GB | 1,05 s |

Recall por emoción:

| | Neutral | Tristeza | Enojo | Felicidad |
|---|---|---|---|---|
| e2v_base | 67 % | 100 % | 0 % | 50 % |
| wavlm | 83 % | 100 % | 0 % | 0 % |
| umuteam | 33 % | 83 % | 0 % | 17 % |
| dusha | 67 % | 0 % | 0 % | 50 % |

En la ronda 1 (mismos 24 audios, sin umuteam ni wavlm válidos), e2v_base obtuvo 13/24,
e2v_large 12/24 (3,1 GB de RAM) y dusha 7/24.

## 5. Conclusión

- **Recomendación: reemplazar dusha por e2v_base.** Quedó primero en las dos rondas, supera a dusha en
  todas las métricas, detecta la tristeza (dusha nunca la detectó) y es el más rápido.
- **La ventaja no es estadísticamente significativa todavía** (McNemar p = 0,070 con n = 24).
  Hacen falta unos 60 audios o más para confirmarla.
- **Ningún modelo reconoce el enojo** (0/6 en todos); suele confundirse con felicidad o tristeza.
- Empathic-Insight-Voice-Small (LAION) se evaluó en papel y se descartó como reemplazo directo:
  no tiene clase neutral y sus 55 cabezas de 295 MB no caben juntas en 8 GB.

## 6. Ronda 2b: nuevos competidores (19-11-2026)

### 6.1 Regla de decisión (escrita ANTES de correr la ronda 2b)

- **Campeón:** e2v_base. Se mantiene **salvo** que un competidor lo supere
  con intervalos de Wilson del 95 % que **no se solapen** y con McNemar
  exacta **p < 0,05** (pareada, sobre los mismos audios).
- **Si se solapan, es empate.** Se decide por latencia, después RAM, después
  licencia.
- Un modelo con licencia no clara **no puede ser campeón**.
- Con n < 60 el resultado es **orientativo**.
- La regla se aplica a la variante **sin normalizar**. La variante `rms` se
  reporta aparte.

### 6.2 Diagnóstico (evidencia reproducida el 19-11-2026, `transformers` 5.23.0)

- **umuteam [MEDIDO].** `config.json`:
  `architectures = ['CustomAudioClassification']`, `auto_map = None`.
  `id2label = {0: anger, 1: disgust, 2: fear, 3: joy, 4: neutral, 5: sadness}`,
  así que las etiquetas son reales, no `LABEL_n`, y `_ETIQUETAS` ya las
  conoce. Al cargarlo con `pipeline()`, el LOAD REPORT de transformers 5.x
  (que reemplaza al viejo aviso "Some weights … newly initialized") dice:
  - `classifier.weight` y `classifier.bias` en **MISSING**: "those params
    were newly initialized";
  - `classifier.dense.*`, `classifier.out_proj.*` y `classifier.layernorm.*`
    en **UNEXPECTED**.

  Es decir, la cabeza real no se carga y la que se usa es aleatoria.
  Corrección en `comparar_ser_v2.py` (`ClasificadorUMU`): se reimplementa la
  cabeza de su toolkit.
- **wavlm [MEDIDO].** Traceback al importar `_ser_wavlm` como módulo suelto:
  `_ser_wavlm.py`, línea 157, en `clasificar`: `from ._ser_engine import agrupar`
  → `ImportError: attempted relative import with no known parent package`.
  Corrección: importarlo como `romeo_drivers._ser_wavlm`. No se cambió el repo.

### 6.3 Modelos nuevos (fichas leídas en Hugging Face el 19-11-2026)

| Nombre | Ficha | Licencia | Datos de entrenamiento | Clases (config.json) | Métricas en la ficha | Decisión |
|---|---|---|---|---|---|---|
| pollito_v5 (`pollitoconpapass/superb-ser-finetuned-spanish-v5`) | wav2vec2-base afinado desde `superb/wav2vec2-base-superb-er`; 361 MB | **MIT** [MEDIDO, tag y README] | "acento mexicano y chileno"; **no nombra los datasets** [no verificado] | `LABEL_0`…`LABEL_5`. Según el README: miedo, triste, neutral, enojo, disgusto, feliz | ninguna | **entra** |
| pollito_v7 (`pollitoconpapass/ser-spanish-model-v7`) | wav2vec2-xls-r-300m afinado; 1,2 GB | **no declara licencia** [MEDIDO: sin tag ni campo en la ficha] | MESD (México) y "audio-emociones-chile" (Kaggle); la ficha no dice si es actuado o espontáneo | disgusto, enojo, feliz, miedo, neutral, triste | test (n = 136): exactitud 0,72, F1 macro 0,68; miedo con recall 0,14 | **excluido** (licencia no clara) |

**Etiquetas frente a `agrupar` [MEDIDO, en el código]:**
- pollito_v5 devuelve `LABEL_n`, que `_ETIQUETAS` no conoce. Aun traducidas
  con la tabla del README, quedan **`triste`, `feliz` y `disgusto`**, que
  tampoco están; `miedo`, `neutral` y `enojo` sí.
- En `comparar_ser_v2.py` se traducen a sinónimos conocidos (tristeza,
  felicidad, asco), sin tocar el repo.
- **Propuesta, sin aplicar:** agregar a `_ETIQUETAS` en `_ser_engine.py`
  `"triste": "tristeza", "feliz": "felicidad", "disgusto": "enojo"`. Para
  `LABEL_n` haría falta un mapeo específico de ese modelo.

**Sin equivalente directo entre las 4 clases:** miedo y disgusto. ROMEO ya
los envía a tristeza y a enojo, respectivamente; se mantiene ese criterio.

**La correspondencia LABEL_n → emoción de pollito_v5 viene solo del README;
el config no la trae** [HIPÓTESIS hasta comprobarla con resultados].

### 6.4 Audios

Hay 24 audios únicos de 2 voces (actuados). `conversacion/` está vacía.
**Faltan al menos 36 audios y 1 voz** para llegar a 60 audios con 3 voces.

### 6.5 Tabla B (costo)

Medida en esta PC (Windows 11, 8 GB, sin GPU, torch CPU). **Solo representa
a ROMEO si esta es la PC de inferencia del robot**, lo que no está
confirmado [pendiente].

### 6.6 Resultados de la ronda 2b [MEDIDO, `ronda2b.log`, `resultados_ser_r2b.*`]

24 audios (2 voces). **Resultado orientativo (n < 60).** Variante sin normalizar.

**Tabla A: exactitud**

| Modelo | Aciertos | IC95 Wilson | UAR | F1 macro | Recall neutral / tristeza / enojo / felicidad | vs e2v_base (solo él / solo e2v) | p McNemar |
|---|---|---|---|---|---|---|---|
| **e2v_base** (campeón) | **13/24 (54 %)** | 35–72 % | 54 % | 48 % | 67 / 100 / 0 / 50 | — | — |
| wavlm | 11/24 (46 %) | 28–65 % | 46 % | 32 % | 83 / 100 / 0 / 0 | 1 / 3 | 0,625 |
| umuteam | 8/24 (33 %) | 18–53 % | 33 % | 30 % | 33 / 83 / 0 / 17 | 1 / 6 | 0,125 |
| dusha (actual) | 7/24 (29 %) | 15–49 % | 29 % | 21 % | 67 / 0 / 0 / 50 | 1 / 7 | 0,070 |
| pollito_v5 | 6/24 (25 %) | 12–45 % | 25 % | 15 % | 0 / 83 / 17 / 0 | 1 / 8 | **0,039** |

Con normalización `rms`, e2v_base, dusha y umuteam no cambian; wavlm baja a
9/24 y pollito_v5 a 5/24 (McNemar contra e2v_base, p = 0,008). Las matrices
de confusión 4x4 y los F1 por clase están en `ronda2b.log`.

**pollito_v5 colapsa en una sola salida [MEDIDO].** Su salida cruda es
`LABEL_1` en 21 de 24 audios y `LABEL_3` en 3, sea cual sea la emoción
actuada (feliz 6/6 en `LABEL_1`). Por tanto, la traducción de etiquetas del
README no influye en el resultado: ninguna reasignación lo arreglaría. No
generaliza a nuestras voces ni a este micrófono [HIPÓTESIS sobre la causa:
datos de entrenamiento distintos, acentos mexicano y chileno, y el aumento de
datos por replicación que describe su ficha].

**Tabla B: costo** (4 hilos, CPU). **No representa a ROMEO mientras no se
confirme que esta PC es la de inferencia.**

| Modelo | Carga (s) | RAM tras cargar (MB) | RAM pico (MB) | Latencia mediana, audio de 5 s | RTF |
|---|---|---|---|---|---|
| pollito_v5 | 4,8 | 801 | 924 | 0,155 s | 0,03 |
| e2v_base | 6,9 | 2003 | 2003 | 0,187 s | 0,04 |
| dusha | 8,8 | 1701 | 1723 | 0,415 s | 0,08 |
| wavlm | 5,4 | 1715 | 1724 | 0,428 s | 0,09 |
| umuteam | 6,2 | 2738 | 2811 | 0,611 s | 0,12 |

### 6.7 Aplicación de la regla

- **Ningún competidor supera a e2v_base.** Todos tienen menos aciertos, y
  ninguno cumple "Wilson sin solaparse y p < 0,05". **e2v_base se mantiene
  como campeón** [MEDIDO, orientativo].
- pollito_v5 es el único **significativamente peor** que el campeón
  (p = 0,039).
- Los intervalos de wavlm, umuteam y dusha se solapan con el de e2v_base. En
  esos casos la regla manda decidir por costo, y e2v_base también gana ahí:
  es más rápido que todos menos pollito_v5. Solo umuteam usa más RAM.
- **El enojo sigue sin reconocerse.** Todos los modelos dan 0/6, salvo
  pollito_v5 con 1/6.
- **Para confirmar** hacen falta al menos 36 audios y una voz más; idealmente
  también audios de conversación, no solo actuados.

### 6.8 Ronda 3: 36 audios (19-11-2026) [MEDIDO, `ronda3.log`, `resultados_ser_r3.*`]

Se agregaron **12 audios nuevos de p1** (misma voz de antes, otra sesión), a
20–30 cm de la placa y con nivel de −38 a −45 dBFS. Siguen siendo **2 voces**
y n < 60, así que el resultado es **orientativo**. Además, los 12 nuevos son
de un hablante que ya estaba, por lo que la muestra efectiva es menor que 36.

| Modelo | Aciertos (36) | IC95 Wilson | UAR | F1 macro | Solo los 12 nuevos | Los 24 anteriores | vs e2v_base (McNemar) |
|---|---|---|---|---|---|---|---|
| wavlm | **17/36 (47 %)** | 32–63 % | 47 % | 33 % | 6/12 | 11/24 | 4 / 3, p = 1,000 |
| **e2v_base** (campeón) | 16/36 (44 %) | 30–60 % | 44 % | 38 % | **3/12** | 13/24 | — |
| dusha | 11/36 (31 %) | 18–47 % | 31 % | 21 % | 4/12 | 7/24 | 4 / 9, p = 0,267 |
| umuteam | 11/36 (31 %) | 18–47 % | 31 % | 24 % | 3/12 | 8/24 | 2 / 7, p = 0,180 |
| pollito_v5 | 9/36 (25 %) | 14–41 % | 25 % | 14 % | 3/12 | 6/24 | 2 / 9, p = 0,065 |

**Aplicación de la regla:**
- wavlm queda 1 acierto por encima de e2v_base, pero los intervalos se
  solapan casi por completo y McNemar da p = 1,0. Es **EMPATE**.
- El empate se decide primero por latencia, y e2v_base fue más rápido en las
  dos corridas (0,19 contra 0,43 s en la ronda 2b; 0,51 contra 1,13 s aquí).
- **e2v_base se mantiene** [orientativo]. wavlm usa menos RAM (1,7 contra
  2,0 GB), pero la RAM solo cuenta si la latencia empata.

**Observaciones:**
- **[MEDIDO]** Con los 12 audios nuevos, e2v_base bajó mucho: 3/12, contra
  13/24 en los anteriores. Los clasificó casi todos como tristeza o neutral;
  enojado, por ejemplo, fue tristeza 3 de 3.
  **[HIPÓTESIS]** La causa sería el cambio de condición: más cerca del
  micrófono (efecto de proximidad) u otra forma de actuar en esta sesión.
  Esto indica que la ventaja de e2v_base no es estable.
- **[MEDIDO]** El enojo sigue en 0 % en todos los modelos, salvo pollito_v5
  con 1/9.
- **[MEDIDO]** La normalización `rms` no mejora a ningún modelo.
- **Las latencias de esta corrida son unas 2,5 veces mayores** que en la
  ronda 2b para todos los modelos (por ejemplo, e2v_base 0,51 contra
  0,19 s), con el mismo PC y los mismos hilos. **[HIPÓTESIS]** Habría otra
  carga en el PC, como las ventanas de grabación. Como referencia de costo
  vale la tabla B de la ronda 2b; el orden entre modelos se mantiene.

**Siguiente paso:** grabar la **tercera voz**, idealmente 2 personas nuevas,
para llegar a 60 o más audios. Sin eso, e2v_base y wavlm no se pueden separar.

## 7. Archivos de resultados (en `Documents/ROMEO`, fuera del repo)

`resultados_ser_v2.csv` / `.json` (ronda 2), `ronda2.log` (con matrices de confusión),
`resultados_ser_persona1.csv`, `resultados_ser_persona2.csv` (ronda 1).
