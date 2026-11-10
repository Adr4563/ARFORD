# Métricas de Romeo — sesión de trivia

Captura de rendimiento de una sesión real de trivia (8 nov 2026), medida con
`ROMEO_MEDIR=1`. El notebook lee los CSV crudos y genera las gráficas.

## Archivos

| Archivo | Qué es |
|---------|--------|
| `metricas_trivia_romeo.ipynb` | Notebook con las gráficas ya ejecutadas |
| `build_notebook.py` | Script que genera el notebook desde cero |
| `metricas_romeo.xlsx` | Las mismas métricas en Excel (una hoja por dato) |
| `turnos.csv` | Afectiva + LLM + tokens, una fila por turno (Pi) |
| `modulos_pc.csv` | STT / SER / TER por frase (PC) |
| `tts.csv` | Tiempo de síntesis de voz por frase (PC) |
| `sistema.csv` | CPU / temp / RAM de la Pi cada 3 s |
| `romeo_run.log` | Log de la sesión (preguntas, veredictos, ventanas FER) |

## Gráficas del notebook

1. **Rocha vs IA** — para cada pregunta, si la respuesta del LLM coincide con
   la respuesta por defecto de Rocha (Coincide / No coincide), por categoría.
2. **Latencia por módulo** — tiempo promedio de STT, TER, SER, capa afectiva
   (PULSE+ROSE+AEB), LLM y TTS.
3. **Latencia por nodo** — etapas agrupadas por nodo ROS2.
4. **Tokens** — tiempo del LLM vs tokens de salida (con regresión) y distribución.
5. **Recursos de la Pi** — CPU total, proceso de cámara, temperatura y RAM en el tiempo.

## Cómo regenerar

```bash
cd pruebas/metricas
jupyter notebook metricas_trivia_romeo.ipynb   # o abrir en VS Code
```

El notebook lee los CSV de esta misma carpeta. Para una sesión nueva, reemplaza
los CSV (de `~/romeo/mediciones/` en la Pi y `~/romeo_medicion_modulos_pc.csv`
en el PC) y vuelve a ejecutar.

## Dónde se capturan las métricas

Con `ROMEO_MEDIR=1`:
- **Pi** (lanzar_romeo.sh): `turnos_*.csv` (afectiva + LLM), `tts_*.csv`.
- **PC** (arrancar_servidores_pc.ps1 -Medir): `romeo_medicion_modulos_pc.csv`
  (STT/SER/TER).
- **Recursos**: `ros2_ws/src/romeo_drivers/scripts/recolector_sistema.sh`.

## Notas

- **Coincidencia Rocha vs IA: 27/30 = 90%** (una pregunta de ciencia y las dos
  de reconocimiento musical no coincidieron).
- **Temperatura**: en `sistema.csv` de esta corrida salió en 0 por un bug del
  recolector ya corregido (`thermal_zone0` en vez de `vcgencmd`). Lectura real
  ~73.5 °C, pico ~80 °C.
- **FER** corre en el sensor IMX500: no se cronometra como CPU, se reporta como
  la ventana de percepción (cuadros de la frase).
- **PULSE/ROSE/AEB** se miden como un bloque (la capa afectiva es aritmética, <1 ms).
