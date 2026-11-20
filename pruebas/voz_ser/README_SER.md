# Comparación de modelos de emoción en la voz (SER)

Esta carpeta está fuera del repo; el repo ARFORD no se modifica. El método y
los resultados están en [INFORME_SER.md](INFORME_SER.md).

## Requisitos

Python 3.14, con `torch` (CPU), `transformers`, `funasr`, `soundfile`,
`sounddevice` y `psutil`. **No se usa GPU**: ROMEO no tiene.

## 1. Grabar (micrófono de la ESP32-S3 Audio Board)

```powershell
python grabar.py --disp 17 --emocion feliz --carpeta audios_r2/actuados
```

- El índice 17 es el micrófono de la placa por WDM-KS; los otros índices
  graban en silencio.
- Un pitido agudo indica que hay que hablar; dos graves, que terminó la
  toma.
- Los archivos se nombran `<voz>_<emocion>_<n>.wav`, y la clase esperada se
  saca de ese nombre.

## 2. Comparar

```powershell
# prueba rápida: 2 o 3 audios en una carpeta aparte
python comparar_ser_v2.py --repo C:\Users\I13306\Desktop\ARFORD --audios <carpeta_prueba> --repeticiones 1 --normalizar ninguna --salida prueba
# corrida completa (variantes sin normalizar y rms, 4 hilos, 3 repeticiones para la latencia)
python comparar_ser_v2.py --repo C:\Users\I13306\Desktop\ARFORD --audios audios_r2 --normalizar ninguna rms --salida resultados_ser_r2b
```

**Modelos** (`--modelos`):

| Nombre | Notas |
|---|---|
| `dusha` | modelo actual |
| `e2v_base` | campeón provisional |
| `umuteam` | |
| `wavlm` | |
| `pollito_v5` | |
| `e2v_large` | opcional |

`pollito_v7` está excluido porque su ficha no declara licencia.

**Salidas:**
- `<salida>.csv`: una fila por audio, modelo y variante.
- `<salida>.json`: métricas, matrices de confusión y McNemar contra el
  campeón y contra dusha.

Los audios (`audios*/`) y los resultados (`resultados_ser*`) son voces del
equipo y no se suben a ningún repo.
