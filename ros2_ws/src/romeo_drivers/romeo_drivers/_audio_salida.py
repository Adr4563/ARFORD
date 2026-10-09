"""Opciones de mpv para sacar audio por un dispositivo concreto.

Lo usan _voice_backends.py (voz) y _music_player.py (música), que antes
llamaban a mpv sin elegir salida.

Por qué se fuerza el muestreo: la placa ESP32-S3 solo acepta 16 kHz mono.
Si mpv le manda 22 kHz (Piper) o 44.1 kHz (los mp3), el `plughw` de ALSA
convierte con un remuestreo lineal sin filtro y la voz suena metálica,
"robótica". Forzando el muestreo en mpv, la conversión la hace libswresample
con un filtro de alta calidad y ALSA ya no toca nada. Comparado de oído en la
placa: suena claramente mejor.
"""


def opciones_mpv(audio_salida=None, muestreo=0):
    """Devuelve las opciones de mpv para `audio_salida` (p. ej.
    "alsa/plughw:CARD=Board,DEV=0"). `muestreo` > 0 fuerza esa frecuencia
    en mono, remuestreada por mpv. Sin salida devuelve [] (salida por
    defecto, como antes)."""
    if not audio_salida:
        return []
    opciones = [f"--audio-device={audio_salida}"]
    if muestreo:
        opciones += [
            f"--audio-samplerate={muestreo}",
            "--audio-channels=mono",
            "--audio-format=s16",
            "--audio-swresample-o=filter_size=64,phase_shift=10,"
            "linear_interp=1,cutoff=0.97",
        ]
    return opciones
