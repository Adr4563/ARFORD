"""Medición de tiempo y recursos por componente -- portado tal cual de
perf_monitor.py del proyecto original. Sigue escribiendo a CSV en vez de
usar mecanismos nativos de ROS2 (ej. /diagnostics) a propósito: mismo
razonamiento que el original ("no hace falta un runtime aparte para
esto") aplica igual acá, y mantiene compatibilidad con perf_report.py si
se lo quiere seguir usando tal cual para post-mortem.

Único cambio: LOGS_DIR ya no es relativo al archivo del módulo (que ahora
vive dentro del paquete instalado, un lugar no siempre escribible) sino
bajo el directorio HOME del proceso -- ~/.lora/logs/ -- configurable con
la env var LORA_LOGS_DIR si hace falta otra ubicación (ej. una carpeta
compartida con el resto del stack en la Pi)."""

import csv
import functools
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime

LOGS_DIR = os.environ.get("LORA_LOGS_DIR", os.path.expanduser("~/.lora/logs"))
RUTA_TIEMPOS = os.path.join(LOGS_DIR, "tiempos.csv")
RUTA_RECURSOS = os.path.join(LOGS_DIR, "recursos.csv")

INTERVALO_MUESTREO_SEG = float(os.environ.get("PERF_MUESTREO_SEG", "5"))

_ENCABEZADOS_TIEMPOS = ["timestamp", "componente", "duracion_ms", "error"]
_ENCABEZADOS_RECURSOS = ["timestamp", "cpu_percent", "memoria_rss_mb", "hilos_activos"]

_lock = threading.Lock()


def _escribir_fila(ruta, encabezados, fila):
    os.makedirs(LOGS_DIR, exist_ok=True)
    with _lock:
        nuevo = not os.path.exists(ruta)
        with open(ruta, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if nuevo:
                writer.writerow(encabezados)
            writer.writerow(fila)


@contextmanager
def medir_bloque(componente):
    inicio = time.perf_counter()
    error = 0
    try:
        yield
    except Exception:
        error = 1
        raise
    finally:
        duracion_ms = (time.perf_counter() - inicio) * 1000
        _escribir_fila(
            RUTA_TIEMPOS, _ENCABEZADOS_TIEMPOS,
            [datetime.now().isoformat(timespec="seconds"), componente,
             f"{duracion_ms:.1f}", error],
        )


def medir(componente):
    def decorador(func):
        @functools.wraps(func)
        def envoltorio(*args, **kwargs):
            with medir_bloque(componente):
                return func(*args, **kwargs)
        return envoltorio
    return decorador


_muestreo_iniciado = False


def iniciar_muestreo_recursos(logger=None):
    """Arranca (una sola vez) el hilo que muestrea CPU/memoria del proceso.
    `logger`: get_logger() del nodo ROS2 -- se usa en vez de print() si se
    pasa, para que los avisos queden en el log estándar de ROS2."""
    global _muestreo_iniciado
    if _muestreo_iniciado:
        return
    _muestreo_iniciado = True

    def _log(msg):
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    try:
        import psutil
    except ImportError:
        _log("[perf_monitor] psutil no instalado -- sin muestreo de CPU/memoria.")
        return

    proceso = psutil.Process(os.getpid())
    proceso.cpu_percent()

    def _loop():
        while True:
            time.sleep(INTERVALO_MUESTREO_SEG)
            try:
                cpu = proceso.cpu_percent()
                mem_mb = proceso.memory_info().rss / (1024 * 1024)
                hilos = proceso.num_threads()
            except Exception:
                continue
            _escribir_fila(
                RUTA_RECURSOS, _ENCABEZADOS_RECURSOS,
                [datetime.now().isoformat(timespec="seconds"),
                 f"{cpu:.1f}", f"{mem_mb:.1f}", hilos],
            )

    threading.Thread(target=_loop, daemon=True).start()
    _log(f"[perf_monitor] muestreo de recursos activo cada {INTERVALO_MUESTREO_SEG:.0f}s -> {RUTA_RECURSOS}")
