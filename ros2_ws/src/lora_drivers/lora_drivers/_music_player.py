"""Reproduce archivos de musica/ con mpv -- portado tal cual de
Clients/Musica_Client.py del proyecto original (deploy-raspberry-standalone/),
sin cambios de lógica. Lo único que cambia es DE DÓNDE saca la carpeta
musica/: antes era una ruta relativa al propio archivo
(os.path.dirname(os.path.dirname(__file__))); acá los .mp3 se instalan como
`data_files` del paquete ament_python (ver setup.py) y se ubican con
ament_index_python, la forma estándar de ROS2 de encontrar assets
instalados de un paquete sin asumir rutas relativas frágiles.

Usado por communication_node.py para atender el servicio PlayMusic.
"""
import os
import subprocess

from ament_index_python.packages import get_package_share_directory

MUSICA_DIR = os.path.join(get_package_share_directory('lora_drivers'), 'musica')
REPRODUCCION_MAX_SEG = 20

# Margen sobre REPRODUCCION_MAX_SEG para el modo bloqueante: mpv tarda un
# poco en arrancar y en cerrar, y si el timeout fuera exacto cortaria el
# final de la cancion. Solo aplica como red de seguridad -- si mpv se
# colgara, el turno no se queda esperando para siempre.
_TIMEOUT_ESPERA_SEG = REPRODUCCION_MAX_SEG + 10


def reproducir(nombre_archivo, esperar=False, logger=None):
    """Reproduce un archivo de musica/ recortado a REPRODUCCION_MAX_SEG.

    `esperar=False` (default): lanza mpv y vuelve enseguida -- la música
    suena de fondo mientras el turno de conversación sigue (equivalente a
    PlayMusic.srv con esperar=false).

    `esperar=True`: BLOQUEA hasta que termina de sonar -- lo usan las
    preguntas de Reconocimiento Musical, donde el usuario tiene que
    escuchar la canción ANTES de poder responder (equivalente a
    PlayMusic.srv con esperar=true).

    Devuelve True si se llegó a reproducir, False si no había archivo o no
    está mpv. `logger`: el logger del nodo ROS2 (node.get_logger()) -- se
    usa en vez de print() para que quede en el log estándar de ROS2, mismo
    criterio del resto de los módulos portados."""
    def _log(msg):
        if logger is not None:
            logger.warning(msg)
        else:
            print(msg)

    ruta = os.path.join(MUSICA_DIR, nombre_archivo)
    if not os.path.isfile(ruta):
        _log(f"[música] no existe {ruta}, no se reproduce")
        return False
    cmd = ["mpv", "--no-video", f"--length={REPRODUCCION_MAX_SEG}",
           "--really-quiet", ruta]
    try:
        if not esperar:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        timeout=_TIMEOUT_ESPERA_SEG)
        return True
    except FileNotFoundError:
        _log("[música] mpv no está instalado, no se puede reproducir")
        return False
    except subprocess.TimeoutExpired:
        # mpv colgado: se reporta y se sigue. Nunca dejar el turno trabado
        # por un problema de audio -- mismo criterio que el resto de los
        # clientes de hardware del proyecto original.
        _log(f"[música] mpv no terminó en {_TIMEOUT_ESPERA_SEG}s, se sigue igual")
        return True
