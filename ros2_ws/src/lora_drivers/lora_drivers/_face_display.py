"""Controla la carita mostrada en pantalla -- portado de display.py del
proyecto original, encapsulado en una clase `Display` (mismo motivo que
`Voz` en _voice_backends.py: visualization_faces_node necesita poder crear
el objeto en on_configure() sin tocar mpv/Tkinter todavía, y recién abrir
la ventana/proceso en on_activate()).

Misma lógica exacta que el original: dos backends elegidos automáticamente
según dónde se corre --

- DRM (Raspberry Pi headless, LCD/HDMI directo): mpv --vo=drm, un solo
  proceso `mpv --idle` vivo toda la sesión, cambio de cara por su socket
  IPC. Caras en .mp4 (hwdec por hardware, ver la nota completa en el
  README de este workspace / display.py original).
- Tkinter (demo en PC/Windows): face_viewer.py (portado tal cual, mismo
  archivo en este paquete), caras en .gif.

Los assets (faces/*.mp4 y *.gif) se instalan como data_files del paquete
(ver setup.py) y se ubican con ament_index_python, no con una ruta
relativa al archivo como en el original.
"""
import json
import os
import socket
import subprocess
import sys
import time

from ament_index_python.packages import get_package_share_directory

CARAS_VALIDAS = {"happy", "sad", "angry", "content", "speaking", "countdown"}
MPV_SOCKET = "/tmp/lora_face.sock"


class Display:
    def __init__(self, logger=None):
        self.logger = logger
        self.faces_dir = os.path.join(get_package_share_directory('lora_drivers'), 'faces')
        self.viewer_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'face_viewer.py')
        self.archivo_senal = '/tmp/lora_face_current.txt'

        # Headless en Linux (sin sesión gráfica) == pantalla conectada
        # directo por HDMI/LCD a la Pi, sin X11/Wayland de por medio -> DRM.
        # Con $DISPLAY seteado (o en Windows) hay un escritorio real -> Tkinter.
        self.usa_drm = sys.platform.startswith('linux') and not os.environ.get('DISPLAY')
        self._proceso = None

    def _log(self, msg):
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

    def _warn(self, msg):
        if self.logger is not None:
            self.logger.warning(msg)
        else:
            print(msg)

    # ─── Backend DRM (mpv, Raspberry Pi headless) ──────────────────────

    def _mpv_enviar(self, comando):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(MPV_SOCKET)
            s.sendall((json.dumps({"command": comando}) + "\n").encode())

    def _mpv_huerfano_vivo(self):
        """Ver la nota completa en display.py original -- DRM solo permite
        un master de pantalla a la vez; si un proceso anterior no cerró
        limpio, hay que REUSAR ese huérfano en vez de competir por la
        pantalla (bug real ya encontrado y resuelto en el proyecto
        original)."""
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                s.connect(MPV_SOCKET)
            return True
        except OSError:
            return False

    def _asegurar_visor_drm(self, ruta_inicial):
        if self._proceso is not None and self._proceso.poll() is None:
            return
        if self._mpv_huerfano_vivo():
            self._warn('[display] mpv huérfano de otra corrida sigue vivo -- lo reuso')
            return
        try:
            os.remove(MPV_SOCKET)
        except FileNotFoundError:
            pass
        self._proceso = subprocess.Popen(
            [
                "mpv", "--fs", "--vo=drm", "--idle=yes", "--loop-file=inf",
                "--no-osc", "--no-input-default-bindings", "--really-quiet",
                "--hwdec=auto",
                f"--input-ipc-server={MPV_SOCKET}", ruta_inicial,
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for _ in range(50):  # hasta 5s
            if os.path.exists(MPV_SOCKET):
                break
            time.sleep(0.1)

    def _mostrar_cara_drm(self, ruta):
        self._asegurar_visor_drm(ruta)
        self._mpv_enviar(["loadfile", ruta, "replace"])

    def _detener_drm(self):
        if self._proceso is not None and self._proceso.poll() is None:
            try:
                self._mpv_enviar(["quit"])
                self._proceso.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                self._proceso.terminate()
        self._proceso = None
        try:
            os.remove(MPV_SOCKET)
        except FileNotFoundError:
            pass

    # ─── Backend Tkinter (demo en PC) ──────────────────────────────────

    def _asegurar_visor_tk(self, ruta_inicial):
        if self._proceso is not None and self._proceso.poll() is None:
            return
        self._proceso = subprocess.Popen(
            [sys.executable, self.viewer_script, ruta_inicial, self.archivo_senal],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def _mostrar_cara_tk(self, ruta):
        self._asegurar_visor_tk(ruta)
        with open(self.archivo_senal, 'w', encoding='utf-8') as f:
            f.write(ruta)

    def _detener_tk(self):
        if self._proceso is not None and self._proceso.poll() is None:
            self._proceso.terminate()
            try:
                self._proceso.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        self._proceso = None
        try:
            os.remove(self.archivo_senal)
        except (FileNotFoundError, PermissionError):
            pass

    # ─── Interfaz pública ───────────────────────────────────────────

    def mostrar_cara(self, nombre):
        """Llamado desde el callback de suscripción a /lora/face_command."""
        if nombre not in CARAS_VALIDAS:
            self._warn(f"[display] Cara desconocida: {nombre!r}")
            return
        extension = "mp4" if self.usa_drm else "gif"
        ruta = os.path.join(self.faces_dir, f"{nombre}.{extension}")
        if not os.path.isfile(ruta):
            self._warn(f"[display] No existe {ruta}")
            return
        if self.usa_drm:
            self._mostrar_cara_drm(ruta)
        else:
            self._mostrar_cara_tk(ruta)

    def detener(self):
        """Llamar desde on_deactivate()/on_shutdown() del LifecycleNode --
        equivalente al try/finally de main() en Orchestrator_Management.py
        original que garantizaba display.detener() incluso ante excepción."""
        if self.usa_drm:
            self._detener_drm()
        else:
            self._detener_tk()
