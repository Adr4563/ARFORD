"""Cliente Serial hacia el carrito mecanum -- portado de
Clients/Carrito_Client.py del proyecto original, sin cambios de protocolo:
el firmware del ESP32-S3 (carrito-mecanum-esp32/2-l298n-mecanum/
mecanum_car_esp32s3.ino) NO SE TOCA, sigue esperando exactamente las mismas
líneas de texto (F/B/SL/SR/RL/RR/S) a 115200 baud por USB.

Encapsulado en una clase `CartSerial` (mismo motivo que Voz/Display en
communication_node/visualization_faces_node): moving_control_node crea el objeto en
on_configure() sin abrir el puerto todavía, y recién lo abre en
on_activate() (abrir el puerto resetea el ESP32 vía DTR, así que no
conviene hacerlo antes de que el nodo esté realmente activo).
"""
import threading
import time

BAUDRATE = 115200
TIMEOUT = 2.0  # cable directo: si no responde rápido, no está conectado/andando.

# Traduce el comando de MotionCommand.msg al que entiende el firmware --
# son casi el mismo alfabeto a propósito (MotionCommand ya se definió
# calcado del protocolo real, ver lora_interfaces/msg/MotionCommand.msg),
# esta tabla solo existe para el caso especial ROTATE360 (no es un comando
# atómico del firmware).
_COMANDOS_DIRECTOS = {"F", "B", "SL", "SR", "RL", "RR", "S"}


class CartSerial:
    def __init__(self, puerto="/dev/ttyACM0", logger=None):
        self.puerto = puerto
        self.logger = logger
        self._conexion = None

    def _log(self, msg, warn=True):
        if self.logger is not None:
            (self.logger.warn if warn else self.logger.info)(msg)
        else:
            print(msg)

    def _abrir_puerto(self):
        """Reconecta si hace falta (primera vez, o si la anterior murió por
        desconexión). None si no se pudo abrir -- nunca lanza."""
        import serial  # pyserial -- import perezoso, mismo motivo que el resto
        # de los clientes de hardware: moving_control_node tiene que poder arrancar
        # igual sin pyserial instalado (ej. para probar el resto del nodo).

        if self._conexion is not None and self._conexion.is_open:
            return self._conexion
        try:
            self._conexion = serial.Serial(self.puerto, BAUDRATE, timeout=TIMEOUT)
            # Abrir el puerto reinicia el ESP32 (toggle de DTR); setup()
            # tarda un instante en volver a dejarlo listo -- sin esta
            # espera, el primer comando de la sesión se puede perder en el
            # reinicio.
            time.sleep(2.0)
            return self._conexion
        except Exception as e:
            self._log(f"[carrito] no se pudo abrir {self.puerto}: {e}")
            self._conexion = None
            return None

    def _mandar(self, comando):
        conexion = self._abrir_puerto()
        if conexion is None:
            return False
        try:
            conexion.write(f"{comando}\n".encode())
            return True
        except Exception as e:
            self._log(f"[carrito] no se pudo mandar {comando!r}: {e}")
            self._conexion = None  # forzar reconexión en el próximo intento
            return False

    def ejecutar(self, comando):
        """Traduce y ejecuta un MotionCommand.command -- llamado desde el
        callback de suscripción de moving_control_node. Nunca lanza; solo loguea si
        el carrito no responde (cable desconectado, ESP32 apagado, etc.),
        mismo criterio de "falla gracioso" del proyecto original."""
        comando = (comando or "").strip().upper()
        if comando == "ROTATE360":
            # 'Girar 360°' no es un comando único en el firmware -- se
            # aproxima mandando 'RR' varias veces seguidas. Sin calibrar
            # contra el hardware real (grados por pulso desconocidos),
            # igual que en el proyecto original -- se lanza en un hilo
            # aparte para no bloquear el callback de suscripción del nodo.
            threading.Thread(target=self._rotar_360, daemon=True).start()
            return True
        if comando not in _COMANDOS_DIRECTOS:
            self._log(f"[carrito] comando desconocido: {comando!r}, se ignora")
            return False
        if not self._mandar(comando):
            self._log(f"[carrito] no se pudo mandar {comando!r}")
            return False
        return True

    def _rotar_360(self, repeticiones=6, pausa=0.4):
        for _ in range(repeticiones):
            if not self._mandar("RR"):
                self._log("[carrito] no se pudo mandar 'ROTATE360'")
                return False
            time.sleep(pausa)
        return True
