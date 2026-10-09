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

WATCHDOG (añadido acá, no estaba en el proyecto original): el firmware
mantiene el último comando indefinidamente -- un "F" deja el carrito
andando hacia adelante para siempre, hasta que llegue otro comando. Si
orchestrator_node muere, o se cae la red, o el nodo se desactiva justo
después de publicar un movimiento, nadie manda el "S" y el robot se va
contra la pared. Así que cada comando de movimiento arma un temporizador
que manda "S" solo si no llega nada nuevo antes de que venza. Es la única
parte de este archivo con riesgo FÍSICO, y el firmware no tiene su propio
watchdog, así que tiene que estar de este lado.
"""
import threading
import time

BAUDRATE = 115200
TIMEOUT = 2.0  # cable directo: si no responde rápido, no está conectado/andando.

# Segundos que el carrito puede seguir moviéndose sin recibir un comando
# nuevo antes de que el watchdog lo pare solo. Generoso a propósito: los
# desplazamientos de una pregunta de trivia son pulsos cortos, así que si
# pasan 3s sin nada es que ya nadie está al mando.
WATCHDOG_SEG = 3.0

# Comandos que DEJAN EL CARRITO EN MOVIMIENTO y por lo tanto arman el
# watchdog. "S" (stop) no está: es justo lo que el watchdog manda.
_COMANDOS_MOVIMIENTO = {"F", "B", "SL", "SR", "RL", "RR"}

# Traduce el comando de MotionCommand.msg al que entiende el firmware --
# son casi el mismo alfabeto a propósito (MotionCommand ya se definió
# calcado del protocolo real, ver romeo_interfaces/msg/MotionCommand.msg),
# esta tabla solo existe para el caso especial ROTATE360 (no es un comando
# atómico del firmware).
_COMANDOS_DIRECTOS = {"F", "B", "SL", "SR", "RL", "RR", "S"}


class CartSerial:
    def __init__(self, puerto="/dev/ttyACM0", logger=None,
                 watchdog_seg=WATCHDOG_SEG):
        self.puerto = puerto
        self.logger = logger
        self._conexion = None
        # watchdog_seg <= 0 lo desactiva (útil para probar el carrito a
        # mano sin que se pare solo cada 3s).
        self.watchdog_seg = watchdog_seg
        self._watchdog = None
        self._watchdog_lock = threading.Lock()

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

    # ─── Watchdog: parar solo si dejan de llegar comandos ──────────────

    def _cancelar_watchdog(self):
        with self._watchdog_lock:
            if self._watchdog is not None:
                self._watchdog.cancel()
                self._watchdog = None

    def _armar_watchdog(self):
        """(Re)arma el temporizador que manda 'S'. Cada comando nuevo
        cancela el anterior y arranca otro, así que el carrito solo se para
        cuando de verdad deja de recibir órdenes."""
        if self.watchdog_seg <= 0:
            return
        self._cancelar_watchdog()
        with self._watchdog_lock:
            self._watchdog = threading.Timer(self.watchdog_seg, self._parada_watchdog)
            self._watchdog.daemon = True
            self._watchdog.start()

    def _parada_watchdog(self):
        with self._watchdog_lock:
            self._watchdog = None
        self._log(f"[carrito] watchdog: {self.watchdog_seg}s sin comandos -- "
                   f"se manda 'S' para parar")
        self._mandar("S")

    def detener(self):
        """Para el carrito y cancela el watchdog. La llama
        moving_control_node en on_deactivate/on_shutdown: si el nodo se va
        mientras el carrito anda, hay que pararlo ANTES de soltar el
        puerto, no dejarlo a merced del watchdog."""
        self._cancelar_watchdog()
        self._mandar("S")

    # ─── API que usa el nodo ───────────────────────────────────────────

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
        # Un "S" explícito no necesita vigilancia: ya está parado.
        if comando in _COMANDOS_MOVIMIENTO:
            self._armar_watchdog()
        else:
            self._cancelar_watchdog()
        return True

    def _rotar_360(self, repeticiones=6, pausa=0.4):
        """Corre en su propio hilo (ver ejecutar()). Rearma el watchdog en
        cada pulso para que no salte a mitad del giro, y al terminar manda
        un "S": si no, el último "RR" dejaría el carrito girando hasta que
        venciera el watchdog."""
        for _ in range(repeticiones):
            if not self._mandar("RR"):
                self._log("[carrito] no se pudo mandar 'ROTATE360'")
                self._armar_watchdog()   # red de seguridad: puede haber quedado girando
                return False
            self._armar_watchdog()
            time.sleep(pausa)
        self._mandar("S")
        self._cancelar_watchdog()
        return True
