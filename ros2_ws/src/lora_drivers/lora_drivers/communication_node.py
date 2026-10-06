"""communication_node -- nodo de COMUNICACIÓN VERBAL Y NO VERBAL de Lora.

Junta lo que en el proyecto original (deploy-raspberry-standalone/) eran
piezas separadas que confluían en el mismo proceso de
Orchestrator_Management.py:

- Clients/Voice_Output_Client.py -> _voice_backends.py (voz de salida:
                                                piper/edge-tts)
- Clients/Musica_Client.py -> _music_player.py (reproducción de música)
- entrada por teclado (_hilo_stdin en Orchestrator_Management.py) -> acá mismo
- entrada por voz: micrófono de la placa ESP32-S3 (tarjeta de sonido USB)
  -> _stt_engine.py (whisper-tiny + VAD, con normalización de volumen)

La voz y la música salen por el parlante de la MISMA placa. Para que Lora no
se escuche a sí misma, el micrófono se pausa mientras habla o suena música
(ver _mic_pausado).

Es un LifecycleNode: on_configure prepara los objetos (Voz) SIN tocar
hardware todavía; on_activate recién ahí precarga Piper y abre el micrófono
-- mismo criterio de "no tocar hardware hasta estar realmente activo" que se
pide para los 3 drivers en el plan de migración.

Expone:
  - Publisher  /lora/user_input          (lora_interfaces/UserInput)
  - Servicio   /lora/speak                (lora_interfaces/Speak)
  - Servicio   /lora/play_music           (lora_interfaces/PlayMusic)
  - Servicio   /lora/is_voice_client_connected (lora_interfaces/IsVoiceClientConnected)
    -- siempre responde False: era la página web del teléfono, que ya no
    existe. Se mantiene porque el orquestador lo sigue declarando.
"""
import threading
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn

from lora_interfaces.msg import UserInput
from lora_interfaces.srv import IsVoiceClientConnected, PlayMusic, Speak

from . import _music_player
from ._stt_engine import reconocedor
from ._voice_backends import Voz

# Frases que whisper "inventa" con ruido o silencio (no las dijo nadie).
_ALUCINACIONES = {
    "thank you", "thanks for watching", "gracias", "gracias por ver",
    "gracias por ver el video", "subtítulos realizados por la comunidad de amara.org",
    "suscríbete", "música", "[música]", "[music]", "you",
}


def _es_alucinacion(texto):
    """True si la transcripción parece ruido y no una frase real."""
    limpio = texto.lower().strip(" .,¡!¿?")
    if len(limpio) < 2 or limpio in _ALUCINACIONES:
        return True
    palabras = [p.strip(".,¡!¿?") for p in limpio.split()]
    # "so, so, so, so" / "hola hola hola hola": la misma palabra 3+ veces seguidas
    for i in range(len(palabras) - 2):
        if palabras[i] and palabras[i] == palabras[i + 1] == palabras[i + 2]:
            return True
    return False


# Margen tras terminar de hablar antes de volver a escuchar: el parlante y el
# buffer USB de la placa todavía están sacando la cola del audio.
_MARGEN_ECO_SEG = 0.6


class CommunicationNode(LifecycleNode):

    def __init__(self):
        super().__init__('communication_node')

        self.declare_parameter('voz_motor', 'piper')
        self.declare_parameter('voz_edge', 'es-AR-ElenaNeural')
        self.declare_parameter('voz_piper_modelo', '')
        # Motor "vits" (sherpa-onnx): carpeta del modelo y ajustes de la voz.
        self.declare_parameter('voz_vits_dir', '')
        self.declare_parameter('voz_vits_ruido', 0.667)
        self.declare_parameter('voz_vits_ruido_w', 0.8)
        self.declare_parameter('voz_vits_duracion', 1.0)
        self.declare_parameter('voz_tono_semitonos', 0.0)
        # Nombre ALSA del micrófono; vacío = no se escucha por micrófono.
        self.declare_parameter('mic_dispositivo', '')
        # Modelo de whisper para el STT: tiny (rápido) | base (más preciso,
        # ~2 s más por frase). Vacío = el default de _stt_engine.py.
        self.declare_parameter('stt_modelo', '')
        # Dispositivo de mpv para voz y música; vacío = salida por defecto.
        self.declare_parameter('audio_salida', '')
        # Volumen de la música (0-100). La placa se desconectaba con la música
        # fuerte: el parlante pedía demasiada corriente.
        self.declare_parameter('musica_volumen', 100)
        # Volumen de la voz (0-100). 100 = sin atenuar (el .wav del TTS ya viene
        # normalizado). Se aplica como --volume de mpv al reproducir la voz.
        self.declare_parameter('voz_volumen', 100)
        # Frecuencia que acepta audio_salida; > 0 hace que mpv remuestree
        # con calidad (ver _audio_salida.py). 0 = no forzar.
        self.declare_parameter('audio_muestreo', 0)

        self._activo = False
        self._voz = None
        self._audio_salida = None
        self._audio_muestreo = 0
        self._pub_user_input = None
        self._hilo_stdin_iniciado = False
        self._hilo_mic = None
        self._reproduciendo = 0          # voces/músicas bloqueantes en curso
        self._silencio_hasta = 0.0       # monotonic: no escuchar antes de esto
        self._lock_audio = threading.Lock()

    # ─── Ciclo de vida ────────────────────────────────────────────────

    def on_configure(self, state):
        self.get_logger().info('[communication_node] configurando...')

        motor = self.get_parameter('voz_motor').value
        voz_edge = self.get_parameter('voz_edge').value
        modelo_piper = self.get_parameter('voz_piper_modelo').value or None
        self._audio_salida = self.get_parameter('audio_salida').value or None
        self._audio_muestreo = self.get_parameter('audio_muestreo').value

        self._voz = Voz(motor=motor, voz_edge=voz_edge, modelo_piper=modelo_piper,
                         logger=self.get_logger(), audio_salida=self._audio_salida,
                         audio_muestreo=self._audio_muestreo,
                         volumen=self.get_parameter('voz_volumen').value,
                         vits={
                             'dir': self.get_parameter('voz_vits_dir').value,
                             'ruido': self.get_parameter('voz_vits_ruido').value,
                             'ruido_w': self.get_parameter('voz_vits_ruido_w').value,
                             'duracion': self.get_parameter('voz_vits_duracion').value,
                             'tono': self.get_parameter('voz_tono_semitonos').value,
                         })

        # create_publisher() en un LifecycleNode ya devuelve un publisher
        # "gestionado" (no publica de verdad hasta que el nodo esté active,
        # ver rclpy.lifecycle.LifecycleNode) -- no existe un método aparte
        # create_lifecycle_publisher() en rclpy.
        self._pub_user_input = self.create_publisher(
            UserInput, '/lora/user_input', 10)

        self.create_service(Speak, '/lora/speak', self._cb_speak)
        self.create_service(PlayMusic, '/lora/play_music', self._cb_play_music)
        self.create_service(IsVoiceClientConnected,
                             '/lora/is_voice_client_connected', self._cb_is_connected)

        # El hilo de stdin arranca una sola vez acá (no en on_activate): un
        # input() bloqueado no se puede "pausar" limpiamente si el nodo pasa
        # a inactive y vuelve a activate -- en vez de relanzar el hilo, se
        # deja corriendo siempre y se filtra por self._activo antes de
        # publicar (ver _publicar_entrada).
        if not self._hilo_stdin_iniciado:
            threading.Thread(target=self._hilo_stdin, daemon=True).start()
            self._hilo_stdin_iniciado = True

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.get_logger().info('[communication_node] activando...')
        self._voz.cargar()
        self._activo = True
        self._iniciar_micro()
        return super().on_activate(state)

    def on_deactivate(self, state):
        self.get_logger().info('[communication_node] desactivando...')
        self._activo = False  # el hilo del micrófono lo ve y termina
        return super().on_deactivate(state)

    def on_cleanup(self, state):
        self._activo = False
        self._voz = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        self._activo = False
        return TransitionCallbackReturn.SUCCESS

    # ─── Entrada (stdin + micrófono) -> /lora/user_input ───────────────

    def _publicar_entrada(self, texto, source):
        if not self._activo or not texto:
            return
        msg = UserInput()
        msg.text = texto
        msg.source = source
        self._pub_user_input.publish(msg)

    def _hilo_stdin(self):
        while True:
            try:
                linea = input()
            except EOFError:
                break
            self._publicar_entrada(linea.strip(), 'stdin')

    def _iniciar_micro(self):
        dispositivo = self.get_parameter('mic_dispositivo').value
        if not dispositivo:
            self.get_logger().info('[micro] mic_dispositivo vacío -- sin entrada por voz')
            return
        if self._hilo_mic is not None and self._hilo_mic.is_alive():
            return
        modelo = self.get_parameter('stt_modelo').value
        if modelo and not reconocedor.listo:
            reconocedor.tam = modelo  # el modelo se carga recién en escuchar()
        self._hilo_mic = threading.Thread(
            target=self._hilo_micro, args=(dispositivo,), daemon=True)
        self._hilo_mic.start()

    def _hilo_micro(self, dispositivo):
        # escuchar() bloquea hasta que el nodo deja de estar activo o el
        # micrófono se cierra; carga el modelo en este hilo (tarda unos
        # segundos) para no frenar la activación del nodo.
        # Si la placa se desconecta, el micrófono se cierra: se reintenta
        # cada 3 s hasta que vuelva, sin que haga falta reiniciar el nodo.
        while self._activo:
            reconocedor.escuchar(
                self._on_texto_micro,
                detener=lambda: not self._activo,
                dispositivo=dispositivo,
                pausado=self._mic_pausado,
            )
            if not self._activo:
                break
            self.get_logger().warning('[micro] se cerró el micrófono -- reintentando en 3 s')
            time.sleep(3)

    def _on_texto_micro(self, texto):
        if _es_alucinacion(texto):
            self.get_logger().info(f'[micro] descartado (ruido): {texto!r}')
            return
        self.get_logger().info(f'[micro] oído: {texto!r}')
        self._publicar_entrada(texto, 'voice_board')

    def _mic_pausado(self):
        with self._lock_audio:
            return self._reproduciendo > 0 or time.monotonic() < self._silencio_hasta

    def _audio_empieza(self):
        with self._lock_audio:
            self._reproduciendo += 1

    def _audio_termina(self, margen=_MARGEN_ECO_SEG):
        with self._lock_audio:
            self._reproduciendo -= 1
            self._silencio_hasta = max(self._silencio_hasta, time.monotonic() + margen)

    # ─── Servicios ──────────────────────────────────────────────────

    def _cb_speak(self, request, response):
        if not self._activo:
            self.get_logger().warning('[communication_node] Speak llamado sin estar activo')
            response.success = False
            return response
        self._audio_empieza()
        try:
            response.success = self._voz.hablar(request.text)
        finally:
            self._audio_termina()
        return response

    def _cb_play_music(self, request, response):
        if not self._activo:
            response.reproducido = False
            return response
        if request.esperar:
            self._audio_empieza()
            try:
                response.reproducido = _music_player.reproducir(
                    request.filename, esperar=True, logger=self.get_logger(),
                    audio_salida=self._audio_salida, audio_muestreo=self._audio_muestreo,
                    volumen=self.get_parameter('musica_volumen').value)
            finally:
                self._audio_termina()
        else:
            # Suena de fondo y no se sabe cuándo termina: se deja de escuchar
            # durante lo máximo que puede durar (REPRODUCCION_MAX_SEG).
            response.reproducido = _music_player.reproducir(
                request.filename, esperar=False, logger=self.get_logger(),
                audio_salida=self._audio_salida, audio_muestreo=self._audio_muestreo,
                volumen=self.get_parameter('musica_volumen').value)
            if response.reproducido:
                self._audio_empieza()
                self._audio_termina(margen=_music_player.REPRODUCCION_MAX_SEG)
        return response

    def _cb_is_connected(self, request, response):
        response.connected = False
        return response


def main(args=None):
    rclpy.init(args=args)
    node = CommunicationNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
