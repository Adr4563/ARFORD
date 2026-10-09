"""communication_node -- nodo de COMUNICACIÓN VERBAL Y NO VERBAL de Romeo.

Junta lo que en el proyecto original (deploy-raspberry-standalone/) eran
piezas separadas que confluían en el mismo proceso de
Orchestrator_Management.py:

- Clients/Voice_Output_Client.py -> _voice_backends.py (voz de salida:
                                                Piper, local)
- Clients/Musica_Client.py -> _music_player.py (reproducción de música)
- entrada por teclado (_hilo_stdin en Orchestrator_Management.py) -> acá mismo

La página web (voz_server.py -> _web_bridge.py: Flask + SSE, entrada de
texto desde el celular y voz de salida por Web Speech API) se ELIMINÓ: ya
no se quiere depender de un teléfono conectado.

La voz de entrada la reemplaza el STT LOCAL (_stt_engine.py, sherpa-onnx)
leyendo el micrófono de la placa Waveshare ESP32-S3-AUDIO-Board, que la Pi
ve como una tarjeta de sonido USB normal (UAC) -- ver
firmware-audio-board/README.md. O sea: el audio entra por hardware USB
estándar y la transcripción ocurre en la Pi, sin teléfono y sin nube. Los
dos caminos de entrada (micrófono y teclado) confluyen en el mismo tópico
/romeo/user_input, igual que antes confluían los 3 en una queue.Queue().

    ATENCIÓN: al día de hoy el micrófono de la placa entrega SILENCIO
    DIGITAL EXACTO (ceros, no ruido bajo) por un problema de firmware aún
    abierto -- ver "Lo que sigue abierto" en firmware-audio-board/README.md.
    El camino de voz de acá queda completo y probado contra un WAV
    (scripts/bench_stt.py), pero no va a transcribir nada hasta que ese
    bug se arregle. Si no hay micrófono, no hay modelos o falta
    sherpa-onnx, el nodo arranca igual y solo queda el teclado.

Es un LifecycleNode: on_configure prepara el objeto Voz SIN tocar
hardware todavía; on_activate recién ahí precarga Piper, abre el micrófono
y habilita la publicación de entrada -- mismo criterio de "no tocar
hardware hasta estar realmente activo" que se pide para los 3 drivers en
el plan de migración.

Expone:
  - Publisher  /romeo/user_input   (romeo_interfaces/UserInput)
  - Servicio   /romeo/speak        (romeo_interfaces/Speak)
  - Servicio   /romeo/play_music   (romeo_interfaces/PlayMusic)
"""
import threading
from contextlib import contextmanager

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn

from romeo_interfaces.msg import UserInput
from romeo_interfaces.srv import PlayMusic, Speak

from . import _music_player
from ._stt_engine import ReconocedorVoz
from ._voice_backends import Voz


class CommunicationNode(LifecycleNode):

    def __init__(self):
        super().__init__('communication_node')

        self.declare_parameter('voz_piper_modelo', '')

        # Entrada de voz por el micrófono de la placa ESP32-S3 (tarjeta de
        # sonido USB). `mic_activo=False` apaga el STT y deja solo el
        # teclado -- útil mientras el micrófono de la placa siga dando
        # ceros (ver la nota del módulo), o para probar en una PC sin placa.
        self.declare_parameter('mic_activo', True)
        # Nombre o índice del dispositivo de captura, tal como lo ve
        # sounddevice. Vacío = el default del sistema. En la Pi la placa es
        # "Romeo Audio Board" (así la nombra CONFIG_UAC_TUSB_PRODUCT del
        # firmware, ver firmware-audio-board/README.md).
        self.declare_parameter('mic_dispositivo', '')
        self.declare_parameter('stt_tam', 'tiny')      # tiny | base | small
        self.declare_parameter('stt_hilos', 2)         # 2 es el óptimo medido

        self._activo = False
        self._voz = None
        self._pub_user_input = None
        self._hilo_stdin_iniciado = False
        self._stt = None
        self._hilo_mic = None

        # ANTI-REALIMENTACIÓN. El micrófono y el altavoz son el MISMO
        # dispositivo (la placa ESP32-S3), así que el micrófono capta la
        # propia voz de ARFORD y el STT la transcribe como si fuera el
        # usuario: ARFORD se contestaría a sí mismo en bucle. Mientras
        # suena algo por el altavoz (Speak o PlayMusic) se descarta lo que
        # llegue del micrófono.
        #
        # Es un CONTADOR, no un bool: Speak y PlayMusic comparten el grupo
        # MutuallyExclusive, pero PlayMusic con esperar=false vuelve
        # enseguida dejando a mpv sonando de fondo, así que puede haber
        # música y voz solapadas -- con un bool, la primera en terminar
        # reabriría el micrófono mientras la otra sigue sonando.
        self._reproduciendo = 0
        self._reproduciendo_lock = threading.Lock()

        # Speak y PlayMusic BLOQUEAN varios segundos (subprocess.run de mpv
        # con la síntesis completa, o la canción entera cuando
        # esperar=true). Van juntos en un grupo MutuallyExclusive para que
        # el executor los serialice: nunca se pisan voz y música en el
        # parlante, y a la vez el nodo nunca queda con un solo hilo
        # bloqueado sin poder atender el resto (ver main(): hace falta un
        # MultiThreadedExecutor para que esto signifique algo).
        self._grupo_audio = MutuallyExclusiveCallbackGroup()

    # ─── Ciclo de vida ────────────────────────────────────────────────

    def on_configure(self, state):
        self.get_logger().info('[communication_node] configurando...')

        modelo_piper = self.get_parameter('voz_piper_modelo').value or None
        self._voz = Voz(modelo_piper=modelo_piper, logger=self.get_logger())

        # El modelo NO se carga acá: ReconocedorVoz es perezoso a propósito
        # (carga en el primer uso), y on_configure no debe tocar hardware ni
        # gastar los ~2s de carga del modelo. Se carga al escuchar.
        if self.get_parameter('mic_activo').value:
            self._stt = ReconocedorVoz(
                tam=self.get_parameter('stt_tam').value,
                hilos=self.get_parameter('stt_hilos').value)

        # create_publisher() en un LifecycleNode ya devuelve un publisher
        # "gestionado" (no publica de verdad hasta que el nodo esté active,
        # ver rclpy.lifecycle.LifecycleNode) -- no existe un método aparte
        # create_lifecycle_publisher() en rclpy.
        self._pub_user_input = self.create_publisher(
            UserInput, '/romeo/user_input', 10)

        self.create_service(Speak, '/romeo/speak', self._cb_speak,
                             callback_group=self._grupo_audio)
        self.create_service(PlayMusic, '/romeo/play_music', self._cb_play_music,
                             callback_group=self._grupo_audio)

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
        self._arrancar_microfono()
        self.get_logger().info(
            '[communication_node] hablale al micrófono, o escribí acá y Enter')
        return super().on_activate(state)

    def on_deactivate(self, state):
        self.get_logger().info('[communication_node] desactivando...')
        # Poner _activo en False ya corta el bucle del micrófono: escuchar()
        # recibe `detener` y lo consulta en cada vuelta (ver
        # _arrancar_microfono). No se hace join() del hilo: puede estar
        # dentro de una transcripción de ~2.5s y no se quiere bloquear la
        # transición de lifecycle por eso -- el hilo sale solo y, como es
        # daemon, tampoco impide que el proceso termine.
        self._activo = False
        return super().on_deactivate(state)

    def on_cleanup(self, state):
        self._activo = False
        self._voz = None
        self._stt = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        self._activo = False
        return TransitionCallbackReturn.SUCCESS

    # ─── Entrada de voz (micrófono de la placa) -> /romeo/user_input ─────

    def _arrancar_microfono(self):
        """Lanza el hilo que escucha el micrófono, si el STT está habilitado.

        Idempotente: si el hilo ya está vivo (p.ej. deactivate -> activate
        rápido, antes de que el anterior saliera) no lanza un segundo, que
        competiría por el mismo dispositivo de captura."""
        if self._stt is None:
            self.get_logger().info(
                '[mic] mic_activo=false -- solo entrada por teclado')
            return
        if self._hilo_mic is not None and self._hilo_mic.is_alive():
            return
        self._hilo_mic = threading.Thread(target=self._escuchar_microfono, daemon=True)
        self._hilo_mic.start()

    def _escuchar_microfono(self):
        """Cuerpo del hilo del micrófono. `escuchar()` bloquea hasta que
        `detener` devuelve True, y nunca propaga excepciones (falla
        gracioso: loguea y vuelve), así que acá no hace falta try/except."""
        dispositivo = self.get_parameter('mic_dispositivo').value or None
        self.get_logger().info(
            f"[mic] escuchando (dispositivo={dispositivo or 'default del sistema'}, "
            f"modelo whisper-{self._stt.tam})")
        self._stt.escuchar(
            al_transcribir=self._on_texto_voz,
            detener=lambda: not self._activo,
            dispositivo=dispositivo,
        )
        self.get_logger().info('[mic] se dejó de escuchar')

    def _on_texto_voz(self, texto):
        # Si ARFORD está hablando o sonando música, esto es su propio audio
        # entrando por el micrófono de la misma placa -- se descarta (ver
        # la nota de anti-realimentación en __init__).
        if self._sonando():
            self.get_logger().debug(f"[mic] descartado (ARFORD suena): {texto!r}")
            return
        self.get_logger().info(f"[mic] transcrito: {texto!r}")
        self._publicar_entrada(texto.strip(), 'voice_board')

    # ─── Anti-realimentación micrófono/altavoz ─────────────────────────

    def _sonando(self):
        with self._reproduciendo_lock:
            return self._reproduciendo > 0

    @contextmanager
    def _mientras_suena(self):
        """Marca que el altavoz está ocupado, para que _on_texto_voz
        descarte lo que el micrófono capte mientras tanto."""
        with self._reproduciendo_lock:
            self._reproduciendo += 1
        try:
            yield
        finally:
            with self._reproduciendo_lock:
                self._reproduciendo -= 1

    # ─── Entrada de texto (stdin) -> /romeo/user_input ──────────────────

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

    # ─── Servicios ──────────────────────────────────────────────────

    def _cb_speak(self, request, response):
        if not self._activo:
            self.get_logger().warning('[communication_node] Speak llamado sin estar activo')
            response.success = False
            return response
        with self._mientras_suena():
            response.success = self._voz.hablar(request.text)
        return response

    def _cb_play_music(self, request, response):
        if not self._activo:
            response.reproducido = False
            return response
        # Solo se silencia el micrófono con esperar=true, donde este
        # callback dura lo que dura la canción. Con esperar=false mpv queda
        # sonando de fondo DESPUÉS de que este callback vuelve, así que el
        # contador no serviría de nada: habría que esperar al proceso para
        # saber cuándo bajar el gate, y justo lo que se quiere ahí es no
        # bloquear el turno. Es una limitación conocida -- la música de
        # fondo puede colarse en el micrófono.
        if request.esperar:
            with self._mientras_suena():
                response.reproducido = _music_player.reproducir(
                    request.filename, esperar=True, logger=self.get_logger())
        else:
            response.reproducido = _music_player.reproducir(
                request.filename, esperar=False, logger=self.get_logger())
        return response


def main(args=None):
    rclpy.init(args=args)
    node = CommunicationNode()
    # MultiThreadedExecutor (no SingleThreaded): Speak y PlayMusic bloquean
    # segundos y, con un solo hilo, el nodo no podría atender ninguna otra
    # callback mientras suena la voz -- las transiciones de lifecycle y el
    # segundo servicio quedaban encoladas detrás. Los dos servicios de
    # audio comparten un MutuallyExclusiveCallbackGroup, así que siguen
    # serializados entre sí (ver __init__).
    executor = MultiThreadedExecutor(num_threads=4)
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
