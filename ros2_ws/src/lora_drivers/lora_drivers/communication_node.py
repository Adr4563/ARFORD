"""communication_node -- nodo de COMUNICACIÓN VERBAL Y NO VERBAL de Lora.

Junta lo que en el proyecto original (deploy-raspberry-standalone/) eran 3
piezas separadas que confluían en el mismo proceso de
Orchestrator_Management.py:

- voz_server.py            -> _web_bridge.py  (página web + SSE, entrada de
                                                texto/voz transcrita)
- Clients/Voice_Output_Client.py -> _voice_backends.py (voz de salida:
                                                teléfono/piper/edge-tts)
- Clients/Musica_Client.py -> _music_player.py (reproducción de música)
- entrada por teclado (_hilo_stdin en Orchestrator_Management.py) -> acá mismo

Es un LifecycleNode: on_configure prepara los objetos (Voz, WebBridge) SIN
tocar hardware/red todavía; on_activate recién ahí arranca Flask, precarga
Piper si corresponde, y empieza a leer stdin -- mismo criterio de
"no tocar hardware hasta estar realmente activo" que se pide para los 3
drivers en el plan de migración.

Expone:
  - Publisher  /lora/user_input          (lora_interfaces/UserInput)
  - Servicio   /lora/speak                (lora_interfaces/Speak)
  - Servicio   /lora/play_music           (lora_interfaces/PlayMusic)
  - Servicio   /lora/is_voice_client_connected (lora_interfaces/IsVoiceClientConnected)
"""
import threading

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn

from lora_interfaces.msg import UserInput
from lora_interfaces.srv import IsVoiceClientConnected, PlayMusic, Speak

from . import _music_player
from ._voice_backends import Voz
from ._web_bridge import WebBridge


class CommunicationNode(LifecycleNode):

    def __init__(self):
        super().__init__('communication_node')

        self.declare_parameter('voz_motor', 'telefono')
        self.declare_parameter('voz_edge', 'es-AR-ElenaNeural')
        self.declare_parameter('voz_piper_modelo', '')
        self.declare_parameter('voz_port', 8081)

        self._activo = False
        self._voz = None
        self._web_bridge = None
        self._pub_user_input = None
        self._hilo_stdin_iniciado = False

    # ─── Ciclo de vida ────────────────────────────────────────────────

    def on_configure(self, state):
        self.get_logger().info('[communication_node] configurando...')

        motor = self.get_parameter('voz_motor').value
        voz_edge = self.get_parameter('voz_edge').value
        modelo_piper = self.get_parameter('voz_piper_modelo').value or None
        puerto = self.get_parameter('voz_port').value

        self._web_bridge = WebBridge(
            port=puerto, on_texto=self._on_texto_web, logger=self.get_logger())
        self._voz = Voz(motor=motor, voz_edge=voz_edge, modelo_piper=modelo_piper,
                         web_bridge=self._web_bridge, logger=self.get_logger())

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
        # publicar (ver _on_texto_stdin), mismo patrón que _on_texto_web.
        if not self._hilo_stdin_iniciado:
            threading.Thread(target=self._hilo_stdin, daemon=True).start()
            self._hilo_stdin_iniciado = True

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.get_logger().info('[communication_node] activando...')
        self._voz.cargar()
        self._web_bridge.iniciar()
        self._activo = True
        return super().on_activate(state)

    def on_deactivate(self, state):
        self.get_logger().info('[communication_node] desactivando...')
        self._activo = False
        return super().on_deactivate(state)

    def on_cleanup(self, state):
        self._activo = False
        self._voz = None
        self._web_bridge = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        self._activo = False
        return TransitionCallbackReturn.SUCCESS

    # ─── Entrada de texto (stdin + web) -> /lora/user_input ────────────

    def _publicar_entrada(self, texto, source):
        if not self._activo or not texto:
            return
        msg = UserInput()
        msg.text = texto
        msg.source = source
        self._pub_user_input.publish(msg)

    def _on_texto_web(self, texto):
        self._publicar_entrada(texto, 'web')

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
        response.success = self._voz.hablar(request.text)
        return response

    def _cb_play_music(self, request, response):
        if not self._activo:
            response.reproducido = False
            return response
        response.reproducido = _music_player.reproducir(
            request.filename, esperar=request.esperar, logger=self.get_logger())
        return response

    def _cb_is_connected(self, request, response):
        response.connected = bool(
            self._web_bridge and self._web_bridge.hay_cliente_conectado())
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
