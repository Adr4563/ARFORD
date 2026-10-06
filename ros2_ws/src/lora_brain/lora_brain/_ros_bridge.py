"""Puente entre orchestrator_node y los 3 nodos de hardware de
lora_drivers -- concentra acá todos los publishers/service-clients y el
patrón de "llamar un servicio de forma síncrona sin deadlockear" que
orchestrator_node necesita en cada turno (ver la nota larga en
_llamar_servicio_sync más abajo, es la pieza más delicada de toda la
migración).

Se lo pasa a Agents/agent_behavior.py como `bridge` para que ese módulo
pueda disparar música/motores sin importar rclpy ni saber nada de tópicos
-- mismo espíritu de agent_behavior.py: lógica de negocio separada de la
integración con hardware/framework.
"""
import time

from lora_interfaces.msg import FaceCommand, MotionCommand
from lora_interfaces.srv import DetectEmotion, IsVoiceClientConnected, PlayMusic, Speak


class RosBridge:
    def __init__(self, node):
        self.node = node

        self.face_pub = node.create_publisher(FaceCommand, '/lora/face_command', 10)
        self.motion_pub = node.create_publisher(MotionCommand, '/lora/motion_command', 10)

        self.speak_client = node.create_client(Speak, '/lora/speak')
        self.music_client = node.create_client(PlayMusic, '/lora/play_music')
        self.emotion_client = node.create_client(DetectEmotion, '/lora/detect_emotion')
        self.voz_conectada_client = node.create_client(
            IsVoiceClientConnected, '/lora/is_voice_client_connected')

    # ─── El patrón central: llamar un servicio sin bloquear el proceso ──

    def _llamar_servicio_sync(self, client, request, timeout_sec=30.0):
        """Llama a un servicio ROS2 y espera la respuesta, de forma
        SÍNCRONA desde dentro del callback de /lora/user_input.

        Por qué NO se usa `client.call(request)` (el método "fácil" de
        rclpy) ni `rclpy.spin_until_future_complete(self.node, future)`:
        los dos, llamados desde DENTRO de un callback que ya está siendo
        ejecutado por el propio executor del nodo, hacen DEADLOCK si el
        nodo corre con un SingleThreadedExecutor -- el único hilo que
        podría procesar la respuesta del servicio es el mismo hilo que
        está bloqueado esperándola.

        La solución (y la razón por la que orchestrator_node.py corre con
        un MultiThreadedExecutor y la suscripción a /lora/user_input va en
        un ReentrantCallbackGroup, ver main() al final de ese archivo):
        `client.call_async(request)` devuelve un future de inmediato, y
        acá se hace una espera activa corta (`time.sleep` en vez de volver
        a llamar a `spin`) sobre `future.done()`. Mientras este hilo
        espera, OTRO hilo del MultiThreadedExecutor sigue libre para
        procesar la respuesta entrante del servicio (que rclpy entrega
        como una callback más, agendada en el mismo executor) y marcar el
        future como terminado.

        Devuelve None si el servicio no está disponible o no contesta a
        tiempo -- el caller decide el fallback (ver
        orchestrator_node._quiere_salir_trivia(), mismo criterio de
        "degradar, nunca romper el turno" que clasificar_salida_trivia()
        ya usaba en el proyecto original para Ollama caído)."""
        if not client.service_is_ready():
            if not client.wait_for_service(timeout_sec=2.0):
                self.node.get_logger().warning(
                    f"[ros_bridge] servicio {client.srv_name} no disponible")
                return None

        future = client.call_async(request)
        inicio = time.monotonic()
        while not future.done():
            if time.monotonic() - inicio > timeout_sec:
                self.node.get_logger().warning(
                    f"[ros_bridge] timeout esperando {client.srv_name}")
                return None
            time.sleep(0.01)
        try:
            return future.result()
        except Exception as e:
            self.node.get_logger().warning(f"[ros_bridge] {client.srv_name} falló: {e}")
            return None

    # ─── API de alto nivel que usan orchestrator_node/agent_behavior ────

    def hablar(self, texto):
        """BLOQUEA hasta que communication_node termina de reproducir la
        voz -- es la pieza clave de la garantía de orden voz -> cara ->
        (música + motores), ver orchestrator_node._reaccionar_veredicto()."""
        if not texto or not texto.strip():
            return False
        req = Speak.Request()
        req.text = texto
        resp = self._llamar_servicio_sync(self.speak_client, req, timeout_sec=60.0)
        return bool(resp and resp.success)

    def mostrar_cara(self, nombre):
        """Fire-and-forget -- un publish no bloquea, igual que
        display.mostrar_cara() en el proyecto original (un request IPC
        corto a mpv)."""
        msg = FaceCommand()
        msg.face_name = nombre
        self.face_pub.publish(msg)

    def mover(self, comando):
        """Fire-and-forget -- igual que Carrito_Client.mover()/mover_360()
        original, que tampoco bloqueaban el turno de trivia."""
        msg = MotionCommand()
        msg.command = comando
        self.motion_pub.publish(msg)

    def reproducir_musica(self, filename, esperar=False):
        """esperar=True SÍ bloquea (Reconocimiento Musical); esperar=False
        vuelve apenas communication_node lanza el Popen -- ver
        PlayMusic.srv."""
        req = PlayMusic.Request()
        req.filename = filename
        req.esperar = esperar
        timeout = 40.0 if esperar else 5.0
        resp = self._llamar_servicio_sync(self.music_client, req, timeout_sec=timeout)
        return bool(resp and resp.reproducido)

    def detectar_emocion(self):
        """BLOQUEA varios segundos (ver el benchmark en DetectEmotion.srv).
        Devuelve (emotion, confidence, detected)."""
        req = DetectEmotion.Request()
        resp = self._llamar_servicio_sync(self.emotion_client, req, timeout_sec=20.0)
        if resp is None:
            return None, None, False
        return resp.emotion, resp.confidence, resp.detected

    def voz_cliente_conectado(self):
        req = IsVoiceClientConnected.Request()
        resp = self._llamar_servicio_sync(self.voz_conectada_client, req, timeout_sec=5.0)
        return bool(resp and resp.connected)
