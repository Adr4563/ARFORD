"""visualization_faces_node -- nodo de VISUALIZACIÓN de Lora: agrupa
percepción (cámara -> emoción) y expresión (cara animada en pantalla)
porque las dos son "lo que se ve" del robot, tal como se acordó en el
diagrama de 4 nodos.

Junta lo que en el proyecto original eran 3 piezas:
- display.py + face_viewer.py             -> _face_display.py (+ face_viewer.py portado tal cual)
- Clients/Camara_Client.py + ai-camera/reconocer_emocion.py -> _emotion_detector.py

Es un LifecycleNode: on_configure crea los objetos `Display`/
`DetectorEmocion` sin tocar hardware; on_activate recién ahí puede mostrar
la primera cara ("content", cara de reposo) -- el detector de emoción no
necesita "activarse" (no mantiene un proceso vivo entre detecciones, cada
DetectEmotion.srv abre y cierra la cámara), así que no hace nada especial
en on_activate más que quedar disponible para el servicio.

Expone:
  - Suscripción /lora/face_command   (lora_interfaces/FaceCommand)
  - Servicio    /lora/detect_emotion (lora_interfaces/DetectEmotion)
"""
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn

from lora_interfaces.msg import FaceCommand
from lora_interfaces.srv import DetectEmotion

from ._emotion_detector import DetectorEmocion
from ._face_display import Display


class VisualizationFacesNode(LifecycleNode):

    def __init__(self):
        super().__init__('visualization_faces_node')
        self._activo = False
        self._display = None
        self._detector = None
        # DetectEmotion puede tardar varios segundos (ver el benchmark en
        # el .srv) -- va en su propio callback group para no bloquear la
        # suscripción de FaceCommand mientras tanto (la cara SÍ tiene que
        # poder seguir cambiando -- ej. a "speaking" -- durante una
        # detección en curso).
        self._grupo_camara = ReentrantCallbackGroup()

    def on_configure(self, state):
        self.get_logger().info('[visualization_faces_node] configurando...')
        self._display = Display(logger=self.get_logger())
        self._detector = DetectorEmocion(logger=self.get_logger())

        self.create_subscription(
            FaceCommand, '/lora/face_command', self._cb_face_command, 10)
        self.create_service(
            DetectEmotion, '/lora/detect_emotion', self._cb_detect_emotion,
            callback_group=self._grupo_camara)

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.get_logger().info('[visualization_faces_node] activando...')
        self._activo = True
        self._display.mostrar_cara('content')
        return super().on_activate(state)

    def on_deactivate(self, state):
        self.get_logger().info('[visualization_faces_node] desactivando...')
        self._activo = False
        # display.detener() -- ver la nota en Orchestrator_Management.py
        # original sobre el try/finally que SIEMPRE lo llama: acá el
        # equivalente es hacerlo en on_deactivate/on_shutdown, para no
        # dejar un mpv/Tk huérfano corriendo si el nodo se desactiva.
        self._display.detener()
        return super().on_deactivate(state)

    def on_cleanup(self, state):
        self._activo = False
        self._display = None
        self._detector = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        self._activo = False
        if self._display is not None:
            self._display.detener()
        return TransitionCallbackReturn.SUCCESS

    def _cb_face_command(self, msg):
        if not self._activo:
            return
        self._display.mostrar_cara(msg.face_name)

    def _cb_detect_emotion(self, request, response):
        if not self._activo:
            response.emotion, response.confidence, response.detected = '', 0.0, False
            return response
        emocion, confianza = self._detector.detectar_emocion()
        if emocion is None:
            response.emotion, response.confidence, response.detected = '', 0.0, False
        else:
            response.emotion, response.confidence, response.detected = emocion, confianza, True
        return response


def main(args=None):
    rclpy.init(args=args)
    node = VisualizationFacesNode()
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
