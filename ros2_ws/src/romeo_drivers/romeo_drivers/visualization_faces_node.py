"""visualization_faces_node -- nodo de VISUALIZACIÓN de Romeo: agrupa
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

FER CONTINUO (parámetro fer_continuo, por defecto true): al activarse deja
la cámara abierta clasificando de fondo a fer_fps cuadros por segundo. Es el
FER de la ventana de percepción afectiva de la tesis: orchestrator_node pide
con PerceiveFace el resumen de lo que se vio mientras el usuario hablaba.

Expone:
  - Suscripción /romeo/face_command    (romeo_interfaces/FaceCommand)
  - Servicio    /romeo/detect_emotion  (romeo_interfaces/DetectEmotion)
  - Servicio    /romeo/perceive_face   (romeo_interfaces/PerceiveFace)
"""
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn

from romeo_interfaces.msg import FaceCommand
from romeo_interfaces.srv import DetectEmotion, PerceiveFace

from ._emotion_detector import ETIQUETAS, DetectorEmocion
from ._face_display import Display


class VisualizationFacesNode(LifecycleNode):

    def __init__(self):
        super().__init__('visualization_faces_node')
        # FER de fondo para la ventana de percepción (ver docstring).
        self.declare_parameter('fer_continuo', True)
        self.declare_parameter('fer_fps', 2.0)
        # Margen alrededor de la frase: la cara suele anticipar o prolongar
        # lo que se dice, y el VAD recorta el silencio de los bordes.
        self.declare_parameter('fer_margen_seg', 0.5)
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
        self._detector = DetectorEmocion(
            logger=self.get_logger(), fps=self.get_parameter('fer_fps').value)

        self.create_subscription(
            FaceCommand, '/romeo/face_command', self._cb_face_command, 10)
        self.create_service(
            DetectEmotion, '/romeo/detect_emotion', self._cb_detect_emotion,
            callback_group=self._grupo_camara)
        self.create_service(
            PerceiveFace, '/romeo/perceive_face', self._cb_perceive_face,
            callback_group=self._grupo_camara)

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.get_logger().info('[visualization_faces_node] activando...')
        self._activo = True
        self._display.mostrar_cara('content')
        if self.get_parameter('fer_continuo').value:
            if not self._detector.iniciar_continuo():
                self.get_logger().warning(
                    '[cámara] sin FER continuo (faltan modelos o dependencias)')
        return super().on_activate(state)

    def on_deactivate(self, state):
        self.get_logger().info('[visualization_faces_node] desactivando...')
        self._activo = False
        self._detector.detener_continuo()
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
        if self._detector is not None:
            self._detector.detener_continuo()
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


    def _cb_perceive_face(self, request, response):
        """Resumen FER de la ventana [t_inicio, t_fin] (más el margen).
        Responde al instante: los cuadros ya están capturados."""
        response.etiquetas = list(ETIQUETAS)
        response.probabilidades = []
        response.cuadros = 0
        if not self._activo or self._detector is None:
            return response
        margen = float(self.get_parameter('fer_margen_seg').value)
        probs, n = self._detector.resumen(
            request.t_inicio - margen, request.t_fin + margen)
        if probs is not None:
            response.probabilidades = [float(p) for p in probs]
            response.cuadros = int(n)
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
