"""moving_control_node -- nodo de MOTORES de Lora: traduce comandos ROS2
(/lora/motion_command) al protocolo Serial que ya entiende el firmware del
ESP32-S3 (carrito-mecanum-esp32/2-l298n-mecanum/mecanum_car_esp32s3.ino),
SIN TOCAR el firmware -- ver _cart_serial.py.

LifecycleNode: on_configure crea el objeto CartSerial sin abrir el puerto
(abrir el puerto resetea el ESP32 por DTR); on_activate recién ahí lo abre
en el primer comando real (apertura perezosa, igual que
Clients/Carrito_Client.py original -- no hace falta abrir nada si nunca
llega un MotionCommand).

Expone:
  - Suscripción /lora/motion_command (lora_interfaces/MotionCommand)
"""
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn

from lora_interfaces.msg import MotionCommand

from ._cart_serial import CartSerial


class MovingControlNode(LifecycleNode):

    def __init__(self):
        super().__init__('moving_control_node')
        self.declare_parameter('carrito_port', '/dev/ttyACM0')
        self._activo = False
        self._cart = None

    def on_configure(self, state):
        self.get_logger().info('[moving_control_node] configurando...')
        puerto = self.get_parameter('carrito_port').value
        self._cart = CartSerial(puerto=puerto, logger=self.get_logger())
        self.create_subscription(
            MotionCommand, '/lora/motion_command', self._cb_motion_command, 10)
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.get_logger().info('[moving_control_node] activando...')
        self._activo = True
        return super().on_activate(state)

    def on_deactivate(self, state):
        self.get_logger().info('[moving_control_node] desactivando...')
        self._activo = False
        return super().on_deactivate(state)

    def on_cleanup(self, state):
        self._activo = False
        self._cart = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        self._activo = False
        return TransitionCallbackReturn.SUCCESS

    def _cb_motion_command(self, msg):
        if not self._activo:
            return
        self._cart.ejecutar(msg.command)


def main(args=None):
    rclpy.init(args=args)
    node = MovingControlNode()
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
