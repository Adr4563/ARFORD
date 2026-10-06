"""Lanza el stack completo de Lora: los 3 LifecycleNode de hardware
(communication_node, visualization_faces_node, moving_control_node -- paquete
lora_drivers) y orchestrator_node (paquete lora_brain).

Cada LifecycleNode se auto-configura y auto-activa al arrancar -- patrón
estándar de launch_ros: se emite un evento ChangeState(CONFIGURE) para cada
uno, y un RegisterEventHandler(OnStateTransition) dispara ACTIVATE apenas
termina de configurarse. No hace falta un "lifecycle manager" externo para
este caso (siempre se quiere el mismo orden arrancar->configurar->activar,
sin coordinación entre nodos) -- ver el README de ros2_ws para cómo
inspeccionar el estado a mano con `ros2 lifecycle`.

Los 3 modelos de Ollama (CHAT_MODEL/TRIVIA_MODEL/SALIDA_TRIVIA_MODEL) siguen
siendo variables de ENTORNO, no parámetros ROS2 -- llama_client.py (en
lora_brain) las lee con os.environ.get() al importarse, igual que
Clients/Llama_Client.py en el proyecto original. Acá se las pasa al proceso
de orchestrator_node vía `additional_env`, configurables como argumentos de
este launch file sin tocar código (mismo espíritu que el README original:
"CHAT_MODEL se puede sobreescribir por variable de entorno sin tocar
código").
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.events import matches_action
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition


def _auto_configure_and_activate(lifecycle_node):
    """Devuelve las acciones que hacen que `lifecycle_node` pase solo por
    configure -> inactive -> activate -> active apenas el proceso arranca."""
    configurar = EmitEvent(event=ChangeState(
        lifecycle_node_matcher=matches_action(lifecycle_node),
        transition_id=Transition.TRANSITION_CONFIGURE,
    ))
    activar_al_configurar = RegisterEventHandler(OnStateTransition(
        target_lifecycle_node=lifecycle_node,
        start_state='configuring', goal_state='inactive',
        entities=[EmitEvent(event=ChangeState(
            lifecycle_node_matcher=matches_action(lifecycle_node),
            transition_id=Transition.TRANSITION_ACTIVATE,
        ))],
    ))
    return configurar, activar_al_configurar


def generate_launch_description():
    params_file = os.path.join(
        get_package_share_directory('lora_bringup'), 'config', 'lora_params.yaml')

    args = [
        DeclareLaunchArgument('voz_motor', default_value='telefono',
                               description='telefono | piper | edge -- ver communication_node'),
        DeclareLaunchArgument('carrito_port', default_value='/dev/ttyACM0'),
        DeclareLaunchArgument('chat_model', default_value='lora-chat-libre-v4'),
        DeclareLaunchArgument('trivia_model', default_value='lora-trivia'),
        DeclareLaunchArgument('salida_trivia_model', default_value='lora-salida-trivia-v2'),
        DeclareLaunchArgument('chat_server_host', default_value='http://localhost:11434'),
    ]

    communication_node = LifecycleNode(
        package='lora_drivers', executable='communication_node',
        name='communication_node', namespace='', output='screen',
        parameters=[params_file, {'voz_motor': LaunchConfiguration('voz_motor')}],
    )

    visualization_faces_node = LifecycleNode(
        package='lora_drivers', executable='visualization_faces_node',
        name='visualization_faces_node', namespace='', output='screen',
        parameters=[params_file],
    )

    moving_control_node = LifecycleNode(
        package='lora_drivers', executable='moving_control_node',
        name='moving_control_node', namespace='', output='screen',
        parameters=[params_file, {'carrito_port': LaunchConfiguration('carrito_port')}],
    )

    orchestrator_node = Node(
        package='lora_brain', executable='orchestrator_node',
        name='orchestrator_node', namespace='', output='screen',
        parameters=[params_file, {'voz_motor': LaunchConfiguration('voz_motor')}],
        additional_env={
            'CHAT_MODEL': LaunchConfiguration('chat_model'),
            'TRIVIA_MODEL': LaunchConfiguration('trivia_model'),
            'SALIDA_TRIVIA_MODEL': LaunchConfiguration('salida_trivia_model'),
            'CHAT_SERVER_HOST': LaunchConfiguration('chat_server_host'),
        },
    )

    acciones = list(args) + [communication_node, visualization_faces_node, moving_control_node, orchestrator_node]
    for nodo in (communication_node, visualization_faces_node, moving_control_node):
        acciones.extend(_auto_configure_and_activate(nodo))

    return LaunchDescription(acciones)
