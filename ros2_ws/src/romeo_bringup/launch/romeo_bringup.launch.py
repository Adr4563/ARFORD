import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.events import matches_action
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from launch_ros.parameter_descriptions import ParameterValue
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
        get_package_share_directory('romeo_bringup'), 'config', 'romeo_params.yaml')

    args = [
        DeclareLaunchArgument('mic_activo', default_value='true',
                               description='false desactiva el STT y deja solo el teclado'),
        DeclareLaunchArgument('carrito_port', default_value='/dev/ttyACM0'),
        DeclareLaunchArgument('chat_model', default_value='arford-empatico-full'),
        DeclareLaunchArgument('chat_server_host', default_value='http://localhost:11434'),
    ]

    communication_node = LifecycleNode(
        package='romeo_drivers', executable='communication_node',
        name='communication_node', namespace='', output='screen',
        parameters=[params_file,
                     # ParameterValue con value_type explícito: una
                     # LaunchConfiguration se resuelve SIEMPRE a string, y
                     # mic_activo se declara como bool en el nodo -- sin
                     # esto llega 'true' (texto) y rclpy rechaza el
                     # parámetro al arrancar.
                     {'mic_activo': ParameterValue(
                         LaunchConfiguration('mic_activo'), value_type=bool)}],
    )

    visualization_faces_node = LifecycleNode(
        package='romeo_drivers', executable='visualization_faces_node',
        name='visualization_faces_node', namespace='', output='screen',
        parameters=[params_file],
    )

    moving_control_node = LifecycleNode(
        package='romeo_drivers', executable='moving_control_node',
        name='moving_control_node', namespace='', output='screen',
        parameters=[params_file, {'carrito_port': LaunchConfiguration('carrito_port')}],
    )

    orchestrator_node = Node(
        package='romeo_brain', executable='orchestrator_node',
        name='orchestrator_node', namespace='', output='screen',
        parameters=[params_file],
        additional_env={
            'CHAT_MODEL': LaunchConfiguration('chat_model'),
            'CHAT_SERVER_HOST': LaunchConfiguration('chat_server_host'),
        },
    )

    acciones = list(args) + [communication_node, visualization_faces_node, moving_control_node, orchestrator_node]
    for nodo in (communication_node, visualization_faces_node, moving_control_node):
        acciones.extend(_auto_configure_and_activate(nodo))

    return LaunchDescription(acciones)
