import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'romeo_drivers'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Assets de hardware -- las caras animadas (mp4 para el backend
        # DRM, gif para el backend Tkinter de demo en PC) y la música de
        # la columna 'musical' del dataset de trivia. Se instalan junto al
        # paquete para que _face_display.py/_music_player.py los
        # encuentren por ruta relativa al propio módulo, mismo criterio
        # que FACES_DIR/MUSICA_DIR en el proyecto original.
        (os.path.join('share', package_name, 'faces'),
            glob(os.path.join('romeo_drivers', 'data', 'faces', '*'))),
        (os.path.join('share', package_name, 'musica'),
            glob(os.path.join('romeo_drivers', 'data', 'musica', '*'))),
        # Modelos ONNX de ai-camera/modelos/ (YuNet + FER+) -- ver
        # _emotion_detector.py. Se copian acá porque visualization_faces_node
        # ya no puede importar ai-camera/ por sys.path (era un truco del
        # proyecto monolítico para reusar código entre carpetas hermanas;
        # en ROS2 cada paquete es autocontenido).
        (os.path.join('share', package_name, 'modelos'),
            glob(os.path.join('romeo_drivers', 'data', 'modelos', '*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Romeo Robot Team',
    maintainer_email='20200812@aloe.ulima.edu.pe',
    description=(
        'Nodos de hardware de Romeo: comunicación verbal/no verbal, '
        'visualización de caras (cámara+display) y motores.'
    ),
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'communication_node = romeo_drivers.communication_node:main',
            'visualization_faces_node = romeo_drivers.visualization_faces_node:main',
            'moving_control_node = romeo_drivers.moving_control_node:main',
        ],
    },
)
