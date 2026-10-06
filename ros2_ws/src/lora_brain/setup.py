import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'lora_brain'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # router_modelo.joblib (Agent_Router) y preguntas.jsonl (preguntas.py)
        # -- se ubican en runtime con ament_index_python, mismo criterio que
        # los assets de lora_drivers.
        (os.path.join('share', package_name, 'data'),
            glob(os.path.join('lora_brain', 'data', '*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Lora Robot Team',
    maintainer_email='20200812@aloe.ulima.edu.pe',
    description='El cerebro de Lora: máquina de estados de la conversación (Trivia/Chat libre).',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'orchestrator_node = lora_brain.orchestrator_node:main',
        ],
    },
)
