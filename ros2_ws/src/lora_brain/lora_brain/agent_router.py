"""Agente router: clasifica cada mensaje del usuario en TRIVIA / CHAT_LIBRE
-- SIN LLM. Portado tal cual de Agents/Agent_Router.py del proyecto
original (TF-IDF de n-gramas de caracteres + regresión logística,
scikit-learn, ~95-96% held-out) -- ningún cambio de lógica, solo de dónde
sale la ruta a router_modelo.joblib (ahora un asset instalado del paquete,
ubicado con ament_index_python en vez de una ruta relativa al archivo).

orchestrator_node llama a enrutar() directo, como función Python normal --
no es un servicio ROS2 (microsegundos, sin hardware, no vale la pena el
overhead de una llamada a servicio para esto, mismo razonamiento del plan
de migración sobre qué queda como librería vs qué se vuelve nodo)."""
import os

import joblib

from ament_index_python.packages import get_package_share_directory

from . import perf_monitor

_MODELO_PATH = os.path.join(
    get_package_share_directory('lora_brain'), 'data', 'router_modelo.joblib')

_modelo = joblib.load(_MODELO_PATH)
_vectorizador = _modelo["vectorizador"]
_clasificador = _modelo["clasificador"]

RUTAS_VALIDAS = {"TRIVIA", "CHAT_LIBRE"}


@perf_monitor.medir("agent_router")
def enrutar(mensaje_usuario):
    """Clasifica mensaje_usuario en una de RUTAS_VALIDAS -- nunca devuelve
    "" (es una cuenta local, siempre responde algo de RUTAS_VALIDAS)."""
    vector = _vectorizador.transform([mensaje_usuario])
    return _clasificador.predict(vector)[0]
