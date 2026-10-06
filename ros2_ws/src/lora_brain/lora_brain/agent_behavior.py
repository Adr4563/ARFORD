"""Agente de comportamiento: decide qué cara mostrar y qué acciones de
música/motores disparar -- sin LLM, 100% determinista. Portado de
Agents/Agent_Behavior.py del proyecto original, con UN cambio de diseño
real (el resto es igual): antes llamaba directo a Clients.Carrito_Client/
Clients.Musica_Client (mismo proceso, imports); ahora recibe un `bridge`
-- un objeto que orchestrator_node arma con su publisher de
/lora/motion_command y su cliente de servicio /lora/play_music -- y le pide
a ÉL que dispare las acciones, en vez de llamar hardware directo. Este
archivo ya no sabe si el motor/música respondieron o no: esa parte
"gracioso ante falla" ahora vive en moving_control_node/communication_node (ver
esos archivos en lora_drivers).

`elegir_cara_pregunta()`/`cara_para_emocion()` NO necesitan `bridge`: son
lookups puros contra las columnas del dataset, sin tocar hardware -- igual
que en el original.
"""

# El dataset usa nombres en español (columna 'cara'); FaceCommand.face_name
# espera las claves en inglés de faces/*.mp4|*.gif. "Neutral" mapea a
# "content" porque es la misma cara de reposo -- no hay un asset de
# "neutral" aparte, igual que en el original.
_CARA_A_DISPLAY = {
    "Feliz": "happy",
    "Triste": "sad",
    "Enojado": "angry",
    "Neutral": "content",
}

# Traduce el texto en español de la columna 'desplazamiento' del dataset al
# comando que entiende MotionCommand.msg (que a su vez es casi 1:1 con el
# protocolo del firmware -- ver lora_interfaces/msg/MotionCommand.msg).
# Esta tabla vivía en Clients/Carrito_Client.py en el proyecto original;
# se mueve acá porque es lógica de NEGOCIO (qué significa "Izquierda" en
# el dataset), no de protocolo de hardware -- moving_control_node ya no sabe nada
# de español, solo de F/B/SL/SR/RL/RR/S/ROTATE360.
_DIRECCION_A_COMANDO = {
    "adelante": "F",
    "atrás": "B",
    "atras": "B",
    "izquierda": "SL",
    "derecha": "SR",
}


def elegir_cara_pregunta(pregunta, acerto):
    """Cara para el veredicto de una pregunta CON respuesta_esperada."""
    campo = "cara_respuesta_buena" if acerto else "cara_respuesta_mala"
    valor = (pregunta.get(campo) or "").strip()
    return _CARA_A_DISPLAY.get(valor)


def cara_para_emocion(nombre):
    """Traduce Feliz/Triste/Enojado/Neutral (columna 'cara') a la clave que
    entiende FaceCommand -- usado por el Juego de emociones/imitación para
    mostrar la cara ANTES de pedirle al usuario que la imite."""
    return _CARA_A_DISPLAY.get((nombre or "").strip())


def expresar_musica(pregunta, bridge, esperar=False, logger=None):
    """Si la pregunta trae algo en 'musical', le pide al `bridge` que la
    reproduzca (PlayMusic.srv vía communication_node). Vacío/ausente -> no
    hace nada, devuelve None. `esperar=True` para Reconocimiento Musical
    (la música es el enunciado, hay que escucharla completa antes de
    responder) -- ver la nota en lora_interfaces/srv/PlayMusic.srv."""
    musica = (pregunta.get("musical") or "").strip()
    if not musica:
        return None
    if logger is not None:
        logger.info(f"[música] {musica}")
    bridge.reproducir_musica(musica, esperar=esperar)
    return musica


def expresar_desplazamiento(pregunta, bridge, logger=None):
    """Si la pregunta trae algo en 'desplazamiento', le pide al `bridge`
    que publique el MotionCommand correspondiente. Vacío/ausente -> no hace
    nada, devuelve None."""
    desplazamiento = (pregunta.get("desplazamiento") or "").strip()
    if not desplazamiento:
        return None
    if logger is not None:
        logger.info(f"[desplazamiento] {desplazamiento}")

    if desplazamiento.lower().startswith("girar"):
        comando = "ROTATE360"
    else:
        comando = _DIRECCION_A_COMANDO.get(desplazamiento.strip().lower())
    if not comando:
        if logger is not None:
            logger.warning(f"[desplazamiento] valor sin comando mapeado: {desplazamiento!r}")
        return None

    bridge.mover(comando)
    return desplazamiento
