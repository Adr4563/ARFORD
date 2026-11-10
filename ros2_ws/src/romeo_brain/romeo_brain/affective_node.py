# -*- coding: utf-8 -*-
"""affective_node -- la CAPA AFECTIVA de Romeo, en una sola pieza.

Las tres etapas de la metodologia de Rocha (tesis UFF, cap. 3) viven hoy
repartidas en varios modulos que ya funcionan y NO se tocan:

  - PULSE  (percepcion del usuario): estado_afectivo.fusionar /
           percibir_categorias / confianza / *_ter.  Convierte lo que ven
           el texto (TER), la voz (SER) y la camara (FER) en UN vector VAD
           del usuario, pesando cada modalidad por su confianza.
  - ROSE   (animo del robot): la clase estado_afectivo.EstadoAfectivo.
           Parte de un animo base (perfil afectivo), empatiza con el
           usuario y decae con el tiempo hacia su base.
  - AEB    (Empathic Behavior Mapping): aeb.mapear / tono_para / tono_trivia.
           Traduce el animo de Romeo a comportamiento: cara de reposo, tono
           del LLM, movimiento del carrito, nivel L0-L4.

Esta clase NO reimplementa nada: es una FACHADA que agrupa esas tres
etapas detras de una interfaz clara, para que "la capa afectiva" sea una
unidad visible y nombrada en el codigo (y en el diagrama de nodos) en vez
de estar diseminada por el orquestador. El orquestador sigue pudiendo
llamar a los modulos directamente; esta clase es el punto unico que
documenta y ordena el flujo PULSE -> ROSE -> AEB.

Se llama affective_node por coherencia con los nodos del sistema
(expression_faces_node, communication_node...), pero NO es un nodo ROS2
aparte: la capa afectiva corre dentro del orchestrator_node, en el mismo
hilo que decide la respuesta, porque el afecto se consulta muchas veces
por turno y partirlo en otro proceso solo añadiria latencia y puntos de
fallo sin cambiar la metodologia.
"""
from . import aeb, estado_afectivo, traza_afectiva


class AffectiveLayer:
    """La capa afectiva completa de Romeo (PULSE -> ROSE -> AEB).

    Envuelve un EstadoAfectivo (ROSE) y expone las tres etapas como
    metodos. El orquestador crea UNA instancia por sesion y la reinicia
    con reiniciar()."""

    def __init__(self, perfil=None):
        # ROSE: el animo de Romeo. El perfil fija su base, empatia y
        # decaimiento (ver estado_afectivo.PERFILES / parametros_de_perfil).
        self.rose = estado_afectivo.EstadoAfectivo(perfil=perfil)

    # ─── PULSE: percepcion del usuario ──────────────────────────────────

    @staticmethod
    def percibir(probabilidades, etiquetas):
        """Una modalidad (TER/SER/FER) -> (vector_vad, confianza).

        La confianza es 1 - entropia normalizada: una distribucion plana
        (el clasificador dudando) pesa menos en la fusion."""
        return estado_afectivo.percibir(probabilidades, etiquetas)

    @staticmethod
    def percibir_categorias(probabilidades, etiquetas):
        """Como percibir(), pero devuelve tambien la emocion dominante y su
        vector por categorias (lo que usa el orquestador para loguear)."""
        return estado_afectivo.percibir_categorias(probabilidades, etiquetas)

    @staticmethod
    def fusionar(lecturas):
        """PULSE: combina las modalidades disponibles de este turno en UN
        vector del usuario, S_u = sum(c_m * S_m) / sum(c_m). `lecturas` es
        una lista de (vector_vad, confianza)."""
        return estado_afectivo.fusionar(lecturas)

    # Ventana de texto (TER) de la conversacion actual.
    @staticmethod
    def acumular_ter(probabilidades, etiquetas):
        return estado_afectivo.acumular_ter(probabilidades, etiquetas)

    @staticmethod
    def resumir_ter():
        return estado_afectivo.resumir_ter()

    @staticmethod
    def olvidar_ter():
        return estado_afectivo.olvidar_ter()

    # ─── ROSE: animo del robot ──────────────────────────────────────────

    def empatizar(self, vector_usuario, confianza=1.0):
        """ROSE: acerca el animo de Romeo al del usuario segun su empatia,
        s_r' = e*s_u + (1-e)*s_r (ponderado por la confianza de la lectura)."""
        return self.rose.empatizar(vector_usuario, confianza=confianza)

    def decaer(self, ahora=None):
        """ROSE: el animo vuelve poco a poco hacia la base del perfil."""
        return self.rose.decaer(ahora=ahora)

    def reiniciar(self):
        """Sesion nueva: el animo de Romeo y la ventana de texto se olvidan."""
        self.rose.reiniciar()
        estado_afectivo.olvidar_ter()

    # ─── AEB: comportamiento ────────────────────────────────────────────

    def comportamiento(self):
        """AEB: el animo actual de Romeo -> dict de comportamiento
        (estado, emocion, nivel L0-L4, cara, tono, movimiento, musica)."""
        return aeb.mapear(self.rose)

    @staticmethod
    def tono(comportamiento, mensaje):
        """AEB: la instruccion de estilo para el LLM segun como se siente
        Romeo y lo que le dijeron (o None en neutral)."""
        return aeb.tono_para(comportamiento, mensaje)

    @staticmethod
    def tono_trivia(comportamiento, acerto):
        """AEB en la trivia: el tono con que Romeo corrige/felicita."""
        return aeb.tono_trivia(comportamiento, acerto)

    # ─── Traza (para el panel y las metricas) ───────────────────────────

    def nueva_traza(self, entrada, antes_decaer):
        """Abre la traza afectiva del turno (romeo_afecto.jsonl)."""
        return traza_afectiva.nueva(entrada, self.rose, antes_decaer)
