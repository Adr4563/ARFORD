"""orchestrator_node -- el "cerebro" de Romeo: máquina de estados de la
conversación (Trivia / Chat libre), portado de Orchestrator_Management.py
del proyecto monolítico original (deploy-raspberry-standalone/). Mismo
comportamiento turno a turno (router puro, Trivia con corrección
determinista, Chat libre con memoria episódica BM25, personalidad) -- lo
que cambia es la integración: en vez de importar display.py/voz_output/
Carrito_Client/Musica_Client/Camara_Client directo, todo lo de hardware
pasa por `_ros_bridge.RosBridge` (tópicos/servicios hacia romeo_drivers).

Diseño de concurrencia (la pieza más delicada de esta migración): el
proceso original era 100% síncrono (un solo hilo, loop bloqueante leyendo
de una cola). Acá el turno sigue siendo síncrono EN SU LÓGICA (se procesa
un mensaje de punta a punta antes de aceptar el siguiente, protegido por
`self._turno_lock`), pero la ESPERA de cada servicio ROS2 (Speak,
PlayMusic, DetectEmotion) tiene que resolverse sin bloquear el proceso
entero -- ver la nota larga en _ros_bridge.RosBridge._llamar_servicio_sync().
Por eso este nodo corre con un MultiThreadedExecutor (ver main() al final)
y no con el rclpy.spin() default de un solo hilo.

Único parámetro propio: `perfil_afectivo` (OPTIMISTIC o EMPATHETIC -- ver
estado_afectivo.PERFILES), el Robot Affective Profile de la tesis. Todo lo de la voz lo resuelve
communication_node.

Capa afectiva (tesis de Rocha et al.): TER/PULSE estima al usuario en cada
turno, ROSE (estado_afectivo.EstadoAfectivo) mantiene el ánimo de Romeo y
el AEB (aeb.py) lo traduce a comportamiento -- cara de reposo, tono del
chat libre y un gesto físico de alegría. Este nodo hace de SENSEI: decide
cuándo se usa cada parámetro del AEB.

Cambio de comportamiento deliberado respecto del original: cuando el
usuario dice "salir", el proceso original terminaba (era una app de
terminal para probar). Acá eso solo termina la SESIÓN -- el robot dice la
despedida y arranca una sesión nueva (_iniciar_sesion()), listo para el
próximo usuario. Ver _finalizar_sesion() más abajo.
"""
import os
import random
import re
import threading
import unicodedata
import time

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node


from romeo_interfaces.msg import UserInput

from . import (aeb, agente_chat, agente_trivia, estado_afectivo, frases,
               llama_client, ordenes, preguntas, traza_afectiva)
from .medicion import activa as medicion_activa
from .medicion import medir_bloque, tiempos, turno
from ._ros_bridge import RosBridge

# A TRIVIA SOLO SI SE PIDE POR SU NOMBRE. Antes decidia un modelo (TF-IDF
# de n-gramas de CARACTERES) por parecido de letras, y eso sacaba a la gente
# de la conversacion sin motivo: "mi perro jueGA mucho" comparte trozos con
# "jugar" y se iba a TRIVIA con 0.85, igual que "ah, persegui una ardilla"
# con 0.64. Ninguna de las dos pedia jugar.
#
# Los dos errores no cuestan lo mismo: no entender que alguien quiere
# trivia se arregla repitiendolo, pero cortar una conversacion para soltar
# el menu de categorias la rompe entera. Asi que ahora hace falta decir la
# palabra.
#
# Las variantes son las que suelta el STT de verdad, vistas en los logs:
# "tribia", "Tribia", "tribu". El acento y las mayusculas dan igual
# (_normalizar_ruta). "trivial" NO cuenta: \b corta en el final de palabra.
_PEDIR_TRIVIA = re.compile(
    r"\b(trivia|trivias|tribia|tribias|trivi|tribu)\b", re.I)

# El juego de caras se pide por su cuenta (_quiere_jugar_emociones), pero
# tambien se menciona en el menu, asi que se reconoce aqui.
_PEDIR_CARAS = re.compile(r"\b(juego de (las )?caras|las caras)\b", re.I)


def _normalizar_ruta(texto):
    """Sin acentos y en minusculas: "Tribia" y "trivia" son lo mismo."""
    t = unicodedata.normalize("NFD", (texto or "").lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def enrutar(mensaje_usuario):
    """TRIVIA solo si el mensaje nombra la trivia (o el juego de caras).

    Todo lo demas es conversacion: ver el comentario de arriba."""
    texto = _normalizar_ruta(mensaje_usuario)
    if _PEDIR_TRIVIA.search(texto) or _PEDIR_CARAS.search(texto):
        return "TRIVIA"
    return "CHAT_LIBRE"


# No hay system prompt en ningún turno: la personalidad viene HORNEADA en
# los modelos fine-tuneados. El original tenía un personalidad.py que armaba
# un perfil Big Five (OCEAN) y lo pasaba como `persona_str` por toda la
# cadena de turno; se eliminó porque para romeo-trivia/romeo-chat devolvía ""
# a propósito, y su único consumidor real -- comentar_resultado_emocion() --
# ya estaba sin caller. Si algún día se usa un modelo base SIN fine-tuning,
# el system prompt hay que volver a construirlo (está en el historial de
# git, y los valores OCEAN en el README principal).


# ══════════════════════════════════════════════════════════════════════
# Juego de emociones -- el usuario pone caras y la CÁMARA las juzga, no el
# texto. Antes salía de las preguntas del dataset ("Juego de emociones" y
# "Juego de imitación", 10 en total), así que qué caras tocaban y cuántas
# dependía de lo que el azar sacara del JSONL. Ahora es una secuencia fija:
# siempre estas cuatro, siempre en este orden.
#
# Son justo las cuatro que el detector sabe devolver -- ver
# ETIQUETAS_RELEVANTES en romeo_drivers/_emotion_detector.py, que mapea las
# 8 clases del modelo FER+ a estas 4. Pedir una quinta (sorpresa, miedo)
# sería pedir algo que la cámara nunca va a reconocer.
# ══════════════════════════════════════════════════════════════════════
SECUENCIA_EMOCIONES = ["Feliz", "Triste", "Enojado", "Neutral"]

# Frases con las que pide cada cara, una por emoción para que no suene
# repetitivo decir cuatro veces "hazme una cara de X".

# Lo que dice cuando la cámara no encuentra ninguna cara. Es frecuente (se
# midió: ~1 de cada 6 intentos) si el usuario se mueve o queda fuera del
# encuadre, así que tiene que sonar natural y no a error.

# Palabras con las que el usuario pide este juego. El router las manda a
# TRIVIA (son "juego"), así que se detectan antes de elegir categoría.
#
# Van SIN TILDE porque se comparan contra el mensaje ya normalizado (ver
# _quiere_jugar_emociones): el STT y el teclado escriben "emoción" e
# "imitación" con tilde, y comparar tal cual no las encontraba.
_PIDE_EMOCIONES = (
    "emocion", "emociones", "cara", "caras", "gesto", "gestos",
    "imitacion", "imitar", "mueca", "muecas", "expresion", "expresiones",
)

# "¿Que haces?", "¿que sabes hacer?" -- lo unico que dispara el menu de
# capacidades, que ya no va en el saludo. Sin tildes porque se compara
# contra preguntas._sin_tildes().
#
# Se responde con una frase fija y no con el LLM a proposito: el chat
# esta afinado en EmpatheticDialogues, que no sabe nada de trivias ni de
# caras, y a "¿que haces?" contesta "estoy para acompañarte y conversar"
# por mucho que el prompt le liste lo demas.
_PIDE_CAPACIDADES = (
    "que haces", "que sabes hacer", "que puedes hacer", "que mas haces",
    "para que sirves", "en que me puedes ayudar", "que cosas haces",
    "cuales son tus funciones", "que sabes",
)


# Saludo de apertura de cada sesión -- se elige uno al azar, así que dos
# usuarios seguidos no oyen lo mismo. Todas tutean (no vosean): el fine-tune
# de Chat libre que usa el robot está entrenado en tuteo, y mezclar las dos
# formas en la misma conversación suena raro.
# La presentacion va SEPARADA del saludo: primero Romeo dice quien es y que
# sabe hacer, despues pregunta el nombre. Se dice en dos frases y no en una
# sola larga porque la voz tarda y una frase de 30 palabras se hace eterna.
#
# Son cinco variantes al azar, igual que los saludos: dos sesiones seguidas
# no suenan identicas. Todas dicen el nombre y las tres cosas que sabe hacer
# (trivia, caras, conversar), que es lo unico que el usuario necesita saber
# para elegir.


# Lo que dice después de que el usuario da su nombre: presenta las dos cosas
# que sabe hacer (trivia y charlar) y le deja elegir. Antes era UNA sola
# frase fija; ahora son 5 variantes al azar.

# Donde cae la fila de cada turno con ROMEO_MEDIR=1 (ver _anotar_turno).
CSV_TURNOS = os.environ.get("ROMEO_CSV_TURNOS",
                            os.path.expanduser("~/romeo_medicion_turnos.csv"))

# Exactitud del veredicto afectivo: una fila por cada cara pedida en el juego
# de emociones, con lo que se pidio, lo que vio la camara y si acerto. Con
# ROMEO_MEDIR=1. Permite sacar aciertos/total sin leer el log a mano.
CSV_VEREDICTO = os.environ.get(
    "ROMEO_CSV_VEREDICTO",
    os.path.expanduser("~/romeo_medicion_veredicto.csv"))

PREGUNTAS_POR_TANDA = 5
PAUSA_CAMBIO_CARA = 1       # segundos en 'content'/veredicto antes de la próxima cara
PAUSA_ANTES_ACCION_FISICA = 1
# Segundos que se ve cada cara de la secuencia "transformacion" (bostezo ->
# pingpong -> transformacion) antes de pasar a la siguiente. Ajustable sin
# recompilar para que calce con la duracion de las animaciones.
SECUENCIA_CARA_S = float(os.environ.get("ROMEO_CARA_SECUENCIA_S", "2.5"))
# Segundos entre pedirle la cara y capturarla. El detector ya tarda 2.5-4.6s
# en resolver (medido en la Pi), así que esto es solo el margen para que
# termine de oír la frase y ponga la cara.
PAUSA_POSAR = 1

# Cuanto se ve la cara de una emocion (happy/sad/angry) antes de volver a
# 'content'. Corta a proposito: la emocion se EXPRESA como una reaccion a
# lo que dijo el usuario, no se queda pegada en la cara mientras el animo
# decae. Y solo se muestra en el turno en que se percibio (ver
# _mostrar_cara_reposo): antes reaparecia en cada turno hasta que el animo
# volvia a la base.
DURACION_EMOCION = float(os.environ.get("ROMEO_DURACION_EMOCION", "1.25"))

# Frases que, a mitad de una tanda de trivia, señalan que el usuario se
# quiere ir a otra cosa -- fallback de _quiere_salir_trivia() cuando no se
# puede mirar las listas de frases de arriba (ver ese método).
_PALABRAS_RESPUESTA_CORTA = 4

RUTAS = ["TRIVIA", "CHAT_LIBRE"]

# Fuentes de la ventana de percepcion afectiva (tesis de Rocha et al.,
# sec. 3.4.1). Todas entregan una observacion por frase y PULSE las fusiona
# (ec. 4.4). Ver _percibir_turno().
#   TER  texto  -> RoBERTuito de la PC sobre la frase (_ter_engine)
#   FER  cara   -> FER+ sobre la camara de fondo, en la Pi (PerceiveFace)
#   SER  voz    -> modelo de la PC sobre el audio de la frase (_ser_engine)
FUENTES = ("TER", "FER", "SER")

# Cada cuanto se reintenta el arranque mientras communication_node no pueda
# hablar, y cuanto se espera como maximo (ver _arrancar). En la Pi tarda ~11 s
# en activarse: carga la voz VITS y el modelo de STT.
ESPERA_SPEAK = 1.0
MAX_ESPERA_SPEAK = 45.0




class OrchestratorNode(Node):

    def __init__(self):
        super().__init__('orchestrator_node')

        self._bridge = RosBridge(self)
        # Robot Affective Profile (tesis, sec. 3.4.2): fija empatia, animo
        # base, decaimiento y latencia de ROSE. Se elige en
        # romeo_params.yaml; las variables ROMEO_EMPATIA etc. lo afinan.
        self.declare_parameter('perfil_afectivo', estado_afectivo.PERFIL_POR_DEFECTO)
        perfil = self.get_parameter('perfil_afectivo').value
        # Que fuentes de la ventana de percepcion entran a PULSE. Quitar una
        # de la lista la desactiva sin tocar codigo (p. ej. ["TER"] para
        # probar solo con texto).
        self.declare_parameter('fuentes_percepcion', list(FUENTES))
        self._fuentes = {str(f).strip().upper()
                         for f in self.get_parameter('fuentes_percepcion').value}
        # El animo de Romeo, que persiste entre turnos y decae solo. Vive
        # en el nodo y no en self.estado porque self.estado se reconstruye
        # entero en cada sesion, y lo que se quiere del afecto es
        # justamente que dure. Se reinicia a mano en _iniciar_sesion().
        self._afecto = estado_afectivo.EstadoAfectivo(perfil=perfil)
        if self._afecto.perfil != str(perfil).strip().upper():
            self.get_logger().warning(
                f"[afecto] perfil {perfil!r} desconocido, se usa {self._afecto.perfil}")
        self.get_logger().info(
            f"[afecto] perfil {self._afecto.perfil}: empatia "
            f"{self._afecto.empatia}, base {self._afecto.animo_base}, "
            f"decaimiento {self._afecto.decaimiento}, latencia "
            f"{self._afecto.latencia}s")
        # Lecturas afectivas del turno en curso, pendientes de
        # fusionar. Ver _aplicar_afecto().
        self._lecturas_turno = []
        # Cada paso del calculo afectivo del turno en curso, para la
        # pagina /calculo del panel (ver traza_afectiva).
        self._traza = None
        traza_afectiva.reiniciar()
        # True si en este turno el animo cambio por algo que se percibio:
        # solo entonces se muestra la cara de la emocion (una vez).
        self._emocion_pendiente = False
        self._timer_emocion = None
        self.estado = None
        self._intentos_arranque = 0
        self._turno_lock = threading.Lock()
        self._grupo_turno = ReentrantCallbackGroup()

        # Que Ollama cargue los dos modelos ANTES del primer turno.
        #
        # Medido: la primera llamada a un modelo cuesta ~5 s (Ollama lo lee
        # de disco) y las siguientes 55-400 ms. Sin esto, esos 5 s los paga
        # el usuario en mitad de la conversacion, y dos veces -- una por el
        # chat y otra por el corrector de trivia, que es otro modelo.
        #
        # En un hilo aparte porque el nodo tiene que quedar listo ya: si la
        # PC esta apagada, precargar() tarda hasta su timeout, y mientras
        # tanto el robot debe poder arrancar igual (ya sabe degradar).
        threading.Thread(
            target=llama_client.precargar,
            args=([llama_client.CHAT_MODEL, agente_trivia.CORRECTOR_LLM],
                  self.get_logger()),
            daemon=True,
        ).start()

        self.create_subscription(
            UserInput, '/romeo/user_input', self._on_user_input, 10,
            callback_group=self._grupo_turno)

        self._precargar_modelos()

        # Arranca la primera sesión un instante después de construirse el
        # nodo -- no directo en __init__ porque _iniciar_sesion() hace
        # llamadas de servicio BLOQUEANTES (Speak), y esas solo se resuelven
        # una vez que el executor está spinneando (ver la nota de
        # RosBridge._llamar_servicio_sync). El timer se cancela apenas
        # dispara una vez.
        self._timer_arranque = self.create_timer(
            ESPERA_SPEAK, self._arrancar, callback_group=self._grupo_turno)

    def _arrancar(self):
        """Arranca la primera sesión, pero solo cuando communication_node ya
        puede hablar.

        Sin esta espera, el orquestador decía la presentación a los 0.5 s de
        existir y communication_node todavía estaba en on_activate() cargando
        la voz y whisper-small (~11 s en la Pi): Speak respondía
        `success=False` con el warning "Speak llamado sin estar activo" y la
        frase se perdía. El usuario veía el texto en el log pero no oía nada.

        El timer se re-dispara cada ESPERA_SPEAK hasta que el servicio
        responde, con un tope para no quedarse colgado si el nodo de voz no
        llega a arrancar (en ese caso se sigue igual: el chat por teclado
        funciona y el log deja constancia).
        """
        if not self._bridge.speak_client.service_is_ready():
            self._intentos_arranque += 1
            if self._intentos_arranque * ESPERA_SPEAK < MAX_ESPERA_SPEAK:
                self.get_logger().info(
                    '[arranque] esperando a que communication_node pueda hablar...')
                return
            self.get_logger().warning(
                f'[arranque] Speak sigue sin responder tras '
                f'{MAX_ESPERA_SPEAK:.0f}s -- se arranca igual, sin voz')
        self._timer_arranque.cancel()
        self._iniciar_sesion()


    # ══════════════════════════════════════════════════════════════════
    # Warmup de Ollama -- portado de _precargar_modelos()/_precargar_uno()/
    # originales. No son llamadas a servicio
    # ROS2 (Ollama es HTTP directo, ver llama_client.py), así que no tienen
    # el problema de deadlock de arriba -- se pueden lanzar ya en __init__.
    # ══════════════════════════════════════════════════════════════════

    def _precargar_uno(self, modelo):
        try:
            llama_client.generar_respuesta(
                [{"role": "user", "content": "hola"}], max_tokens=1, modelo=modelo)
        except Exception as e:
            self.get_logger().warning(f"[warmup] no se pudo precargar {modelo}: {e}")

    def _precargar_modelos(self):
        # Solo CHAT_MODEL. Antes se precargaban tres:
        #
        #   TRIVIA_MODEL        -- se cargaba sin que nadie lo llamara nunca:
        #                          las preguntas salen del banco
        #                          (preguntas.py -> preguntas.jsonl).
        #   SALIDA_TRIVIA_MODEL -- lo sustituyen las listas de frases de
        #                          _quiere_salir_trivia(), que deciden en
        #                          microsegundos en vez de segundos.
        threading.Thread(target=self._precargar_uno, args=(llama_client.CHAT_MODEL,), daemon=True).start()

    # ══════════════════════════════════════════════════════════════════
    # Helpers de Chat libre / prompts -- portados tal cual de
    # Orchestrator_Management.py.
    # ══════════════════════════════════════════════════════════════════

    def responder(self, mensaje_usuario, tono=None):
        """Chat libre -- delega en agente_chat.

        ATRAPA LOS FALLOS DE RED. El LLM vive en otra maquina, asi que
        apagarla, un cable suelto o un cambio de IP dan ConnectionError. Sin
        este try el nodo ENTERO se moria: la excepcion subia por el callback
        de /romeo/user_input hasta executor.spin() y tumbaba el proceso, con
        lo que el robot se quedaba mudo hasta relanzarlo a mano.

        El resto del turno ya degrada en vez de romperse (ver
        _llamar_servicio_sync en _ros_bridge), asi que esto es lo mismo para
        el chat: se dice que no se puede ahora y se sigue vivo."""
        try:
            return agente_chat.responder(mensaje_usuario, tono=tono)
        except Exception as e:  # noqa: BLE001 - seguir vivo es lo que importa
            self.get_logger().warning(
                f"[chat] el LLM en {llama_client.CHAT_SERVER_HOST} no "
                f"responde ({type(e).__name__}); se sigue sin el")
            return ("Ahora mismo no puedo pensar bien. ¿Jugamos una trivia "
                    "mientras tanto?")

    # ══════════════════════════════════════════════════════════════════
    # Trivia: reacciones fijas, resolución de tema, salida a mitad de
    # pregunta -- portados tal cual.
    # ══════════════════════════════════════════════════════════════════

    def _aplicar_afecto(self):
        """Fusiona las lecturas de este turno (PULSE) y actualiza el animo
        una sola vez (ROSE).

        Se llama cada vez que se añade una lectura, y es idempotente: lo
        que ya se aplico se descarta de la lista. Asi, si en un turno
        perciben varias modalidades, el animo se mueve una vez con la
        media ponderada de todas y no una vez por cada una.

        Cada lectura es (fuente, vector_vad, confianza, dominante).

        Si TODAS las fuentes vieron neutral, no se mueve el animo: neutral
        es el origen del espacio VAD, y como el animo base de Romeo es
        positivo, acercarlo al origen se veria como tristeza. Basta con que
        una fuente exprese algo para que se fusionen todas, y entonces las
        neutras moderan la lectura, como en la tesis."""
        lecturas = getattr(self, "_lecturas_turno", None)
        if not lecturas:
            return
        traza = self._traza
        if traza is not None:
            traza["fusion"] = {
                "entradas": [{"fuente": f, "vad": [round(x, 4) for x in v],
                              "confianza": round(c, 4), "dominante": d}
                             for f, v, c, d in lecturas],
                "todos_neutral": all(d == "neutral" for _f, _v, _c, d in lecturas),
            }
        if all(dom == "neutral" for _f, _v, _c, dom in lecturas):
            self._lecturas_turno = []
            return
        self._lecturas_turno = [(v, c) for _f, v, c, _d in lecturas]
        vector = estado_afectivo.fusionar(self._lecturas_turno)
        if traza is not None and vector is not None:
            traza["fusion"]["suma_confianzas"] = round(sum(c for _v, c in self._lecturas_turno), 4)
            traza["fusion"]["s_u"] = [round(x, 4) for x in vector]
        if vector is not None and len(self._lecturas_turno) > 1:
            self.get_logger().info(
                f"[afecto] usuario (fusion de "
                f"{'+'.join(f for f, *_ in lecturas)}): "
                f"v={vector[0]:+.2f} a={vector[1]:+.2f} d={vector[2]:+.2f}")
        # La confianza con la que se cree la lectura fusionada es la mayor
        # de las que entraron: si la camara estaba segura, que el texto no
        # lo estuviera no debe frenar la reaccion.
        conf = max((c for _s, c in self._lecturas_turno), default=0.0)
        self._lecturas_turno = []
        if vector is None:
            return
        antes = self._afecto.vector
        self._afecto.empatizar(vector, conf)
        self._emocion_pendiente = True
        v, a, d = self._afecto.vector
        if traza is not None:
            traza["empatia"] = {"e": self._afecto.empatia,
                                "s_u": [round(x, 4) for x in vector],
                                "antes": [round(x, 4) for x in antes],
                                "despues": [round(x, 4) for x in self._afecto.vector]}
        self.get_logger().info(
            f"[afecto] v {antes[0]:+.2f}->{v:+.2f}  "
            f"a {antes[1]:+.2f}->{a:+.2f}  d {antes[2]:+.2f}->{d:+.2f}")

    def _percibir_cara(self, t_inicio, t_fin):
        """FER de la ventana de percepcion afectiva (tesis, sec. 3.4.1.2):
        lo que vio la camara MIENTRAS el usuario decia esta frase.

        La ventana la marca el propio usuario: t_inicio/t_fin vienen con el
        mensaje (communication_node los saca del VAD). Por teclado no hay
        ventana y no se mira la camara.

        expression_faces_node ya resumio los cuadros (media ponderada
        por la calidad YuNet, normalizada, sec. 4.1.4.1); aca se aplica
        PULSE: confianza por margen top-1/top-2 x fiabilidad del FER y
        CAT -> VAD con la distribucion entera (ec. 4.3). La lectura queda en _lecturas_turno para la fusion (ver
        _aplicar_afecto, que tambien decide que hacer si todo es neutral)."""
        if not t_fin:
            return
        etiquetas, probs, cuadros = self._bridge.percibir_cara(t_inicio, t_fin)
        if cuadros <= 0:
            self.get_logger().info(
                f"[afecto] la camara no vio cara en la frase ({t_fin - t_inicio:.1f} s)")
            if self._traza is not None:
                self._traza["fer_sin_cara"] = round(t_fin - t_inicio, 1)
            return
        vector, c, dominante = estado_afectivo.percibir_categorias(
            probs, etiquetas)
        if vector is None:
            return
        self.get_logger().info(
            f"[afecto] la camara vio {dominante} (confianza {c:.2f}, "
            f"{cuadros} cuadros en {t_fin - t_inicio:.1f} s)")
        self._lecturas_turno.append(("FER", vector, c, dominante))
        self._trazar_lectura("FER", probs, etiquetas, cuadros=cuadros)

    def _percibir_voz(self, ser):
        """SER: la emocion de la VOZ de esta frase. La calculo la PC con el
        mismo audio que transcribio (_ser_engine.py) y llega en el mensaje
        como categorias con probabilidades; aca se aplica PULSE igual que
        al FER: margen top-1/top-2 x fiabilidad del SER y CAT -> VAD (4.3)."""
        if not ser or not ser[0]:
            return
        vector, c, dominante = estado_afectivo.percibir_categorias(
            ser[1], ser[0])
        if vector is None:
            return
        self.get_logger().info(
            f"[afecto] la voz suena a {dominante} (confianza {c:.2f})")
        self._lecturas_turno.append(("SER", vector, c, dominante))
        self._trazar_lectura("SER", ser[1], ser[0])

    def _trazar_lectura(self, fuente, probs, etiquetas, **extra):
        if self._traza is None:
            return
        detalle = traza_afectiva.lectura(fuente, probs, etiquetas, **extra)
        if detalle is not None:
            self._traza["lecturas"].append(detalle)

    def _cara_reposo(self):
        """La cara con la que Romeo se queda ESCUCHANDO, segun como se
        siente (ver estado_afectivo).

        Es el unico sitio donde el estado afectivo decide la cara. Las
        demas -- 'speaking', 'countdown', y los veredictos del juego de
        imitacion -- comunican lo que el robot esta HACIENDO, y
        sustituirlas por el animo haria que dejara de verse si esta
        hablando o esperando. En reposo, en cambio, la cara no esta
        diciendo nada mas, asi que puede decir como esta.

        Con el animo cerca de su base devuelve 'content', que es lo que se
        mostraba siempre: el comportamiento solo cambia cuando de verdad
        hay algo que mostrar.

        La decide el AEB (ver _comportamiento)."""
        return self._comportamiento()["cara"]

    def _mostrar_cara_reposo(self, cara=None):
        """Pone la cara de reposo. Si el animo cambio en este turno, muestra
        la cara de esa emocion DURACION_EMOCION segundos y vuelve sola a
        'content'; si no, pone 'content' directamente.

        La vuelta va en un temporizador para no frenar el turno, y solo
        ocurre si la cara sigue siendo esa: si mientras tanto Romeo empezo
        a hablar ('speaking'), no se pisa."""
        cara = cara or self._cara_reposo()
        if cara == "content" or not self._emocion_pendiente:
            self._bridge.mostrar_cara("content")
            return
        self._emocion_pendiente = False
        self._bridge.mostrar_cara(cara)
        if self._timer_emocion is not None:
            self._timer_emocion.cancel()

        def _volver():
            if self._bridge._cara_actual == cara:
                self._bridge.mostrar_cara("content")

        self._timer_emocion = threading.Timer(DURACION_EMOCION, _volver)
        self._timer_emocion.daemon = True
        self._timer_emocion.start()

    def _comportamiento(self):
        """AEB: lo que el animo actual de Romeo pide expresar (cara de
        reposo, tono, movimiento). Ver aeb.py.

        Solo DEVUELVE los parametros; cada caller decide cuales usa. Asi la
        trivia toma solo la cara de reposo y deja el movimiento a su
        dataset, y el chat libre los usa todos."""
        comp = aeb.mapear(self._afecto)
        # Se registra el vector, no solo el resultado: la emocion se queda
        # igual durante varios turnos, asi que sin el vector no se ve si el
        # animo se esta moviendo o esta clavado.
        v, a, d = self._afecto.vector
        self.get_logger().info(
            f"[aeb] v={v:+.2f} a={a:+.2f} d={d:+.2f} -> {comp['estado']} "
            f"(cara {comp['cara']}, movimiento {comp['movimiento']})")
        return comp

    @staticmethod
    def comentar_resultado(esperada, acerto):
        plantilla = random.choice(frases.RESPUESTAS_TRIVIA_ACIERTO if acerto
                               else frases.RESPUESTAS_TRIVIA_ERROR)
        return plantilla.format(esperada=esperada)

    @staticmethod
    def _enrutar_mensaje(mensaje_usuario):
        ruta = enrutar(mensaje_usuario)
        return ruta if ruta in RUTAS else "CHAT_LIBRE"

    @staticmethod
    def _quiere_salir_trivia(mensaje_usuario, pregunta_pendiente=None):
        """True si el mensaje contiene una frase de _SALIR_TRIVIA (quiere
        dejar el juego) o de _TEMA_PERSONAL (quiere hablar de lo suyo).

        Solo busca la frase, a propósito. Antes decidía esto
        romeo-salida-trivia-v2, un LLM por Ollama: costaba segundos en cada
        turno de trivia y encima su veredicto pisaba las palabras clave. Las
        frases con las que se pide salir son un puñado y están todas en las
        dos listas; si falta alguna, se añade ahí y no hace falta ni LLM ni
        clasificador.

        `pregunta_pendiente` se mantiene en la firma porque el caller lo pasa,
        pero ya no se usa.
        """
        texto = mensaje_usuario.strip().lower()
        return (any(f in texto for f in frases.SALIR_TRIVIA)
                or any(f in texto for f in frases.TEMA_PERSONAL))

    # ══════════════════════════════════════════════════════════════════
    # Reacciones (voz -> cara -> música+motores) y flujo de una tanda.
    # ══════════════════════════════════════════════════════════════════

    def _reaccionar_veredicto(self, cara, pregunta, musica_ya_sonada=False):
        """GARANTÍA DE ORDEN: para cuando se llega acá, la voz del
        veredicto YA se dijo (bloqueante, ver manejar_trivia/_jugar_emociones
        más abajo) -- acá solo va cara + música + motores, disparados en
        paralelo entre sí (publish/servicio no bloqueante, salvo el sleep
        final que es solo para que la cara alcance a verse antes del
        próximo cambio, no para "esperar" a música/motores)."""
        self._bridge.mostrar_cara(cara)
        if not musica_ya_sonada:
            agente_trivia.expresar_musica(pregunta, self._bridge, logger=self.get_logger())
        # El movimiento de la trivia (columna 'desplazamiento' del dataset)
        # queda desactivado: el UNICO sistema que mueve a Romeo es el AEB
        # (ver aeb.py). En la trivia solo cambia la cara y suena la musica.
        time.sleep(PAUSA_CAMBIO_CARA)

    def _preguntar_siguiente(self):
        """Lanza la siguiente pregunta. False si ya no quedan.

        Al vaciarse la cola se recarga sola con otras tantas: el juego
        encadena preguntas sin cortarse cada 5. Devuelve False solo
        cuando se acabaron las 200 con respuesta esperada."""
        estado = self.estado
        if not estado["cola_preguntas"]:
            mas = preguntas.preguntas_mezcladas(
                estado["ya_usados"], cantidad=PREGUNTAS_POR_TANDA)
            if not mas:
                return False
            estado["ya_usados"].update(p["id"] for p in mas)
            estado["cola_preguntas"] = mas
        actual = estado["cola_preguntas"].pop(0)
        estado["pregunta_pendiente"] = actual
        self._bridge.mostrar_cara("speaking")
        self.get_logger().info(f"Asistente [{actual['cara']}]: {actual['pregunta']}")
        self._bridge.hablar(actual["pregunta"])  # bloquea hasta terminar de decirla

        # Reconocimiento Musical: la canción ES el enunciado -- suena
        # bloqueando, ANTES de habilitar la respuesta (ver la nota completa
        # en el original sobre el bug que esto arregla).
        es_musical = bool(actual.get("musical"))
        if es_musical:
            self._bridge.mostrar_cara("countdown")
        estado["musica_ya_sonada"] = bool(agente_trivia.expresar_musica(
            actual, self._bridge, esperar=True, logger=self.get_logger()))
        if es_musical:
            self._bridge.mostrar_cara("speaking")

        time.sleep(PAUSA_CAMBIO_CARA)
        # Mientras se presenta la pregunta y se espera la respuesta, cara
        # neutral -- NO la afectiva. El animo (ROSE/AEB) ya se mostro tras
        # la respuesta anterior (ver _manejar_trivia); mostrarlo otra vez
        # aqui lo dejaba pegado al enunciado de la pregunta, como si Romeo
        # reaccionara a algo que el usuario todavia no contesto.
        self._bridge.mostrar_cara("content")
        return True

    def _cerrar_tanda(self):
        """Cierra el juego y dice el marcador.

        Ya no salta cada 5 preguntas: _preguntar_siguiente recarga la cola
        sola, asi que aqui solo se llega cuando se acabaron TODAS las
        preguntas con respuesta esperada. Al salir a mano el marcador lo
        dice _manejar_trivia."""
        estado = self.estado
        estado["en_trivia"] = False
        self._bridge.mostrar_cara("speaking")
        cierre = "Ya no me quedan preguntas."
        if estado["total"]:
            cierre += f" Acertaste {estado['aciertos']} de {estado['total']}."
        cierre += " ¿Charlamos un rato?"
        self.get_logger().info(f"Asistente: {cierre}")
        self._bridge.hablar(cierre)
        time.sleep(PAUSA_CAMBIO_CARA)
        self._mostrar_cara_reposo()

    def _pedir_una_emocion(self, objetivo):
        """Pide UNA cara, la captura y la juzga.

        Devuelve True/False según acierte, o None si la cámara no vio
        ninguna cara (eso no cuenta para el marcador: no es culpa del
        usuario)."""
        # Romeo pone primero la cara que pide, para que el usuario la imite.
        self._bridge.mostrar_cara(agente_trivia.cara_para_emocion(objetivo) or "content")
        pedido = frases.una(frases.PEDIDOS_EMOCION[objetivo])
        self.get_logger().info(f"Asistente: {pedido}")
        self._bridge.hablar(pedido)
        time.sleep(PAUSA_POSAR)   # que le dé tiempo a poner la cara

        detectada, confianza, ok = self._bridge.detectar_emocion()
        # Lo que ve la camara aca NO entra al animo de Romeo (ROSE). En
        # este juego el usuario IMITA la cara que Romeo le pide: no esta
        # mostrando lo que siente. Si entrara, pedir "pon cara triste"
        # entristeceria a Romeo, que es justo lo contrario de una lectura
        # afectiva. Para percibir al usuario por la camara hace falta una
        # ventana de percepcion fuera del juego (tesis, sec. 3.4.1.2).
        self._bridge.mostrar_cara("speaking")

        if not ok or detectada is None:
            aviso = frases.una(frases.SIN_CARA)
            self.get_logger().info(f"[cámara] sin cara -- {aviso}")
            self._bridge.hablar(aviso)
            self._anotar_veredicto(objetivo, None, None, None)
            return None

        acerto = detectada == objetivo
        self.get_logger().info(
            f"[cámara] pedido={objetivo} detectado={detectada} "
            f"(confianza {confianza:.0%}) -> {'OK' if acerto else 'MAL'}")
        self._anotar_veredicto(objetivo, detectada, confianza, acerto)
        if acerto:
            self._bridge.hablar(random.choice(
                ["¡Esa es!", "¡Perfecto!", "¡Muy bien!", "¡Clavado!"]))
            self._bridge.mostrar_cara("happy")
        else:
            # Se le dice QUÉ cara vio: es más útil que un "incorrecto" seco,
            # y le explica por qué no acertó.
            self._bridge.hablar(f"Casi. Te vi cara de {detectada.lower()}.")
            self._bridge.mostrar_cara("sad")
        time.sleep(PAUSA_CAMBIO_CARA)
        return acerto

    def _obedecer(self, orden):
        """Ejecuta una orden directa de ordenes.interpretar().

        La cara pedida se queda hasta el turno siguiente, igual que la del
        estado afectivo: _mostrar_cara_reposo() la repondria al momento y
        el usuario no llegaria a verla."""
        respuesta = frases.una(frases.ORDENES)
        self.get_logger().info(
            f"[orden] {orden['tipo']} -> {orden['valor']}")
        self.get_logger().info(f"Asistente: {respuesta}")

        if orden["tipo"] == "cara":
            self._bridge.hablar(respuesta)
            self._bridge.mostrar_cara(orden["valor"])
            return

        if orden["tipo"] == "secuencia_caras":
            # "Transformacion": muestra las caras una tras otra y al final
            # vuelve a reposo. Cada cara se queda en pantalla SU animacion
            # entera: 'duraciones' trae la duracion real de cada .mp4
            # (ver ordenes.SECUENCIA_TRANSFORMACION), asi la secuencia se
            # reproduce completa en vez de cortarse a un tiempo fijo. Si una
            # cara no trae duracion, cae a SECUENCIA_CARA_S como respaldo.
            self._bridge.hablar(respuesta)
            duraciones = orden.get("duraciones") or []
            for i, cara in enumerate(orden["valor"]):
                self._bridge.mostrar_cara(cara)
                espera = duraciones[i] if i < len(duraciones) else SECUENCIA_CARA_S
                time.sleep(espera)
            self._mostrar_cara_reposo()
            return

        self._bridge.mostrar_cara("speaking")
        self._bridge.hablar(respuesta)
        self._bridge.mover(orden["valor"])
        self._mostrar_cara_reposo()

    @staticmethod
    def _pregunta_capacidades(mensaje):
        """True si el usuario pregunta que sabe hacer Romeo.

        Normaliza quitando tildes, igual que _quiere_jugar_emociones: el
        STT transcribe "qué" acentuado y _PIDE_CAPACIDADES va sin tilde."""
        limpio = preguntas._sin_tildes(mensaje)
        return any(p in limpio for p in _PIDE_CAPACIDADES)

    @staticmethod
    def _quiere_jugar_emociones(mensaje):
        """True si el usuario pide el JUEGO de caras (no UNA cara concreta).

        Normaliza quitando tildes: el STT transcribe "emoción" e
        "imitación" acentuadas, y _PIDE_EMOCIONES las lleva sin tilde.

        Si la frase nombra una EMOCION concreta (feliz, triste, enojado,
        neutral) NO es el juego: es un pedido de esa cara ("cara de feliz",
        "pon triste"), que ya resolvio ordenes.interpretar() antes. Sin esto,
        cualquier "cara feliz" arrancaba el juego de las 4 emociones en vez
        de poner la cara. El juego se pide sin nombrar emocion: "juguemos a
        las caras", "el juego de los gestos", "a imitar muecas"."""
        limpio = preguntas._sin_tildes(mensaje)
        if not any(p in limpio for p in _PIDE_EMOCIONES):
            return False
        # Nombra una emocion concreta -> es pedir esa cara, no el juego.
        if any(re.search(rf"\b{e}", limpio)
               for e in ("feliz", "alegre", "content", "triste", "enoja",
                         "enfada", "molest", "bravo", "neutral", "normal",
                         "seri")):
            return False
        return True

    def _jugar_emociones(self):
        """La ronda completa: las 4 emociones de SECUENCIA_EMOCIONES, una
        tras otra, y al final el marcador. No usa el dataset de preguntas --
        ver la nota de SECUENCIA_EMOCIONES."""
        estado = self.estado
        estado["en_trivia"] = False   # no es una tanda de trivia

        intro = (f"¡Vamos a jugar con las caras! Te voy a pedir "
                  f"{len(SECUENCIA_EMOCIONES)} y la cámara te mira. ¿Listo?")
        self.get_logger().info(f"Asistente: {intro}")
        self._bridge.mostrar_cara("speaking")
        self._bridge.hablar(intro)

        aciertos = juzgadas = 0
        for objetivo in SECUENCIA_EMOCIONES:
            acerto = self._pedir_una_emocion(objetivo)
            if acerto is not None:
                juzgadas += 1
                aciertos += acerto

        self._bridge.mostrar_cara("speaking")
        if juzgadas == 0:
            cierre = "No pude verte bien en ninguna. ¿Probamos otra vez?"
        elif aciertos == juzgadas:
            cierre = f"¡Las {juzgadas} perfectas! Eres muy bueno con las caras."
        else:
            cierre = (f"Terminamos: acertaste {aciertos} de {juzgadas}. "
                       f"¿Jugamos otra vez o prefieres una trivia?")
        self.get_logger().info(f"Asistente: {cierre}")
        self._bridge.hablar(cierre)
        time.sleep(PAUSA_CAMBIO_CARA)
        self._mostrar_cara_reposo()

    def _iniciar_tanda(self, categoria=None):
        """Carga preguntas y lanza la primera.

        `categoria` None -- lo normal -- mezcla todas las familias. Con una
        categoria se filtran, que es lo que hace _manejar_trivia cuando el
        usuario pide un tema concreto.

        La cola se rellena de PREGUNTAS_POR_TANDA en PREGUNTAS_POR_TANDA y
        se vuelve a llamar sola al vaciarse (ver _preguntar_siguiente), asi
        que el juego no se corta cada 5: sigue hasta que el usuario pida
        salir o se acaben las 200 preguntas con respuesta."""
        estado = self.estado
        if categoria:
            tanda = preguntas.preguntas_por_categoria(
                categoria, estado["ya_usados"], cantidad=PREGUNTAS_POR_TANDA)
        else:
            tanda = preguntas.preguntas_mezcladas(
                estado["ya_usados"], cantidad=PREGUNTAS_POR_TANDA)
        if not tanda:
            aviso = "Ya jugamos todas las que tengo. ¿Charlamos un rato?"
            self.get_logger().info(f"Asistente: {aviso}")
            self._bridge.hablar(aviso)
            estado["en_trivia"] = False
            self._mostrar_cara_reposo()
            return
        estado["ya_usados"].update(p["id"] for p in tanda)
        estado["cola_preguntas"] = tanda
        self._preguntar_siguiente()

    def _manejar_trivia(self, mensaje_usuario):
        estado = self.estado

        if estado["esperando_tema"]:
            categoria = preguntas.resolver_categoria(mensaje_usuario)
            if categoria is None:
                # No se entendió la categoría. Antes se repetía la lista
                # ENTERA cada vez y sin límite: con el STT fallando dos o
                # tres veces seguidas, el usuario oía el mismo párrafo de
                # 7 segundos una y otra vez. Insoportable.
                #
                # Ahora la primera vez se repite corto (solo las opciones,
                # sin el "no te entendí" ni la coletilla de las caras), y a
                # la SEGUNDA se elige una categoría y se empieza a jugar:
                # cualquier tema es mejor que seguir atascado en el menú.
                estado["intentos_tema"] = estado.get("intentos_tema", 0) + 1
                cats = preguntas.categorias_disponibles()
                if estado["intentos_tema"] < 2:
                    repregunta = f"¿{', '.join(cats[:-1])} o {cats[-1]}?"
                    self.get_logger().info(f"Asistente: {repregunta}")
                    self._bridge.hablar(repregunta)
                    return
                # Antes se elegia una categoria al azar y se empezaba a
                # jugar: "cualquier tema es mejor que seguir atascado en el
                # menu". Con el STT fallando eso sale caro -- el usuario
                # decia "no quiero jugar" tres veces y acababa en una tanda
                # de 5 preguntas de matematica. Si en dos intentos no se
                # entendio el tema, lo honesto es soltar la trivia y volver
                # a conversar: pedirla otra vez cuesta una frase.
                estado["intentos_tema"] = 0
                estado["esperando_tema"] = False
                self.get_logger().info(
                    "[trivia] no se entendió el tema en 2 intentos, "
                    "se vuelve al chat")
                salida = "Mejor seguimos charlando. Si quieres una trivia, pídemela."
                self.get_logger().info(f"Asistente: {salida}")
                self._bridge.hablar(salida)
                return
            estado["intentos_tema"] = 0
            estado["esperando_tema"] = False
            estado["tema_actual"] = categoria
            anuncio = (f"¡Vamos con {categoria}! Van "
                        f"{PREGUNTAS_POR_TANDA} preguntas seguidas.")
            self.get_logger().info(f"Asistente: {anuncio}")
            self._bridge.hablar(anuncio)
            self._iniciar_tanda(categoria)
            return

        pendiente = estado["pregunta_pendiente"]
        if pendiente is not None:
            estado["pregunta_pendiente"] = None
            self._bridge.mostrar_cara("speaking")
            if pendiente["respuesta_esperada"]:
                estado["total"] += 1
                # El LLM decide (SI/NO) y despues comenta, comparando lo que
                # dijo la persona con la respuesta esperada. Se le pasa
                # tambien la PREGUNTA: sin ella no puede juzgar "de cuero"
                # por "piel de animales". El comentario cambia segun el
                # animo de Romeo (aeb.tono_trivia): ya se aplico el afecto
                # de esta respuesta, asi que es el animo de ahora.
                acerto, comentario = agente_trivia.juzgar(
                    pendiente["pregunta"], pendiente["respuesta_esperada"],
                    mensaje_usuario, comportamiento=self._comportamiento())
                estado["aciertos"] += acerto
                cara = agente_trivia.elegir_cara_pregunta(pendiente, acerto) or (
                    "happy" if acerto else "sad")
                self.get_logger().info(f"Asistente: {comentario}")
                self._bridge.hablar(comentario)
                self._reaccionar_veredicto(cara, pendiente, estado.get("musica_ya_sonada", False))
            else:
                # Temas "sin veredicto" (Dilema del coche autónomo, etc.):
                # a pedido del usuario original, tampoco hay reacción
                # hablada -- se pasa directo a música/desplazamiento.
                time.sleep(PAUSA_ANTES_ACCION_FISICA)
                if not estado.get("musica_ya_sonada", False):
                    agente_trivia.expresar_musica(pendiente, self._bridge, logger=self.get_logger())
                # Sin movimiento de trivia: el unico que mueve es el AEB.

            # La cara afectiva (ROSE/AEB) se muestra AQUI, justo despues de
            # la respuesta del usuario y su veredicto -- no al presentar la
            # pregunta siguiente. Asi la cara acompaña el turno que acaba de
            # pasar, con el animo que dejo ESA respuesta, y no aparece
            # pegada al enunciado de la proxima pregunta (que el usuario aun
            # no contesto).
            self._mostrar_cara_reposo()

            if not self._preguntar_siguiente():
                self._cerrar_tanda()
            return

        # SE EMPIEZA A JUGAR YA, sin preguntar tema. Antes se recitaba el
        # menu de categorias y habia que elegir una: dos vueltas perdidas
        # antes de la primera pregunta, y con el STT fallando ("matematica"
        # llegaba como "Tematica") a veces mas. Las preguntas salen
        # mezcladas de cualquier familia -- preguntas.preguntas_mezcladas().
        #
        # El juego de caras se sigue pudiendo pedir en cualquier turno
        # (_quiere_jugar_emociones), solo que ya no se anuncia aqui.
        estado["esperando_tema"] = False
        estado["en_trivia"] = True
        # Se avisa de que empieza el juego y de como salir, ANTES de la
        # primera pregunta: sin esto Romeo pasaba de "quiero jugar trivia"
        # a soltar una pregunta de geografia sin mas, y no quedaba claro
        # que la trivia habia empezado ni como pararla.
        intro = frases.una(frases.INICIO_TRIVIA)
        self.get_logger().info(f"Asistente: {intro}")
        self._bridge.mostrar_cara("speaking")
        self._bridge.hablar(intro)
        self._iniciar_tanda(None)

    def _reanudar_trivia(self):
        estado = self.estado
        self._bridge.mostrar_cara("speaking")
        if estado["esperando_tema"]:
            self.get_logger().info("Asistente: Retomamos. ¿Qué tema eliges?")
            self._bridge.hablar("Retomamos. ¿Qué tema eliges?")
        else:
            pendiente = estado["pregunta_pendiente"]
            self.get_logger().info("Asistente: Retomamos donde quedamos.")
            self.get_logger().info(f"Asistente [{pendiente['cara']}]: {pendiente['pregunta']}")
            self._bridge.hablar(f"Retomamos donde quedamos. {pendiente['pregunta']}")
        time.sleep(PAUSA_CAMBIO_CARA)
        self._mostrar_cara_reposo()

    def _manejar_chat_libre(self, mensaje_usuario):
        # El AEB se consulta ANTES de responder: el animo ya incluye lo que
        # el TER leyo de este mensaje (ver _procesar_turno), asi que el tono
        # de la respuesta reacciona a lo que el usuario acaba de decir.
        with medir_bloque("aeb"):
            comp = self._comportamiento()
        # En la CONVERSACION la cara es solo el animo de Romeo: sin
        # 'speaking' y sin el destello de DURACION_EMOCION. Se pone antes
        # de responder y se queda durante todo el turno, asi la cara cuenta
        # como se siente mientras habla y mientras escucha. (En la trivia y
        # el juego de caras sigue el comportamiento de siempre.)
        if self._timer_emocion is not None:
            self._timer_emocion.cancel()
        self._emocion_pendiente = False
        self._bridge.mostrar_cara(comp["cara"])
        # tono_para: si le preguntan como esta, el tono lleva su animo.
        texto_final = self.responder(mensaje_usuario, tono=aeb.tono_para(comp, mensaje_usuario))
        self.get_logger().info(f"Asistente: {texto_final}")
        self._bridge.hablar(texto_final)
        # Mismo orden que el resto del robot: voz -> movimiento.
        if comp["movimiento"]:
            self.get_logger().info(f"[aeb] gesto: {comp['movimiento']}")
            self._ejecutar_gesto(comp["movimiento"])

    # Duracion de cada paso del gesto de ida y vuelta (ver _ejecutar_gesto).
    # El firmware del carrito para la rueda a los 0.5 s si no se repite el
    # comando (watchdog), asi que se reenvia el comando cada ~0.2 s mientras
    # dure el paso, y luego se manda el siguiente. Ajustable sin recompilar.
    GESTO_PASO_S = float(os.environ.get("ROMEO_AEB_PASO_S", "0.4"))
    GESTO_REFRESCO_S = 0.2

    def _ejecutar_gesto(self, gesto):
        """Ejecuta el gesto del AEB: una SECUENCIA de comandos (p.ej.
        ("F","B") = avanza y vuelve) con una pausa entre uno y otro, en un
        hilo aparte para NO bloquear el turno.

        Acepta tambien un comando suelto (str) por compatibilidad. Cada
        paso se mantiene GESTO_PASO_S segundos reenviando el comando cada
        GESTO_REFRESCO_S para burlar el watchdog del firmware, y al final
        se manda 'S' para dejar el carrito quieto."""
        pasos = (gesto,) if isinstance(gesto, str) else tuple(gesto)

        def _correr():
            for paso in pasos:
                fin = time.monotonic() + self.GESTO_PASO_S
                while time.monotonic() < fin:
                    self._bridge.mover(paso)
                    time.sleep(self.GESTO_REFRESCO_S)
            self._bridge.mover("S")

        threading.Thread(target=_correr, daemon=True).start()

    # ══════════════════════════════════════════════════════════════════
    # Arranque/cierre de sesión y el callback de entrada.
    # ══════════════════════════════════════════════════════════════════

    def _iniciar_sesion(self):
        # Sesion nueva, hilo nuevo: el usuario siguiente no hereda la
        # conversacion del anterior.
        agente_chat.reiniciar()
        self._afecto.reiniciar()
        # Y la ventana del TER: su mediana es de las frases de ESTA
        # conversacion, no de las del usuario anterior.
        estado_afectivo.olvidar_ter()
        self._bridge.mostrar_cara("content")
        time.sleep(PAUSA_CAMBIO_CARA)
        self._bridge.mostrar_cara("speaking")
        # Quien soy, que se hacer y el nombre, EN UNA SOLA FRASE. Antes eran
        # dos hablar() seguidos, y como cada uno bloquea hasta que termina de
        # sonar (sintetizar + mandar + reproducir), entre la presentacion y
        # la pregunta del nombre se oia un silencio de ~3.6 s: parecia que
        # Romeo se habia colgado. Al mandar las dos frases juntas, el TTS las
        # sintetiza de corrido y suenan como una sola intervencion.
        apertura = (f"{frases.una(frases.PRESENTACIONES)} "
                    f"{frases.una(frases.SALUDOS_APERTURA)}")
        self.get_logger().info(f"Asistente: {apertura}")
        self._bridge.hablar(apertura)
        # El chat tiene que saber que Romeo acaba de preguntar el nombre:
        # es el orquestador quien lo dice, no el agente. Sin este turno
        # sembrado, al oir "Adrian" a secas el modelo no ve un nombre y
        # contesta "¿Que tal tu?".
        agente_chat.sembrar_turno_robot(apertura)
        time.sleep(PAUSA_CAMBIO_CARA)
        self._mostrar_cara_reposo()

        self.estado = {
            "esperando_nombre": True,
            "pregunta_pendiente": None, "esperando_tema": False, "en_trivia": False,
            # Veces seguidas que no se entendio la categoria; a la
            # segunda se elige una en vez de seguir preguntando.
            "intentos_tema": 0,
            "cola_preguntas": [], "ya_usados": set(), "aciertos": 0, "total": 0,
            "tema_actual": None, "nombre": None, "musica_ya_sonada": False,
        }


    def _finalizar_sesion(self):
        estado = self.estado
        self._bridge.mostrar_cara("speaking")
        # Resumen y despedida van juntos, por lo mismo que la apertura: en
        # dos hablar() quedaba un silencio en medio de lo que es una sola
        # cosa que decir ("Terminamos... Hasta luego, Adrian").
        despedida = frases.una(frases.DESPEDIDAS, nombre=estado["nombre"])
        if estado["total"]:
            despedida = (f"Terminamos. Acertaste {estado['aciertos']} de "
                         f"{estado['total']}. {despedida}")
        self.get_logger().info(f"Asistente: {despedida}")
        self._bridge.hablar(despedida)
        self._bridge.mostrar_cara("happy")
        # Ver la nota del módulo: acá el original terminaba el proceso,
        # esto arranca una sesión nueva en su lugar.
        self._iniciar_sesion()

    def _on_user_input(self, msg):
        # Un solo turno a la vez -- el lock preserva el mismo invariante
        # que la cola de un solo consumidor (_entrada_queue) del proyecto
        # original, aunque la suscripción esté en un ReentrantCallbackGroup
        # (necesario para que las llamadas a servicio internas no
        # deadlockeen, ver el docstring del módulo).
        #
        # Si ya hay un turno en curso, la frase se DESCARTA en vez de
        # esperar al lock. Antes esperaba: cada frase que llegaba mientras
        # Romeo pensaba o hablaba ocupaba un hilo del executor, y con tres
        # en cola los cuatro hilos quedaban bloqueados -- no quedaba ninguno
        # para recibir la respuesta de /romeo/speak, cada frase esperaba
        # los 60 s del timeout y las respuestas llegaban desfasadas.
        if not self._turno_lock.acquire(blocking=False):
            self.get_logger().info(
                f"[turno] Romeo sigue con el turno anterior, se descarta: {msg.text!r}")
            return
        try:
            ser = None
            if msg.ser_etiquetas:
                ser = (list(msg.ser_etiquetas), list(msg.ser_probabilidades))
            ter_nn = None
            if msg.ter_etiquetas:
                ter_nn = (list(msg.ter_etiquetas), list(msg.ter_probabilidades))
            self._procesar_turno((msg.text or "").strip(),
                                 msg.t_inicio, msg.t_fin, ser, ter_nn)
        finally:
            self._turno_lock.release()

    def _procesar_turno(self, entrada, t_inicio=0.0, t_fin=0.0, ser=None,
                        ter_nn=None):
        if self.estado is None or not entrada:
            return
        # Con ROMEO_MEDIR=1, el turno entero se cronometra y al final se
        # escribe una fila. Sin la variable, turno() y _anotar_turno() no
        # hacen nada.
        with turno():
            self._turno_medido(entrada, t_inicio, t_fin, ser, ter_nn)
            self._anotar_turno(entrada, t_fin)

    def _anotar_turno(self, entrada, t_fin):
        """Una fila por turno con lo que tardo cada etapa.

        t_fin va tal cual para poder calcular la latencia L: el otro
        extremo (t_inicio_audio) lo escribe _voice_backends en su propio
        CSV, porque corre en communication_node y no aqui. El script de
        analisis cruza los dos por tiempo.

        Nunca lanza: medir no puede tumbar un turno."""
        if not medicion_activa():
            return
        try:
            fila = tiempos()
            fila["t_fin"] = round(t_fin, 6)
            fila["texto"] = (entrada or "").replace(",", " ")[:60]
            nuevo = not os.path.exists(CSV_TURNOS)
            with open(CSV_TURNOS, "a", encoding="utf-8") as f:
                if nuevo:
                    print(",".join(fila), file=f)
                print(",".join(str(v) for v in fila.values()), file=f)
        except Exception as e:  # noqa: BLE001 - medir nunca es critico
            self.get_logger().warning(f"[medicion] no se pudo anotar: {e}")

    def _anotar_veredicto(self, pedido, detectado, confianza, acerto):
        """Una fila por cara juzgada en el juego de emociones: exactitud del
        veredicto afectivo. 'sin_cara' cuando la camara no vio a nadie (no
        cuenta como acierto ni fallo). Nunca lanza."""
        if not medicion_activa():
            return
        try:
            if detectado is None:
                resultado, conf = "sin_cara", ""
            else:
                resultado = "acierto" if acerto else "fallo"
                conf = round(float(confianza), 3)
            fila = {
                "t": round(time.time(), 3),
                "pedido": pedido,
                "detectado": detectado if detectado is not None else "",
                "confianza": conf,
                "resultado": resultado,
            }
            nuevo = not os.path.exists(CSV_VEREDICTO)
            with open(CSV_VEREDICTO, "a", encoding="utf-8") as f:
                if nuevo:
                    print(",".join(fila), file=f)
                print(",".join(str(v) for v in fila.values()), file=f)
        except Exception as e:  # noqa: BLE001 - medir nunca es critico
            self.get_logger().warning(f"[medicion] veredicto no anotado: {e}")

    def _turno_medido(self, entrada, t_inicio, t_fin, ser, ter_nn):

        # Empieza un turno: ha pasado tiempo desde el anterior, asi que el
        # animo se acerca un paso a su base (ROSE). Sin esta llamada el
        # estado se quedaria clavado en lo ultimo que se percibio.
        # decaer() no hace nada hasta que pasa la latencia, asi que dos
        # turnos seguidos y rapidos no borran lo que se acaba de percibir.
        antes_decaer = self._afecto.vector
        self._afecto.decaer()
        self._traza = traza_afectiva.nueva(entrada, self._afecto, antes_decaer)

        # TER: que emocion expresa lo que acaba de decir el usuario.
        #
        # Va ACA y en ningun otro sitio. Este es el unico punto por el que
        # pasa todo mensaje (_on_user_input es el unico productor, y llama
        # solo a esta funcion), y esta ANTES de que el turno se bifurque a
        # trivia o chat libre: asi cubre las dos rutas sin tocar el router
        # ni el banco de preguntas. Como _procesar_turno se ejecuta una
        # sola vez por mensaje -- las transiciones trivia<->chat son ramas
        # de dentro, no reentradas -- ROSE no se puede actualizar dos veces
        # por turno.
        #
        # Las lecturas se GUARDAN y se aplican juntas con _aplicar_afecto()
        # (fusion, ec. 4.4): TER del texto, FER de la camara y SER de la
        # voz, las tres de la misma ventana -- la frase. ROSE se actualiza
        # una sola vez por turno. La camara del juego de caras NO cuenta:
        # ver _pedir_una_emocion().
        self._lecturas_turno = []
        self._emocion_pendiente = False
        if "TER" in self._fuentes and ter_nn:
            # El TER lo calcula RoBERTuito en la PC (ver _ter_engine.py) y
            # llega con la frase. Si no llego -- entrada por teclado, o
            # servidor sin el modelo -- este turno no hay TER, igual que el
            # FER o el SER cuando no hay fuente. No hay respaldo local.
            #
            # Paso 0 del TER (tesis, sec. 4.1.4.1): esta frase entra en la
            # ventana y PULSE recibe la mediana normalizada de las ultimas
            # VENTANA_TER, no solo la de ahora.
            estado_afectivo.acumular_ter(ter_nn[1], ter_nn[0])
            etiq_ter, probs_ter = estado_afectivo.resumir_ter()
            vector_txt, c_txt, emocion_txt = estado_afectivo.percibir_categorias(
                probs_ter, etiq_ter)
            if vector_txt is not None:
                self._lecturas_turno.append(("TER", vector_txt, c_txt, emocion_txt))
                self._trazar_lectura(
                    "TER", probs_ter, etiq_ter,
                    ventana=estado_afectivo.tam_ventana_ter(),
                    frase=dict(zip(ter_nn[0], (round(float(p), 4) for p in ter_nn[1]))))
                modo_ter = ("frase directa" if estado_afectivo.VENTANA_TER <= 1
                            else f"ventana de {estado_afectivo.VENTANA_TER}")
                self.get_logger().info(
                    f"[afecto] el texto expresa {emocion_txt} "
                    f"(confianza {c_txt:.2f}, RoBERTuito, {modo_ter})")
        if "FER" in self._fuentes:
            self._percibir_cara(t_inicio, t_fin)
        if "SER" in self._fuentes:
            self._percibir_voz(ser)
        # PULSE (fusion de las fuentes) + ROSE (animo del robot). El AEB
        # se mide aparte, en _comportamiento(): son dos etapas distintas
        # de la capa afectiva y conviene saber cual pesa.
        with medir_bloque("pulse_rose"):
            self._aplicar_afecto()
        # La traza se escribe YA, antes de la respuesta del LLM: la pagina
        # /calculo muestra el calculo apenas termina, sin esperar al chat.
        if self._traza is not None:
            self._traza["aeb"] = traza_afectiva.comportamiento(self._afecto)
            traza_afectiva.escribir(self._traza)
            self._traza = None

        if self.estado["esperando_nombre"]:
            nombre = entrada or "amigo"
            self.estado["nombre"] = nombre
            self.estado["esperando_nombre"] = False
            # Saluda el chat, no una frase fija: asi el nombre entra en su
            # historial y lo puede usar despues. La pregunta que lo provoco
            # ya esta sembrada en _iniciar_sesion(), asi que el modelo ve
            # "Adrian" como la respuesta que es.
            #
            # Si el chat no contesta (servidor caido), queda la frase de
            # bienvenidas.txt: sin esto el usuario se quedaria sin saludo.
            # self.responder() y no agente_chat.responder(): el envoltorio
            # atrapa el fallo si el servidor del LLM no esta y devuelve una
            # frase de cortesia en vez de tumbar el turno.
            bienvenida = self.responder(entrada) if entrada else ""
            if not bienvenida:
                bienvenida = frases.una(frases.BIENVENIDAS, nombre=nombre)
            self.get_logger().info(f"Asistente: {bienvenida}")
            self._bridge.hablar(bienvenida)
            self._mostrar_cara_reposo()
            return

        if entrada.lower() in ("salir", "exit", "quit"):
            self._finalizar_sesion()
            return

        # "¿Que haces?" se contesta con una frase fija, ANTES de llegar al
        # chat: el saludo ya no recita el menu, asi que esta es la unica
        # forma de enterarse. Y tiene que ser frase fija porque el modelo
        # de chat esta afinado en EmpatheticDialogues y a "¿que haces?"
        # responde "estoy para acompañarte y conversar", sin nombrar ni la
        # trivia ni las caras aunque el prompt se las liste.
        # Ordenes directas ("gira 360", "ponte triste") ANTES del chat: el
        # LLM no mueve el carrito ni cambia la cara, y afinado en
        # EmpatheticDialogues a "gira" contesta algo empatico. Ver
        # ordenes.py para por que lo decide una tabla y no el modelo.
        orden = ordenes.interpretar(entrada)
        if orden is not None:
            self._obedecer(orden)
            return

        if self._pregunta_capacidades(entrada):
            respuesta = frases.una(frases.CAPACIDADES)
            self.get_logger().info(f"Asistente: {respuesta}")
            self._bridge.mostrar_cara("speaking")
            self._bridge.hablar(respuesta)
            self._mostrar_cara_reposo()
            return

        if self.estado["en_trivia"]:
            pendiente = self.estado["pregunta_pendiente"]
            if self._quiere_salir_trivia(entrada, pendiente["pregunta"] if pendiente else None):
                self.estado["en_trivia"] = False
                self._mostrar_cara_reposo()
                # El marcador se dice AQUI porque ya no hay tandas de 5 que
                # lo digan al cerrar: ahora las preguntas se encadenan hasta
                # que el usuario pide salir, y este es el unico momento en
                # que sabe como le fue.
                if estado["total"]:
                    marcador = (f"Acertaste {estado['aciertos']} "
                                f"de {estado['total']}.")
                    self.get_logger().info(f"Asistente: {marcador}")
                    self._bridge.hablar(marcador)
                # El mensaje con el que pide salir se atiende EN ESTE MISMO
                # turno: antes solo se cambiaba de modo y se logueaba un aviso
                # que nadie decia en voz alta, asi que el usuario escuchaba
                # silencio y tenia que repetir lo que acababa de decir.
                #
                # Si ademas de pedir salir dijo algo de fondo ("ya no quiero
                # jugar, me siento triste"), eso va al chat libre. Si solo
                # pidio salir ("basta"), basta con confirmarlo.
                if len(entrada.split()) > _PALABRAS_RESPUESTA_CORTA:
                    self._manejar_chat_libre(entrada)
                else:
                    cierre = ("Listo, dejamos la trivia. "
                              "¿De qué quieres que hablemos?")
                    self.get_logger().info(f"Asistente: {cierre}")
                    self._bridge.mostrar_cara("speaking")
                    self._bridge.hablar(cierre)
                    self._mostrar_cara_reposo()
                return
            self._manejar_trivia(entrada)
            return

        # El juego de caras se pide con sus propias palabras ("emociones",
        # "caras", "gestos"). El router las clasifica como TRIVIA porque son
        # un juego, así que hay que mirarlo ANTES de entrar en la trivia de
        # preguntas -- si no, pediría categoría y nunca llegaría a la cámara.
        if self._quiere_jugar_emociones(entrada):
            self._jugar_emociones()
            return

        ruta = self._enrutar_mensaje(entrada)
        if ruta == "TRIVIA":
            self.estado["en_trivia"] = True
            if self.estado["pregunta_pendiente"] is not None or self.estado["esperando_tema"]:
                self._reanudar_trivia()
            else:
                self._manejar_trivia(entrada)
        else:
            self._manejar_chat_libre(entrada)


def main(args=None):
    rclpy.init(args=args)
    node = OrchestratorNode()
    # >= 2 hilos es obligatorio -- ver la nota en
    # RosBridge._llamar_servicio_sync() sobre por qué un SingleThreadedExecutor
    # deadlockearía la primera vez que el turno llama a un servicio.
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
