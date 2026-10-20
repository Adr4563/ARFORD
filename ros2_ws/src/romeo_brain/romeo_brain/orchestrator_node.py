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

Este nodo no declara ningún parámetro propio: todo lo de la voz lo resuelve
communication_node. Antes acá se declaraba un `voz_motor` duplicado solo
para saber si valía la pena esperar a que se conectara un teléfono antes
del saludo de apertura -- eso desapareció junto con la página web (y el
motor de voz por teléfono), así que la sesión arranca de una.

Cambio de comportamiento deliberado respecto del original: cuando el
usuario dice "salir", el proceso original terminaba (era una app de
terminal para probar). Acá eso solo termina la SESIÓN -- el robot dice la
despedida y arranca una sesión nueva (_iniciar_sesion()), listo para el
próximo usuario. Ver _finalizar_sesion() más abajo.
"""
import os
import random
import threading
import time

import joblib
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory

from romeo_interfaces.msg import UserInput

from . import (agente_chat, agente_trivia, estado_afectivo, frases,
               llama_client, preguntas, ter)
from ._ros_bridge import RosBridge

# El enrutado (TRIVIA / CHAT_LIBRE) vive aca y no en un agente aparte: son
# cuatro lineas de scikit-learn sobre un joblib ya entrenado, sin estado ni
# hardware. Tenerlo en su propio modulo solo añadia un archivo que leer.
_RUTA_MODELO = os.path.join(
    get_package_share_directory('romeo_brain'), 'data', 'router_modelo.joblib')
_router = joblib.load(_RUTA_MODELO)


def enrutar(mensaje_usuario):
    """Clasifica el mensaje en TRIVIA o CHAT_LIBRE (TF-IDF de n-gramas de
    caracteres + regresion logistica, ~95-96% medido)."""
    vector = _router["vectorizador"].transform([mensaje_usuario])
    return _router["clasificador"].predict(vector)[0]

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


# Saludo de apertura de cada sesión -- se elige uno al azar, así que dos
# niños seguidos no oyen lo mismo. Todas tutean (no vosean): el fine-tune
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


# Lo que dice después de que el niño da su nombre: presenta las dos cosas
# que sabe hacer (trivia y charlar) y le deja elegir. Antes era UNA sola
# frase fija; ahora son 5 variantes al azar.

PREGUNTAS_POR_TANDA = 5
PAUSA_CAMBIO_CARA = 1       # segundos en 'content'/veredicto antes de la próxima cara
PAUSA_ANTES_ACCION_FISICA = 1
# Segundos entre pedirle la cara y capturarla. El detector ya tarda 2.5-4.6s
# en resolver (medido en la Pi), así que esto es solo el margen para que
# termine de oír la frase y ponga la cara.
PAUSA_POSAR = 1

# Frases que, a mitad de una tanda de trivia, señalan que el usuario se
# quiere ir a otra cosa -- fallback de _quiere_salir_trivia() cuando no se
# puede mirar las listas de frases de arriba (ver ese método).
_PALABRAS_RESPUESTA_CORTA = 4

RUTAS = ["TRIVIA", "CHAT_LIBRE"]

# Cada cuanto se reintenta el arranque mientras communication_node no pueda
# hablar, y cuanto se espera como maximo (ver _arrancar). En la Pi tarda ~11 s
# en activarse: carga la voz VITS y el modelo de STT.
ESPERA_SPEAK = 1.0
MAX_ESPERA_SPEAK = 45.0




class OrchestratorNode(Node):

    def __init__(self):
        super().__init__('orchestrator_node')

        self._bridge = RosBridge(self)
        # El animo de Romeo, que persiste entre turnos y decae solo. Vive
        # en el nodo y no en self.estado porque self.estado se reconstruye
        # entero en cada sesion, y lo que se quiere del afecto es
        # justamente que dure. Se reinicia a mano en _iniciar_sesion().
        self._afecto = estado_afectivo.EstadoAfectivo()
        # Lecturas afectivas del turno en curso, pendientes de
        # fusionar. Ver _aplicar_afecto().
        self._lecturas_turno = []
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

    def responder(self, mensaje_usuario):
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
            return agente_chat.responder(mensaje_usuario)
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
        hablan el texto y la camara, el animo se mueve una vez con la
        media ponderada de las dos y no dos veces seguidas.

        fusionar() se usa tal cual: ya filtra las lecturas nulas o sin
        confianza y, con una sola, devuelve esa misma."""
        if not getattr(self, "_lecturas_turno", None):
            return
        vector = estado_afectivo.fusionar(self._lecturas_turno)
        # La confianza con la que se cree la lectura fusionada es la mayor
        # de las que entraron: si la camara estaba segura, que el texto no
        # lo estuviera no debe frenar la reaccion.
        conf = max((c for _s, c in self._lecturas_turno), default=0.0)
        self._lecturas_turno = []
        if vector is None:
            return
        antes = self._afecto.vector
        self._afecto.empatizar(vector, conf)
        v, a, d = self._afecto.vector
        self.get_logger().info(
            f"[afecto] v {antes[0]:+.2f}->{v:+.2f}  "
            f"a {antes[1]:+.2f}->{a:+.2f}  d {antes[2]:+.2f}->{d:+.2f}")

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
        hay algo que mostrar."""
        cara = self._afecto.cara()
        # Se registra el vector, no solo la cara: la cara es una de cuatro
        # y se queda igual durante varios turnos, asi que sin el vector no
        # se ve si el animo se esta moviendo o esta clavado.
        v, a, d = self._afecto.vector
        self.get_logger().info(
            f"[afecto] v={v:+.2f} a={a:+.2f} d={d:+.2f} -> cara {cara}")
        return cara

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
        agente_trivia.expresar_desplazamiento(pregunta, self._bridge, logger=self.get_logger())
        time.sleep(PAUSA_CAMBIO_CARA)

    def _preguntar_siguiente(self):
        estado = self.estado
        if not estado["cola_preguntas"]:
            return False
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
        self._bridge.mostrar_cara(self._cara_reposo())
        return True

    def _cerrar_tanda(self):
        estado = self.estado
        estado["en_trivia"] = False
        self._bridge.mostrar_cara("speaking")
        cierre = "Esas eran las 5."
        if estado["total"]:
            cierre += f" Vas {estado['aciertos']} de {estado['total']}."
        cierre += " ¿Seguimos con más trivia o prefieres charlar?"
        self.get_logger().info(f"Asistente: {cierre}")
        self._bridge.hablar(cierre)
        time.sleep(PAUSA_CAMBIO_CARA)
        self._bridge.mostrar_cara(self._cara_reposo())

    def _pedir_una_emocion(self, objetivo):
        """Pide UNA cara, la captura y la juzga.

        Devuelve True/False según acierte, o None si la cámara no vio
        ninguna cara (eso no cuenta para el marcador: no es culpa del
        usuario)."""
        # Romeo pone primero la cara que pide, para que el niño la imite.
        self._bridge.mostrar_cara(agente_trivia.cara_para_emocion(objetivo) or "content")
        pedido = frases.una(frases.PEDIDOS_EMOCION[objetivo])
        self.get_logger().info(f"Asistente: {pedido}")
        self._bridge.hablar(pedido)
        time.sleep(PAUSA_POSAR)   # que le dé tiempo a poner la cara

        detectada, confianza, ok = self._bridge.detectar_emocion()
        # Lo que ve la camara alimenta el animo de Romeo (PULSE -> ROSE).
        # Se hace aca y no en el bridge porque el bridge no debe saber de
        # logica afectiva, solo de topicos y servicios.
        if ok:
            # Se acumula igual que el TER y se aplica con _aplicar_afecto,
            # para que un turno que percibe por las dos vias mueva el animo
            # UNA vez con la media de ambas, no dos veces seguidas.
            vector, c = estado_afectivo.percibir_del_servicio(detectada, confianza)
            if vector is not None:
                self._lecturas_turno = getattr(self, "_lecturas_turno", [])
                self._lecturas_turno.append((vector, c))
                self.get_logger().info(
                    f"[afecto] la camara vio {detectada} ({confianza:.0%})")
                self._aplicar_afecto()
        self._bridge.mostrar_cara("speaking")

        if not ok or detectada is None:
            aviso = frases.una(frases.SIN_CARA)
            self.get_logger().info(f"[cámara] sin cara -- {aviso}")
            self._bridge.hablar(aviso)
            return None

        acerto = detectada == objetivo
        self.get_logger().info(
            f"[cámara] pedido={objetivo} detectado={detectada} "
            f"(confianza {confianza:.0%}) -> {'OK' if acerto else 'MAL'}")
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

    @staticmethod
    def _quiere_jugar_emociones(mensaje):
        """True si el usuario pide el juego de caras.

        Normaliza quitando tildes: el STT transcribe "emoción" e
        "imitación" acentuadas, y _PIDE_EMOCIONES las lleva sin tilde."""
        return any(p in preguntas._sin_tildes(mensaje) for p in _PIDE_EMOCIONES)

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
        self._bridge.mostrar_cara(self._cara_reposo())

    def _iniciar_tanda(self, categoria):
        estado = self.estado
        preguntas_tanda = preguntas.preguntas_por_categoria(
            categoria, estado["ya_usados"], cantidad=PREGUNTAS_POR_TANDA)
        if not preguntas_tanda:
            aviso = f"Ya jugamos todas las de {categoria}. ¿Elegimos otra?"
            self.get_logger().info(f"Asistente: {aviso}")
            self._bridge.hablar(aviso)
            estado["esperando_tema"] = True   # sigue en trivia, pidiendo categoría
            return
        estado["ya_usados"].update(p["id"] for p in preguntas_tanda)
        estado["cola_preguntas"] = preguntas_tanda
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
                categoria = random.choice(cats)
                self.get_logger().info(
                    f"[trivia] no se entendió el tema en "
                    f"{estado['intentos_tema']} intentos, se elige "
                    f"{categoria}")
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
                acerto = agente_trivia.evaluar_respuesta(pendiente["respuesta_esperada"], mensaje_usuario)
                estado["aciertos"] += acerto
                cara = agente_trivia.elegir_cara_pregunta(pendiente, acerto) or (
                    "happy" if acerto else "sad")
                # Primero se le pide al LLM un comentario sobre lo que dijo
                # el usuario; si no puede, se usan las frases fijas.
                comentario = agente_trivia.comentar(
                    pendiente["pregunta"], pendiente["respuesta_esperada"],
                    mensaje_usuario, acerto)
                if not comentario:
                    comentario = self.comentar_resultado(
                        pendiente["respuesta_esperada"], acerto)
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
                agente_trivia.expresar_desplazamiento(pendiente, self._bridge, logger=self.get_logger())

            if not self._preguntar_siguiente():
                self._cerrar_tanda()
            return

        # Se le ofrecen las 5 CATEGORÍAS, siempre las mismas y en el mismo
        # orden. Antes se recitaban 5 de los 49 "<familia> - Nivel N" al
        # azar, que en voz alta era imposible de seguir ("Geografía Nivel 7,
        # Razonamiento matemático Nivel 5, ...") y además cambiaba en cada
        # turno. Los niveles siguen usándose para elegir las preguntas, solo
        # que ya no se mencionan -- ver preguntas.CATEGORIAS.
        cats = preguntas.categorias_disponibles()
        estado["esperando_tema"] = True
        # Se menciona también el juego de caras: si no, el usuario no tiene
        # cómo enterarse de que existe. Decirlo acá basta, porque
        # _quiere_jugar_emociones() lo detecta en cualquier turno.
        anuncio = (f"¿Qué quieres jugar? Tengo trivias de "
                    f"{', '.join(cats[:-1])} o {cats[-1]}, "
                    f"y también el juego de las caras.")
        self.get_logger().info(f"Asistente: {anuncio}")
        self._bridge.hablar(anuncio)

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
        self._bridge.mostrar_cara(self._cara_reposo())

    def _manejar_chat_libre(self, mensaje_usuario):
        self._bridge.mostrar_cara("speaking")
        texto_final = self.responder(mensaje_usuario)
        self.get_logger().info(f"Asistente: {texto_final}")
        self._bridge.hablar(texto_final)
        time.sleep(PAUSA_CAMBIO_CARA)
        self._bridge.mostrar_cara(self._cara_reposo())

    # ══════════════════════════════════════════════════════════════════
    # Arranque/cierre de sesión y el callback de entrada.
    # ══════════════════════════════════════════════════════════════════

    def _iniciar_sesion(self):
        # Sesion nueva, hilo nuevo: el usuario siguiente no hereda la
        # conversacion del anterior.
        agente_chat.reiniciar()
        self._afecto.reiniciar()
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
        time.sleep(PAUSA_CAMBIO_CARA)
        self._bridge.mostrar_cara(self._cara_reposo())

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
        with self._turno_lock:
            self._procesar_turno((msg.text or "").strip())

    def _procesar_turno(self, entrada):
        if self.estado is None or not entrada:
            return

        # Empieza un turno: ha pasado tiempo desde el anterior, asi que el
        # animo se acerca un paso a su base (ROSE). Sin esta llamada el
        # estado se quedaria clavado en lo ultimo que se percibio.
        # decaer() no hace nada hasta que pasa la latencia, asi que dos
        # turnos seguidos y rapidos no borran lo que se acaba de percibir.
        self._afecto.decaer()

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
        # La lectura se GUARDA en vez de aplicarse: si mas adelante en este
        # mismo turno la camara tambien percibe algo (el juego de
        # emociones), las dos se fusionan y ROSE se actualiza una sola vez.
        # Ver _aplicar_afecto().
        emocion_txt, confianza_txt = ter.analizar(entrada)
        self._lecturas_turno = []
        vector_txt, c_txt = estado_afectivo.percibir_del_texto(
            emocion_txt, confianza_txt)
        if vector_txt is not None:
            self._lecturas_turno.append((vector_txt, c_txt))
            self.get_logger().info(
                f"[afecto] el texto expresa {emocion_txt} "
                f"(confianza {confianza_txt:.2f})")
        self._aplicar_afecto()

        if self.estado["esperando_nombre"]:
            nombre = entrada or "amigo"
            self.estado["nombre"] = nombre
            self.estado["esperando_nombre"] = False
            bienvenida = frases.una(frases.BIENVENIDAS, nombre=nombre)
            self.get_logger().info(f"Asistente: {bienvenida}")
            self._bridge.hablar(bienvenida)
            self._bridge.mostrar_cara(self._cara_reposo())
            return

        if entrada.lower() in ("salir", "exit", "quit"):
            self._finalizar_sesion()
            return

        if self.estado["en_trivia"]:
            pendiente = self.estado["pregunta_pendiente"]
            if self._quiere_salir_trivia(entrada, pendiente["pregunta"] if pendiente else None):
                self.estado["en_trivia"] = False
                self._bridge.mostrar_cara(self._cara_reposo())
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
                    self._bridge.mostrar_cara(self._cara_reposo())
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
