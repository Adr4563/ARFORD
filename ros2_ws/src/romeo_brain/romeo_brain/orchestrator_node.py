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
import random
import threading
import time

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from romeo_interfaces.msg import UserInput

from . import agent_behavior, llama_client, memoria_episodica, perf_monitor, preguntas, registro_chat
from .agent_corrector import evaluar_respuesta
from .agent_router import enrutar
from ._ros_bridge import RosBridge

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
PEDIDOS_EMOCION = {
    "Feliz": ["¡Hazme una cara de felicidad!",
               "A ver esa sonrisa, ¡ponte feliz!"],
    "Triste": ["Ahora una cara triste.",
                "Pon cara de tristeza."],
    "Enojado": ["Ahora una cara de enojado.",
                 "Ponte bravo, cara de enojado."],
    "Neutral": ["Y por último, una cara neutra.",
                 "Ahora relaja la cara, cara neutra."],
}

# Lo que dice cuando la cámara no encuentra ninguna cara. Es frecuente (se
# midió: ~1 de cada 6 intentos) si el usuario se mueve o queda fuera del
# encuadre, así que tiene que sonar natural y no a error.
SIN_CARA = [
    "No te vi bien, ¿te acercas un poco?",
    "No alcancé a verte. Ponte frente a mí.",
    "Se me escapó esa. Mírame de frente.",
]

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

RESPUESTAS_TRIVIA_ACIERTO = [
    "¡Exacto, {esperada}!", "Sí, correcto.", "Así es, nada mal.", "Correcto, bien ahí.",
]
RESPUESTAS_TRIVIA_ERROR = [
    "No, era {esperada}.", "Nop, la correcta es {esperada}.",
    "Incorrecto, la respuesta era {esperada}.", "No es así, correcta: {esperada}.",
]

# Saludo de apertura de cada sesión -- se elige uno al azar, así que dos
# niños seguidos no oyen lo mismo. Todas tutean (no vosean): el fine-tune
# de Chat libre que usa el robot está entrenado en tuteo, y mezclar las dos
# formas en la misma conversación suena raro.
# La presentacion va SEPARADA del saludo: primero Romeo dice quien es y que
# sabe hacer, despues pregunta el nombre. Se dice en dos frases y no en una
# sola larga porque la voz tarda y una frase de 30 palabras se hace eterna.
PRESENTACION = (
    "Hola, soy Romeo, un robot. Puedo jugar trivia contigo, hacer caras y "
    "emociones, o simplemente conversar de lo que quieras."
)

SALUDOS_APERTURA = [
    # Ya no se presentan ("soy Romeo"): eso lo dice PRESENTACION justo antes,
    # y repetirlo sonaba raro. Aca solo se pide el nombre.
    "¿Y tú, cómo te llamas?",
    "Cuéntame, ¿cuál es tu nombre?",
    "¿Me dices tu nombre?",
    "Antes de empezar, ¿cómo te llamas?",
    "¿Quién eres tú?",
]

# Lo que dice después de que el niño da su nombre: presenta las dos cosas
# que sabe hacer (trivia y charlar) y le deja elegir. Antes era UNA sola
# frase fija; ahora son 5 variantes al azar.
BIENVENIDAS = [
    "Mucho gusto, {nombre}. Podemos jugar trivia o simplemente charlar, "
    "¿qué prefieres?",
    "¡Qué gusto, {nombre}! ¿Jugamos una trivia o prefieres que conversemos?",
    "Encantado, {nombre}. Tengo trivias de varios temas, o si quieres "
    "charlamos un rato. ¿Qué te gusta más?",
    "¡Hola {nombre}! Dime qué quieres hacer: una trivia, o conversar.",
    "Un placer, {nombre}. Podemos competir en una trivia o hablar de lo "
    "que quieras. Tú eliges.",
]
DESPEDIDAS = [
    "¡Hasta luego, {nombre}! Que te vaya bien.",
    "Nos vemos, {nombre}. ¡Fue un gusto!",
    "Listo por hoy, {nombre}. ¡Cuidate!",
    "¡Chau, {nombre}! Volvé cuando quieras.",
]

PREGUNTAS_POR_TANDA = 5
PAUSA_CAMBIO_CARA = 4       # segundos en 'content'/veredicto antes de la próxima cara
PAUSA_ANTES_ACCION_FISICA = 1
# Segundos entre pedirle la cara y capturarla. El detector ya tarda 2.5-4.6s
# en resolver (medido en la Pi), así que esto es solo el margen para que
# termine de oír la frase y ponga la cara.
PAUSA_POSAR = 2

# Frases que, a mitad de una tanda de trivia, señalan que el usuario se
# quiere ir a otra cosa -- fallback de _quiere_salir_trivia() cuando no se
# puede consultar SALIDA_TRIVIA_MODEL (ver ese método).
_SALIR_TRIVIA = [
    "otra cosa", "cambiar de tema", "cambiemos de tema",
    "ya no quiero seguir", "ya no quiero jugar", "no quiero seguir jugando",
    "quiero parar", "para de preguntar", "deja de preguntar", "detente",
    "salir de la trivia", "salir del juego", "salir de trivia",
    "pausa la trivia", "pausar la trivia", "pausemos",
    "dejemos la trivia", "dejar la trivia",
    "hablemos de otra cosa", "quiero charlar", "prefiero charlar", "quiero conversar",
    "basta", "ya basta", "no quiero jugar más", "no quiero jugar mas",
    "no quiero más preguntas", "no quiero mas preguntas",
    "olvida la trivia", "olvida el trivia", "no quiero trivia",
    "no más trivia", "no mas trivia",
    "cansado de las preguntas", "cansada de las preguntas",
    "quiero hacer otra cosa", "vamos a otra cosa",
    "suficiente trivia", "ya fue suficiente",
    "terminemos la trivia", "terminemos el trivia",
]
_TEMA_PERSONAL = [
    "me paso", "me pasó", "el otro dia", "el otro día",
    "cuando era", "recuerdo cuando", "me acuerdo cuando",
    "extraño a", "extraño mi", "mi profesor", "mi profesora",
    "mi maestro", "mi maestra", "mi mama", "mi mamá", "mi papa", "mi papá",
    "mi abuela", "mi abuelo", "mi amigo", "mi amiga",
    "problema con", "tuve un problema", "tengo un problema",
    "me regaño", "me regañó", "se pelearon", "se estan peleando",
    "se están peleando", "estoy triste", "estoy preocupado",
    "estoy preocupada", "me siento mal", "quiero contarte",
    "te quiero contar", "queria contarte", "quería contarte",
]
_PALABRAS_RESPUESTA_CORTA = 4

RUTAS = ["TRIVIA", "CHAT_LIBRE"]




class OrchestratorNode(Node):

    def __init__(self):
        super().__init__('orchestrator_node')

        self._bridge = RosBridge(self)
        self.estado = None
        self._turno_lock = threading.Lock()
        self._grupo_turno = ReentrantCallbackGroup()

        self.create_subscription(
            UserInput, '/romeo/user_input', self._on_user_input, 10,
            callback_group=self._grupo_turno)

        perf_monitor.iniciar_muestreo_recursos(logger=self.get_logger())
        self._precargar_modelos()

        # Arranca la primera sesión un instante después de construirse el
        # nodo -- no directo en __init__ porque _iniciar_sesion() hace
        # llamadas de servicio BLOQUEANTES (Speak), y esas solo se resuelven
        # una vez que el executor está spinneando (ver la nota de
        # RosBridge._llamar_servicio_sync). El timer se cancela apenas
        # dispara una vez.
        self._timer_arranque = self.create_timer(
            0.5, self._arrancar, callback_group=self._grupo_turno)

    def _arrancar(self):
        self._timer_arranque.cancel()
        self._iniciar_sesion()


    # ══════════════════════════════════════════════════════════════════
    # Warmup de Ollama -- portado de _precargar_modelos()/_precargar_uno()/
    # _precargar_salida_trivia() originales. No son llamadas a servicio
    # ROS2 (Ollama es HTTP directo, ver llama_client.py), así que no tienen
    # el problema de deadlock de arriba -- se pueden lanzar ya en __init__.
    # ══════════════════════════════════════════════════════════════════

    def _precargar_uno(self, modelo):
        try:
            llama_client.generar_respuesta(
                [{"role": "user", "content": "hola"}], max_tokens=1, modelo=modelo)
        except Exception as e:
            self.get_logger().warning(f"[warmup] no se pudo precargar {modelo}: {e}")

    def _precargar_salida_trivia(self):
        try:
            llama_client.clasificar_salida_trivia("pregunta de prueba", "respuesta de prueba")
        except Exception as e:
            self.get_logger().warning(
                f"[warmup] no se pudo precargar {llama_client.SALIDA_TRIVIA_MODEL}: {e}")

    def _precargar_modelos(self):
        threading.Thread(target=self._precargar_uno, args=(llama_client.CHAT_MODEL,), daemon=True).start()
        threading.Thread(target=self._precargar_uno, args=(llama_client.TRIVIA_MODEL,), daemon=True).start()
        threading.Thread(target=self._precargar_salida_trivia, daemon=True).start()

    # ══════════════════════════════════════════════════════════════════
    # Helpers de Chat libre / prompts -- portados tal cual de
    # Orchestrator_Management.py.
    # ══════════════════════════════════════════════════════════════════

    def responder(self, mensaje_usuario):
        """Chat libre: memoria episódica BM25 + CHAT_MODEL, sin RAG (ver la
        nota larga del original sobre por qué se sacó el RAG viejo) y sin
        system prompt (el comportamiento sale del fine-tuning)."""
        recuerdo = memoria_episodica.buscar_relevante(mensaje_usuario)
        contenido = f"{recuerdo}\n{mensaje_usuario}" if recuerdo else mensaje_usuario
        mensajes = [{"role": "user", "content": contenido}]
        texto_final = llama_client.generar_respuesta(mensajes).strip()
        registro_chat.registrar(mensaje_usuario, texto_final, llama_client.CHAT_MODEL)
        return texto_final

    # ══════════════════════════════════════════════════════════════════
    # Trivia: reacciones fijas, resolución de tema, salida a mitad de
    # pregunta -- portados tal cual.
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def comentar_resultado(esperada, acerto):
        plantilla = random.choice(RESPUESTAS_TRIVIA_ACIERTO if acerto else RESPUESTAS_TRIVIA_ERROR)
        return plantilla.format(esperada=esperada)

    @staticmethod
    def _enrutar_mensaje(mensaje_usuario):
        ruta = enrutar(mensaje_usuario)
        return ruta if ruta in RUTAS else "CHAT_LIBRE"

    def _quiere_salir_trivia(self, mensaje_usuario, pregunta_pendiente=None):
        texto = mensaje_usuario.strip().lower()
        tiene_keyword = (any(frase in texto for frase in _SALIR_TRIVIA)
                          or any(frase in texto for frase in _TEMA_PERSONAL))
        es_corto_y_sin_keyword = len(texto.split()) <= _PALABRAS_RESPUESTA_CORTA and not tiene_keyword
        if pregunta_pendiente and not es_corto_y_sin_keyword:
            veredicto = llama_client.clasificar_salida_trivia(pregunta_pendiente, mensaje_usuario)
            if veredicto is not None:
                return veredicto
        return tiene_keyword

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
            agent_behavior.expresar_musica(pregunta, self._bridge, logger=self.get_logger())
        agent_behavior.expresar_desplazamiento(pregunta, self._bridge, logger=self.get_logger())
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
        estado["musica_ya_sonada"] = bool(agent_behavior.expresar_musica(
            actual, self._bridge, esperar=True, logger=self.get_logger()))
        if es_musical:
            self._bridge.mostrar_cara("speaking")

        time.sleep(PAUSA_CAMBIO_CARA)
        self._bridge.mostrar_cara("content")
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
        self._bridge.mostrar_cara("content")

    def _pedir_una_emocion(self, objetivo):
        """Pide UNA cara, la captura y la juzga.

        Devuelve True/False según acierte, o None si la cámara no vio
        ninguna cara (eso no cuenta para el marcador: no es culpa del
        usuario)."""
        # Romeo pone primero la cara que pide, para que el niño la imite.
        self._bridge.mostrar_cara(agent_behavior.cara_para_emocion(objetivo) or "content")
        pedido = random.choice(PEDIDOS_EMOCION[objetivo])
        self.get_logger().info(f"Asistente: {pedido}")
        self._bridge.hablar(pedido)
        time.sleep(PAUSA_POSAR)   # que le dé tiempo a poner la cara

        detectada, confianza, ok = self._bridge.detectar_emocion()
        self._bridge.mostrar_cara("speaking")

        if not ok or detectada is None:
            aviso = random.choice(SIN_CARA)
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
        self._bridge.mostrar_cara("content")

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
                # No se entendió la categoría: se repregunta en vez de
                # elegir una al azar, que es lo que hacía resolver_tema() --
                # con 5 opciones claras, acertar la que quiso el usuario
                # importa más que no insistir.
                cats = preguntas.categorias_disponibles()
                repregunta = (f"No te entendí. Elige una: "
                               f"{', '.join(cats[:-1])} o {cats[-1]}. "
                               f"O dime 'caras' para el juego de emociones.")
                self.get_logger().info(f"Asistente: {repregunta}")
                self._bridge.hablar(repregunta)
                return
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
                acerto = evaluar_respuesta(pendiente["respuesta_esperada"], mensaje_usuario)
                estado["aciertos"] += acerto
                cara = agent_behavior.elegir_cara_pregunta(pendiente, acerto) or (
                    "happy" if acerto else "sad")
                comentario = self.comentar_resultado(pendiente["respuesta_esperada"], acerto)
                self.get_logger().info(f"Asistente: {comentario}")
                self._bridge.hablar(comentario)
                self._reaccionar_veredicto(cara, pendiente, estado.get("musica_ya_sonada", False))
                self._bridge.mostrar_cara("speaking")
            else:
                # Temas "sin veredicto" (Dilema del coche autónomo, etc.):
                # a pedido del usuario original, tampoco hay reacción
                # hablada -- se pasa directo a música/desplazamiento.
                time.sleep(PAUSA_ANTES_ACCION_FISICA)
                if not estado.get("musica_ya_sonada", False):
                    agent_behavior.expresar_musica(pendiente, self._bridge, logger=self.get_logger())
                agent_behavior.expresar_desplazamiento(pendiente, self._bridge, logger=self.get_logger())

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
        self._bridge.mostrar_cara("content")

    def _manejar_chat_libre(self, mensaje_usuario):
        self._bridge.mostrar_cara("speaking")
        texto_final = self.responder(mensaje_usuario)
        self.get_logger().info(f"Asistente: {texto_final}")
        self._bridge.hablar(texto_final)
        time.sleep(PAUSA_CAMBIO_CARA)
        self._bridge.mostrar_cara("content")

    # ══════════════════════════════════════════════════════════════════
    # Arranque/cierre de sesión y el callback de entrada.
    # ══════════════════════════════════════════════════════════════════

    def _iniciar_sesion(self):
        self._bridge.mostrar_cara("content")
        time.sleep(PAUSA_CAMBIO_CARA)
        self._bridge.mostrar_cara("speaking")
        # Primero quien soy y que se hacer, despues el nombre.
        self.get_logger().info(f"Asistente: {PRESENTACION}")
        self._bridge.hablar(PRESENTACION)
        saludo = random.choice(SALUDOS_APERTURA)
        self.get_logger().info(f"Asistente: {saludo}")
        self._bridge.hablar(saludo)
        time.sleep(PAUSA_CAMBIO_CARA)
        self._bridge.mostrar_cara("content")

        self.estado = {
            "esperando_nombre": True,
            "pregunta_pendiente": None, "esperando_tema": False, "en_trivia": False,
            "cola_preguntas": [], "ya_usados": set(), "aciertos": 0, "total": 0,
            "tema_actual": None, "nombre": None, "musica_ya_sonada": False,
        }


    def _finalizar_sesion(self):
        estado = self.estado
        self._bridge.mostrar_cara("speaking")
        if estado["total"]:
            resumen = f"Terminamos. Acertaste {estado['aciertos']} de {estado['total']}."
            self.get_logger().info(f"Asistente: {resumen}")
            self._bridge.hablar(resumen)
            time.sleep(PAUSA_CAMBIO_CARA)
            self._bridge.mostrar_cara("speaking")
        despedida = random.choice(DESPEDIDAS).format(nombre=estado["nombre"])
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

        if self.estado["esperando_nombre"]:
            nombre = entrada or "amigo"
            self.estado["nombre"] = nombre
            self.estado["esperando_nombre"] = False
            bienvenida = random.choice(BIENVENIDAS).format(nombre=nombre)
            self.get_logger().info(f"Asistente: {bienvenida}")
            self._bridge.hablar(bienvenida)
            self._bridge.mostrar_cara("content")
            return

        if entrada.lower() in ("salir", "exit", "quit"):
            self._finalizar_sesion()
            return

        if self.estado["en_trivia"]:
            pendiente = self.estado["pregunta_pendiente"]
            if self._quiere_salir_trivia(entrada, pendiente["pregunta"] if pendiente else None):
                self.estado["en_trivia"] = False
                self._bridge.mostrar_cara("content")
                # Igual que el original: este aviso queda solo en el log,
                # no se dice en voz alta -- se porta tal cual, sin "corregir"
                # esa asimetría acá.
                self.get_logger().info(
                    "Asistente: Listo, dejamos la trivia pausada — la "
                    "retomamos cuando quieras. ¿Qué tienes en mente?")
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
