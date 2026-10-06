"""orchestrator_node -- el "cerebro" de Lora: máquina de estados de la
conversación (Trivia / Chat libre), portado de Orchestrator_Management.py
del proyecto monolítico original (deploy-raspberry-standalone/). Mismo
comportamiento turno a turno (router puro, Trivia con corrección
determinista, Chat libre con memoria episódica BM25, personalidad) -- lo
que cambia es la integración: en vez de importar display.py/voz_output/
Carrito_Client/Musica_Client/Camara_Client directo, todo lo de hardware
pasa por `_ros_bridge.RosBridge` (tópicos/servicios hacia lora_drivers).

Diseño de concurrencia (la pieza más delicada de esta migración): el
proceso original era 100% síncrono (un solo hilo, loop bloqueante leyendo
de una cola). Acá el turno sigue siendo síncrono EN SU LÓGICA (se procesa
un mensaje de punta a punta antes de aceptar el siguiente, protegido por
`self._turno_lock`), pero la ESPERA de cada servicio ROS2 (Speak,
PlayMusic, DetectEmotion) tiene que resolverse sin bloquear el proceso
entero -- ver la nota larga en _ros_bridge.RosBridge._llamar_servicio_sync().
Por eso este nodo corre con un MultiThreadedExecutor (ver main() al final)
y no con el rclpy.spin() default de un solo hilo.

Cambio de comportamiento deliberado respecto del original: cuando el
usuario dice "salir", el proceso original terminaba (era una app de
terminal para probar). Acá eso solo termina la SESIÓN -- el robot dice la
despedida y arranca una sesión nueva (_iniciar_sesion()), listo para el
próximo usuario. Ver _finalizar_sesion() más abajo.
"""
import difflib
import random
import threading
import time

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from lora_interfaces.msg import UserInput

from . import agent_behavior, llama_client, memoria_episodica, perf_monitor, preguntas, registro_chat
from .agent_corrector import evaluar_respuesta
from .agent_router import enrutar
from ._ros_bridge import RosBridge
from .personalidad import construir_personalidad, obtener_system_prompt

# ══════════════════════════════════════════════════════════════════════
# Catálogo de Trivia -- portado tal cual de Orchestrator_Management.py.
# Tiene que calzar EXACTO con el campo "tema" de preguntas.jsonl (columna
# "Actividad / Tema" del Excel original). Si el dataset gana/renombra un
# tema, esta lista se actualiza también -- ver la nota completa en el
# archivo original sobre por qué NO hay un tema "catch-all" genérico.
# ══════════════════════════════════════════════════════════════════════
TEMAS_CATALOGO = [
    "Arte, música y cultura - Nivel 1", "Arte, música y cultura - Nivel 2",
    "Arte, música y cultura - Nivel 3", "Arte, música y cultura - Nivel 4",
    "Arte, música y cultura - Nivel 5", "Arte, música y cultura - Nivel 6",
    "Arte, música y cultura - Nivel 7",
    "Ciencia y naturaleza - Nivel 1", "Ciencia y naturaleza - Nivel 2",
    "Ciencia y naturaleza - Nivel 3", "Ciencia y naturaleza - Nivel 4",
    "Ciencia y naturaleza - Nivel 5", "Ciencia y naturaleza - Nivel 6",
    "Ciencia y naturaleza - Nivel 7",
    "Cultura general - Nivel 1", "Cultura general - Nivel 2",
    "Deporte y entretenimiento - Nivel 1", "Deporte y entretenimiento - Nivel 2",
    "Dilema del coche autónomo - Nivel 1", "Dilema del coche autónomo - Nivel 2",
    "Geografía - Nivel 1", "Geografía - Nivel 2", "Geografía - Nivel 3",
    "Geografía - Nivel 4", "Geografía - Nivel 5", "Geografía - Nivel 6",
    "Geografía - Nivel 7", "Geografía - Nivel 8", "Geografía - Nivel 9",
    "Historia - Nivel 1", "Historia - Nivel 2", "Historia - Nivel 3",
    "Interacción personalizada (comida) - Nivel 1",
    "Interacción personalizada (comida) - Nivel 2",
    "Juego de emociones - Nivel 1",
    "Juego de imitación - Nivel 1", "Juego de imitación - Nivel 2",
    "Matemática: multiplicación - Nivel 1", "Matemática: multiplicación - Nivel 2",
    "Matemática: multiplicación - Nivel 3", "Matemática: multiplicación - Nivel 4",
    "Razonamiento matemático - Nivel 1", "Razonamiento matemático - Nivel 2",
    "Razonamiento matemático - Nivel 3", "Razonamiento matemático - Nivel 4",
    "Razonamiento matemático - Nivel 5", "Razonamiento matemático - Nivel 6",
    "Razonamiento matemático - Nivel 7",
    "Reconocimiento musical - Nivel 1", "Reconocimiento musical - Nivel 2",
    "Socialización / presentación - Nivel 1",
]

# Únicos temas con veredicto real vía cámara en vez de texto.
TEMAS_JUEGO_EMOCIONES = {
    "Juego de emociones - Nivel 1",
    "Juego de imitación - Nivel 1",
    "Juego de imitación - Nivel 2",
}

RESPUESTAS_TRIVIA_ACIERTO = [
    "¡Exacto, {esperada}!", "Sí, correcto.", "Así es, nada mal.", "Correcto, bien ahí.",
]
RESPUESTAS_TRIVIA_ERROR = [
    "No, era {esperada}.", "Nop, la correcta es {esperada}.",
    "Incorrecto, la respuesta era {esperada}.", "No es así, correcta: {esperada}.",
]

SALUDOS_APERTURA = [
    "Hola, mi nombre es Lora, ¿cuál es tu nombre?",
    "¡Hola! Soy Lora. ¿Y vos cómo te llamás?",
    "Hola, hola. Soy Lora, tu robot. ¿Quién sos vos?",
    "¡Buenas! Me llamo Lora. Contame tu nombre.",
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

        # Solo para decidir si vale la pena esperar a un teléfono antes del
        # saludo (ver _esperar_telefono_si_corresponde) -- el motor de voz
        # REAL lo decide communication_node con su propio parámetro
        # `voz_motor`; los dos tienen que declararse con el mismo valor por
        # separado (mismo trade-off ya documentado en personalidad.py del
        # proyecto original sobre duplicar un default entre dos módulos).
        self.declare_parameter('voz_motor', 'telefono')
        self.declare_parameter('esperar_telefono_seg', 60)

        self._bridge = RosBridge(self)
        self.estado = None
        self._turno_lock = threading.Lock()
        self._grupo_turno = ReentrantCallbackGroup()

        self.create_subscription(
            UserInput, '/lora/user_input', self._on_user_input, 10,
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

    def _mensajes_con_personalidad(self, persona_str, contenido_usuario, modelo=None):
        modelo = modelo or llama_client.CHAT_MODEL
        system_prompt = obtener_system_prompt(persona_str, modelo=modelo)
        mensajes = []
        if system_prompt:
            mensajes.append({"role": "system", "content": system_prompt})
        mensajes.append({"role": "user", "content": contenido_usuario})
        return mensajes

    @staticmethod
    def _quitar_pregunta_final(texto):
        texto = texto.strip()
        if not texto.endswith("?"):
            return texto
        idx = texto.rfind("¿")
        return texto[:idx].strip() if idx > 0 else texto

    def responder(self, mensaje_usuario, persona_str):
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

    def comentar_resultado_emocion(self, objetivo, detectada, acerto, persona_str):
        """Mantenido por paridad con el original -- SIN caller activo hoy
        (_jugar_emociones() usa "¡Correcto!"/"¡Incorrecto!" fijo, no esto),
        mismo estado que en Orchestrator_Management.py original."""
        if acerto:
            instruccion = (
                f"Le pediste al usuario que pusiera cara de {objetivo.lower()} y la cámara "
                f"detectó justo esa cara. Celébraselo en máximo 5 palabras.")
        else:
            instruccion = (
                f"Le pediste al usuario que pusiera cara de {objetivo.lower()} pero la cámara "
                f"detectó cara de {detectada.lower()}. Dile en máximo 5 palabras que no era esa, "
                "con humor.")
        mensajes = self._mensajes_con_personalidad(
            persona_str, f"{instruccion} No vuelvas a pedir otra cara ni hagas otra pregunta.",
            modelo=llama_client.TRIVIA_MODEL)
        return llama_client.generar_respuesta(
            mensajes, modelo=llama_client.TRIVIA_MODEL, max_tokens=20).strip()

    @staticmethod
    def resolver_tema(eleccion_usuario):
        texto = eleccion_usuario.strip().lower()
        catalogo_low = [t.lower() for t in TEMAS_CATALOGO]
        for tema, tema_low in zip(TEMAS_CATALOGO, catalogo_low):
            if texto == tema_low or tema_low in texto or texto in tema_low:
                return tema
        cercano = difflib.get_close_matches(texto, catalogo_low, n=1, cutoff=0.5)
        if cercano:
            return TEMAS_CATALOGO[catalogo_low.index(cercano[0])]
        return random.choice(TEMAS_CATALOGO)

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

    def _jugar_emociones(self, pregunta, persona_str):
        objetivo = (pregunta.get("respuesta_esperada") or "").strip()
        if not objetivo:
            opciones = [o.strip() for o in pregunta["cara"].split("/") if o.strip()]
            objetivo = random.choice(opciones) if opciones else "Feliz"

        self._bridge.mostrar_cara(agent_behavior.cara_para_emocion(objetivo) or "content")
        pedido = f"¡Hazme una cara de {objetivo.lower()}!"
        self.get_logger().info(f"Asistente: {pedido}")
        self._bridge.hablar(pedido)
        time.sleep(1)  # un segundo para que el usuario pose antes de capturar
        detectada, confianza, ok = self._bridge.detectar_emocion()

        if not ok or detectada is None:
            self._bridge.mostrar_cara("speaking")
            time.sleep(PAUSA_ANTES_ACCION_FISICA)
            agent_behavior.expresar_musica(pregunta, self._bridge, logger=self.get_logger())
            agent_behavior.expresar_desplazamiento(pregunta, self._bridge, logger=self.get_logger())
            return None

        acerto = detectada == objetivo
        self.get_logger().info(
            f"[cámara] pedido={objetivo} detectado={detectada} "
            f"(confianza {confianza:.0%}) -> {'OK' if acerto else 'MAL'}")
        self._bridge.hablar("¡Correcto!" if acerto else "¡Incorrecto!")
        cara = agent_behavior.elegir_cara_pregunta(pregunta, acerto) or ("happy" if acerto else "sad")
        self._reaccionar_veredicto(cara, pregunta)
        self._bridge.mostrar_cara("speaking")
        return acerto

    def _correr_tanda_emociones(self, persona_str):
        estado = self.estado
        while self._preguntar_siguiente():
            pendiente = estado["pregunta_pendiente"]
            estado["pregunta_pendiente"] = None
            acerto = self._jugar_emociones(pendiente, persona_str)
            if acerto is not None:
                estado["total"] += 1
                estado["aciertos"] += acerto
        self._cerrar_tanda()

    def _iniciar_tanda(self, tema, persona_str=None):
        estado = self.estado
        preguntas_tanda = preguntas.preguntas_por_tema(
            tema, estado["ya_usados"], cantidad=PREGUNTAS_POR_TANDA)
        if not preguntas_tanda:
            self.get_logger().info(f"Asistente: No quedan preguntas de {tema}.")
            self._bridge.hablar(f"No quedan preguntas de {tema}.")
            return
        estado["ya_usados"].update(p["id"] for p in preguntas_tanda)
        estado["cola_preguntas"] = preguntas_tanda
        if tema in TEMAS_JUEGO_EMOCIONES:
            self._correr_tanda_emociones(persona_str)
        else:
            self._preguntar_siguiente()

    def _manejar_trivia(self, mensaje_usuario, persona_str):
        estado = self.estado

        if estado["esperando_tema"]:
            estado["esperando_tema"] = False
            tema = self.resolver_tema(mensaje_usuario)
            estado["tema_actual"] = tema
            anuncio = f"Vamos con {tema}. Van {PREGUNTAS_POR_TANDA} preguntas seguidas."
            self.get_logger().info(f"Asistente: {anuncio}")
            self._bridge.hablar(anuncio)
            self._iniciar_tanda(tema, persona_str)
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

        opciones = ", ".join(random.sample(TEMAS_CATALOGO, 5))
        estado["esperando_tema"] = True
        anuncio = f"Puedes elegir entre: {opciones}."
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

    def _manejar_chat_libre(self, mensaje_usuario, persona_str):
        self._bridge.mostrar_cara("speaking")
        texto_final = self.responder(mensaje_usuario, persona_str)
        self.get_logger().info(f"Asistente: {texto_final}")
        self._bridge.hablar(texto_final)
        time.sleep(PAUSA_CAMBIO_CARA)
        self._bridge.mostrar_cara("content")

    # ══════════════════════════════════════════════════════════════════
    # Arranque/cierre de sesión y el callback de entrada.
    # ══════════════════════════════════════════════════════════════════

    def _esperar_telefono_si_corresponde(self):
        if self.get_parameter('voz_motor').value != 'telefono':
            return
        timeout = int(self.get_parameter('esperar_telefono_seg').value)
        self.get_logger().info(
            f"[voz] esperando hasta {timeout}s a que se conecte un teléfono con la voz activada...")
        esperado = 0
        while esperado < timeout and not self._bridge.voz_cliente_conectado():
            time.sleep(1)
            esperado += 1
        if self._bridge.voz_cliente_conectado():
            self.get_logger().info(f"[voz] teléfono conectado a los {esperado}s")
        else:
            self.get_logger().info(f"[voz] nadie se conectó en {timeout}s -- el saludo cae al respaldo")

    def _iniciar_sesion(self):
        self._esperar_telefono_si_corresponde()
        self._bridge.mostrar_cara("content")
        time.sleep(PAUSA_CAMBIO_CARA)
        self._bridge.mostrar_cara("speaking")
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
            "persona_str": None,
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
            self.estado["persona_str"] = construir_personalidad()
            bienvenida = (f"Mucho gusto, {nombre}. Podemos jugar Trivia o simplemente charlar "
                          "-- vos decidís, decime qué querés hacer.")
            self.get_logger().info(f"Asistente: {bienvenida}")
            self._bridge.hablar(bienvenida)
            self._bridge.mostrar_cara("content")
            return

        if entrada.lower() in ("salir", "exit", "quit"):
            self._finalizar_sesion()
            return

        persona_str = self.estado["persona_str"]

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
            self._manejar_trivia(entrada, persona_str)
            return

        ruta = self._enrutar_mensaje(entrada)
        if ruta == "TRIVIA":
            self.estado["en_trivia"] = True
            if self.estado["pregunta_pendiente"] is not None or self.estado["esperando_tema"]:
                self._reanudar_trivia()
            else:
                self._manejar_trivia(entrada, persona_str)
        else:
            self._manejar_chat_libre(entrada, persona_str)


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
