# ARFORD v2: sesión guiada + memoria de conversaciones

Plan de arquitectura. Nada de esto está implementado todavía.

## El problema

Hoy la sesión arranca con un saludo que no dice qué sabe hacer el robot, y la
memoria de Chat libre busca por palabras compartidas (BM25): si en una sesión
dijiste "me sentía solo" y hoy decís "estaba triste", no lo encuentra.

## Qué NO hacer: Agentic RAG

Un router con LLM decidiendo en cada turno cuesta ~7 s en la Pi
(`arford-empatico` mide 2,7 tok/s, 20 tokens de decisión). El router actual
(`orchestrator_node.enrutar()`, TF-IDF + regresión logística) decide en microsegundos con
95-96% de acierto medido. Cambiarlo sería más lento y menos preciso.

**Decisión: el enrutado sigue siendo determinista. RAG solo para la memoria.**

## Lo que ya existe y se reusa

| Pieza | Archivo | Qué hace |
|---|---|---|
| Router TRIVIA/CHAT_LIBRE | `orchestrator_node.enrutar()` | ML, microsegundos, 95-96% |
| Juego de caras | `orchestrator_node._jugar_emociones()` | pide emociones, muestra caras |
| Trivia por categorías | `orchestrator_node._manejar_trivia()` | 5 categorías |
| ~~Registro de conversaciones~~ | eliminado | nadie lo leia; los 469 turnos quedan en `~/.romeo/chat_libre_training/conversaciones.jsonl` |
| ~~Memoria BM25~~ | eliminada | buscaba por palabras y la memoria era compartida entre usuarios |
| Embeddings en la Pi | `nomic-embed-text` en Ollama | 274 MB, ya instalado, soporta español |

## Cambio 1: presentación con funcionalidades

`_iniciar_sesion()` hoy saluda y pide el nombre. Pasa a enumerar lo que sabe
hacer, en una frase corta por capacidad (la voz tarda, así que no más de tres).

    Hola, soy ARFORD. Puedo jugar trivia con vos, hacer caras y emociones,
    o simplemente conversar. ¿Cómo te llamás?

Texto en una constante nueva junto a `SALUDOS_APERTURA`, no hardcodeado en el
método.

## Cambio 2: ruta CARAS en el router

Hoy `_quiere_jugar_emociones()` es un match de palabras aparte del router. Se
integra como tercera clase del clasificador: `TRIVIA` / `CARAS` / `CHAT_LIBRE`.

Hace falta reentrenar `router_modelo.joblib` con ejemplos de la clase CARAS
("hacé una cara", "mostrame cómo te sentís", "jugamos a las emociones").
El script de entrenamiento original está en el proyecto previo; si no se
encuentra, se reconstruye: TF-IDF de n-gramas de caracteres + LogisticRegression,
que es lo que el docstring documenta.

## Cambio 3: memoria RAG de conversaciones

Reemplaza BM25 por búsqueda semántica sobre lo que YA se guarda en
`conversaciones.jsonl`. No se entrena ningún modelo: `nomic-embed-text` por
Ollama, igual que en `rag_eval.py`.

Flujo por turno de Chat libre:

1. Al arrancar, leer `conversaciones.jsonl` y pedir a Ollama el embedding de
   cada turno pasado. Cachear en `~/.romeo/memoria/vectores.npy` + un
   `.json` con los textos, para no recalcular en cada arranque (373 turnos
   ≈ 373 llamadas; con caché solo se embeddan los turnos nuevos).
2. Al llegar un mensaje, embeddarlo y buscar los 3 turnos pasados más
   parecidos por coseno con numpy (sin ChromaDB: no hay wheel para
   Python 3.14 ARM64, ya se comprobó en `rag_eval.py`).
3. Filtro de relevancia: descartar los que queden por debajo de 0.5 de
   similitud. Sin filtro, siempre devuelve algo y mete contexto que no
   viene al caso -- el mismo razonamiento del `_OVERLAP_MINIMO` de BM25.
4. Si sobrevive alguno, inyectarlo en el prompt de `arford-empatico` como
   contexto ("en una conversación anterior dijiste: ...").

Archivo nuevo: `romeo_brain/memoria_rag.py`. Misma interfaz pública que
`memoria_episodica.py`, que era la implementación BM25 y se eliminó.

### Coste en la Pi

| | |
|---|---|
| Embeddar 1 mensaje | ~0,3 s (nomic-embed-text, medido en rag_eval.py) |
| Buscar en 373 vectores | <10 ms (numpy) |
| Primer arranque (sin caché) | ~2 min para 373 turnos, una sola vez |
| RAM extra | 373 × 768 × 4 B ≈ 1,1 MB |

El sobrecoste real por turno es el embedding: ~0,3 s. Aceptable frente a los
~6 s que ya tarda `arford-empatico` en responder.

## Cambio 4: sincronización PC <-> Pi

Hoy no existe: los cambios se editan en el PC y hay que copiarlos a mano
(esto ya causó que `arford-empatico` pareciera no estar implementado -- el
launch de la Pi seguía apuntando a `romeo-chat-libre-v4`).

Script `sincronizar.sh` con dos direcciones:

    ./sincronizar.sh subir     # PC -> Pi: código (rsync del ros2_ws/src)
    ./sincronizar.sh bajar     # Pi -> PC: conversaciones.jsonl + logs

La IP va como variable, no fija: ya cambió una vez (10.143.75.137 ->
10.143.102.167) y eso dejó `4_desplegar_pi.sh` apuntando a una IP muerta.

El `bajar` es lo que permite entrenar en el PC con las conversaciones reales
del robot, que es el ciclo que cierra todo esto.

## Orden de implementación

1. **Presentación** -- es texto, 10 min, se nota de inmediato.
2. **sincronizar.sh** -- desbloquea todo lo demás, evita el desfase PC/Pi.
3. **memoria_rag.py** -- el cambio de fondo; probar en el PC primero (Ollama
   local), después en la Pi.
4. **Ruta CARAS** -- requiere reentrenar el router, es el más largo.

## Lo que hay que medir, no suponer

- Si `nomic-embed-text` distingue bien frases cortas en español ("estoy
  triste" vs "estoy cansado"). Si falla, la alternativa es
  `paraphrase-multilingual-MiniLM-L12-v2` por sentence-transformers, que
  habría que instalar en la Pi.
- Si el umbral de 0.5 de coseno es el correcto: se ajusta con las 373
  conversaciones reales que ya hay, no a ojo.
