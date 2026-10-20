# -*- coding: utf-8 -*-
"""Terminal de prueba PARA CORRER EN LA RASPBERRY, con las IA en la PC.

Tu escribes aca (en la Pi, por SSH) y los modelos responden desde la PC:
Ollama en :11434. No usa ROS, ni microfono, ni camara, ni altavoz -- solo
el cerebro, para medir que tan fluido quedo el dialogo.

Usa las piezas REALES del robot, sin copiar nada:
    orchestrator_node.enrutar()   router TRIVIA / CHAT_LIBRE (sklearn)
    agente_chat.responder()       chat libre con historial
    agente_trivia                 corrector + comentario
    preguntas                     banco por categorias
    frases                        las frases de archivo

    python3 probar_en_pi.py                    # autodetecta la PC
    python3 probar_en_pi.py 10.143.161.32      # IP de la PC a mano

Comandos:
    /nuevo          sesion nueva (olvida el hilo)
    /trivia [cat]   plantea una pregunta (al azar, o de esa categoria)
    /categorias     lista las categorias
    /historial      que recuerda ahora mismo
    /tiempos        resumen de latencias
    /salir
"""
import os
import random
import sys
import time

AQUI = os.path.dirname(os.path.abspath(__file__))
for _p in ("romeo_brain", "romeo_drivers"):
    _ruta = os.path.join(AQUI, "ros2_ws", "src", _p)
    if os.path.isdir(_ruta):
        sys.path.insert(0, _ruta)

# ── Donde viven las IA ────────────────────────────────────────────────
PC = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ROMEO_PC", "10.143.161.32")
os.environ["CHAT_SERVER_HOST"] = f"http://{PC}:11434"
os.environ.setdefault("CHAT_MODEL", "llama3.2:1b")
os.environ.setdefault("TRIVIA_CORRECTOR_MODEL", "romeo")

from romeo_brain import agente_chat, agente_trivia, frases, llama_client  # noqa: E402
from romeo_brain import preguntas as banco  # noqa: E402
from romeo_brain.orchestrator_node import enrutar  # noqa: E402

tiempos = []


def medir(etiqueta, fn, *a, **k):
    """Llama a fn cronometrando, y guarda la medida para /tiempos."""
    t0 = time.monotonic()
    r = fn(*a, **k)
    ms = (time.monotonic() - t0) * 1000
    tiempos.append((etiqueta, ms))
    return r, ms


def resumen_tiempos():
    if not tiempos:
        return ["  (sin medidas aun)"]
    por_tipo = {}
    for e, ms in tiempos:
        por_tipo.setdefault(e, []).append(ms)
    filas = [f"  {len(tiempos)} llamadas:"]
    for e, xs in sorted(por_tipo.items()):
        xs_ord = sorted(xs)
        filas.append(f"    {e:18s} n={len(xs):2d}  min {xs_ord[0]:6.0f}  "
                     f"med {xs_ord[len(xs) // 2]:6.0f}  max {xs_ord[-1]:6.0f} ms")
    return filas


# ── Trivia ────────────────────────────────────────────────────────────
pregunta_abierta = None
_ya_preguntadas = []


def plantear_trivia(categoria=None):
    """Saca una pregunta como el robot: por categoria y sin repetir."""
    global pregunta_abierta
    if categoria is None:
        categoria = random.choice(banco.categorias_disponibles())
    lote = banco.preguntas_por_categoria(categoria, excluir=_ya_preguntadas,
                                         cantidad=1)
    if not lote:
        return f"Ya no quedan preguntas de {categoria}."
    p = lote[0]
    _ya_preguntadas.append(p["id"])
    pregunta_abierta = p
    return f"[{categoria}]  {p['pregunta']}"


def resolver_trivia(respuesta):
    global pregunta_abierta
    p = pregunta_abierta
    pregunta_abierta = None
    esperada = p["respuesta_esperada"]
    ok, ms_eval = medir("trivia.evaluar", agente_trivia.evaluar_respuesta,
                        esperada, respuesta)
    salida = [f"  [{'CORRECTO' if ok else 'INCORRECTO'}]  esperada: "
              f"{esperada}   ({ms_eval:.0f} ms)"]
    com, ms_com = medir("trivia.comentar", agente_trivia.comentar,
                        p["pregunta"], esperada, respuesta, ok)
    salida.append(f"ROMEO> {com}   ({ms_com:.0f} ms)")
    return "\n".join(salida)


# ── Arranque ──────────────────────────────────────────────────────────
print()
print("  === Romeo: cerebro en la Pi, IA en la PC ===")
print(f"  PC            {PC}")
print(f"  chat libre    {llama_client.CHAT_MODEL}")
print(f"  corrector     {agente_trivia.CORRECTOR_LLM}")
print(f"  historial     {agente_chat.TURNOS_HISTORIAL} turnos")
print("  comandos      /nuevo  /trivia [cat]  /categorias  /historial"
      "  /tiempos  /salir")
print()
print(f"ROMEO> {frases.una(frases.PRESENTACIONES)}")
print()

while True:
    try:
        entrada = input("tu> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if not entrada:
        continue

    if entrada == "/salir":
        break

    if entrada == "/nuevo":
        agente_chat.reiniciar()
        pregunta_abierta = None
        _ya_preguntadas.clear()
        print("  (sesion nueva)")
        print()
        continue

    if entrada == "/historial":
        h = getattr(agente_chat, "_historial", [])
        print(f"  {len(h)} mensajes en memoria:")
        for m in h:
            print(f"    {m['role']:9s} {m['content'][:70]}")
        print()
        continue

    if entrada == "/tiempos":
        for linea in resumen_tiempos():
            print(linea)
        print()
        continue

    if entrada == "/categorias":
        print(f"  {', '.join(banco.categorias_disponibles())}")
        print()
        continue

    if entrada.startswith("/trivia"):
        partes = entrada.split(None, 1)
        cat = None
        if len(partes) > 1:
            cat = banco.resolver_categoria(partes[1])
            if not cat:
                print(f"  no conozco esa categoria. Hay: "
                      f"{', '.join(banco.categorias_disponibles())}")
                print()
                continue
        print(f"ROMEO> {plantear_trivia(cat)}")
        print()
        continue

    # Con una pregunta abierta, lo que escribas ES la respuesta.
    if pregunta_abierta is not None:
        try:
            print(resolver_trivia(entrada))
        except Exception as e:
            print(f"  ERROR en trivia: {type(e).__name__}: {str(e)[:120]}")
        print()
        continue

    # Si no, decide el router.
    ruta, ms_ruta = medir("router", enrutar, entrada)
    print(f"  [router: {ruta}  {ms_ruta:.0f} ms]")

    if ruta == "TRIVIA":
        print(f"ROMEO> {plantear_trivia()}")
    else:
        try:
            r, ms = medir("chat", agente_chat.responder, entrada)
            print(f"ROMEO> {r}   ({ms:.0f} ms)")
        except Exception as e:
            print(f"  ERROR en chat: {type(e).__name__}: {str(e)[:120]}")
    print()

print()
print("  === resumen de la sesion ===")
for linea in resumen_tiempos():
    print(linea)
print("  adios")
print()
