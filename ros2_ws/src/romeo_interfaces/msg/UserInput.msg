# Un mensaje de texto del usuario, sin importar de qué camino vino --
# el micrófono de la placa ESP32-S3 (transcrito en la Pi con sherpa-onnx,
# ver romeo_drivers/_stt_engine.py) o el teclado/stdin de la Pi.
# communication_node unifica los dos caminos en este único tópico
# (/romeo/user_input), igual que la queue.Queue() compartida del proyecto
# original -- ahí confluían en _entrada_queue.put(); acá confluyen en un
# solo publisher.
#
# Hubo un tercer camino, "web"/"voice_browser": una página que se abría en
# el celular para escribirle o hablarle (Web Speech API del navegador). Se
# eliminó al dejar de depender de un teléfono; la voz ahora se procesa en
# la Pi con el audio de la placa.
#
# orchestrator_node NO usa `source` para decidir nada (trata todas las
# entradas igual) -- queda como traza en los logs, para saber si un turno
# entró por voz o por teclado.

string text
string source   # "voice_board" | "stdin"

# Ventana de percepcion afectiva de ESTA frase (tesis de Rocha et al.,
# sec. 3.4.1.2): cuando empezo y cuando termino de hablar el usuario, en
# segundos de reloj (time.time()) de la Pi. orchestrator_node pide con
# ellos el resumen de la camara (PerceiveFace) de ese mismo intervalo.
# 0.0 cuando no hay ventana (entrada por teclado).
float64 t_inicio
float64 t_fin

# SER: emocion de la VOZ de esta frase, calculada en la PC sobre el mismo
# audio que se transcribe (romeo_drivers/_ser_engine.py). Categorias con
# probabilidades, en claves de estado_afectivo.ANCLAJES. Vacios si no hubo
# SER (teclado, frase muy corta o servidor sin modelo).
string[] ser_etiquetas
float32[] ser_probabilidades

# TER: emocion del TEXTO de esta frase, calculada en la PC sobre el texto
# que acaba de transcribir (romeo_drivers/_ter_engine.py, RoBERTuito).
# Mismo formato y mismas claves que el SER. Vacios si no hubo TER (teclado,
# o servidor sin el modelo): entonces orchestrator_node cae a las palabras
# clave de romeo_brain/ter.py, que corren en la Pi y no cuestan nada.
string[] ter_etiquetas
float32[] ter_probabilidades
