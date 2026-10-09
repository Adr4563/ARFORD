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
