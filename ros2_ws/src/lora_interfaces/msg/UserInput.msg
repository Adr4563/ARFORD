# Un mensaje de texto del usuario, sin importar de qué camino vino --
# teclado/stdin de la Pi, la página web (texto tipeado), o voz transcrita
# en el navegador del teléfono (Web Speech API, SpeechRecognition).
# communication_node.py unifica los 3 caminos en este único tópico
# (/lora/user_input), igual que la queue.Queue() compartida que usaban
# voz_server.py + Orchestrator_Management.py en el proyecto original --
# ahí confluían los 3 caminos en _entrada_queue.put(); acá confluyen en un
# solo publisher.

string text
string source   # "stdin" | "web" | "voice_browser"
