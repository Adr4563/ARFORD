# Qué cara mostrar en pantalla -- mismas 6 claves que CARAS_VALIDAS en
# display.py del proyecto original. visualization_faces_node.py es quien
# sabe traducir esto a faces/<face_name>.mp4 (backend DRM, Pi headless) o
# faces/<face_name>.gif (backend Tkinter, demo en PC) -- orchestrator_node
# no necesita saber cuál de los dos backends está activo, igual que antes
# display.mostrar_cara() abstraía eso para chat.py/Orchestrator_Management.py.

string face_name   # happy | sad | angry | content | speaking | countdown
