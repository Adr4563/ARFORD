# Comando de MOTORES -- mismo protocolo de texto que ya entiende el
# firmware del ESP32-S3 sin modificar (ver carrito-mecanum-esp32/
# 2-l298n-mecanum/mecanum_car_esp32s3.ino::ejecutarComando()), no se
# inventa un protocolo nuevo: moving_control_node.py solo traduce este mensaje a
# la misma línea de texto que ya escribía Clients/Carrito_Client.py por
# Serial.
#
# ROTATE360 es el único valor que NO es un comando directo del firmware --
# moving_control_node.py lo expande a varios pulsos de "RR" seguidos con pausa
# entre cada uno (igual que mover_360() en el código original), porque
# "girar 360°" no existe como comando atómico en el firmware. Sin
# calibrar contra el hardware real (grados por pulso desconocidos) -- se
# hereda tal cual del proyecto original.

string command   # F | B | SL | SR | RL | RR | S | ROTATE360
