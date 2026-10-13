// Mecanum Wheel Robot Car - ESP32-S3 (DevKitC-1) version
// Basado en https://robotlk.com/web-controlled-mecanum-wheel-robot-car-using-esp32/
// Cambios respecto al original:
//   1. Pines remapeados: el original usa GPIO 26-33, reservados/no expuestos en ESP32-S3.
//   2. Corregidas forwardLeft/forwardRight/backLeft/backRight (estaban cruzadas izq/der).
//   3. Watchdog: si no llega ningun comando en 500ms, detiene motores (evita carrito descontrolado
//      si se corta el cable, el proceso del otro lado se cuelga, o se cierra la pagina de control).
//   4. Control por Serial (USB): el ESP32 lee comandos línea a línea desde el cable a la
//      Raspberry Pi. En ARFORD los manda romeo_drivers/_cart_serial.py (clase CartSerial),
//      que usa moving_control_node tanto para la trivia como para la pagina de control.
//   5. SIN WiFi: el ESP32 no abre red propia ni se une a ninguna. La pagina de control
//      manual la sirve la Raspberry Pi (romeo_drivers/_control_web.py, servida por
//      moving_control_node) y manda los mismos comandos por el cable USB.
//   6. Comandos T1..T4 de diagnostico: mueven UNA rueda a la vez (ver ejecutarComando).

// Motor pins (ESP32-S3-DevKitC-1: evita 0,3,19,20,26-32,43-46)
//
// Dos correcciones acumuladas sobre el mapeo original, las dos en los pines y
// no en cada funcion de movimiento: asi los 10 comandos quedan bien de una
// sola vez, y vale igual para la pagina de control y para ROS, que comparten
// ejecutarComando().
//
// 1) Chasis montado 180 grados respecto a lo que asumia el codigo: lo que se
//    llamaba "frente" es fisicamente la parte trasera, asi que cada rueda pasa
//    a su opuesta en diagonal (GPIO 4-7 <-> 15-18). Lo delataron adelante/atras
//    y las diagonales invertidas, PERO la rotacion correcta: una inversion de
//    polaridad de los 4 motores habria dado vuelta tambien el giro; una
//    rotacion de 180 grados lo preserva.
//
// 2) Lado izquierdo con la polaridad invertida: FL y BL giraban hacia atras
//    mientras FR y BR giraban hacia adelante, asi que el carrito rotaba en vez
//    de avanzar. Se comprobo rueda por rueda con los comandos T1..T4 de abajo.
//    Arreglado intercambiando el par de pines de las dos ruedas izquierdas
//    (cada rueda se queda en sus mismos GPIO; cambia cual de los dos es el
//    "adelante").
//
//    OJO: se probo ademas intercambiar los cuerpos de rotateLeft()/rotateRight(),
//    suponiendo que invertir un solo lado daria vuelta el sentido de giro.
//    Resulto al reves en el carrito real -- "RL" giraba a la derecha -- asi que
//    se deshizo: el mapeo de pines ya compensa el giro por si solo.
int IN1 = 17, IN2 = 18;  // Front Left  (el motor que el codigo tenia como Back Right)
int IN3 = 16, IN4 = 15;  // Front Right (el que tenia como Back Left)
int IN5 = 6,  IN6 = 7;   // Back Left   (el que tenia como Front Right)
int IN7 = 5,  IN8 = 4;   // Back Right  (el que tenia como Front Left)

unsigned long lastCmdTime = 0;
const unsigned long CMD_TIMEOUT = 500; // ms

void setup() {
  Serial.begin(115200);

  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);
  pinMode(IN5, OUTPUT); pinMode(IN6, OUTPUT);
  pinMode(IN7, OUTPUT); pinMode(IN8, OUTPUT);

  stopCar();
  lastCmdTime = millis();

  Serial.println("Listo. Comandos por Serial (sin WiFi): F/B/SL/SR/RL/RR/FL/FR/BL/BR/S.");
}

void loop() {
  // Un comando por línea (terminada en '\n'), igual que antes era un query
  // param por request — se lee de a línea completa, no byte a byte, para no
  // ejecutar un comando a medio escribir.
  if (Serial.available()) {
    String linea = Serial.readStringUntil('\n');
    linea.trim();
    if (linea.length() > 0) {
      ejecutarComando(linea);
      lastCmdTime = millis();
    }
  }

  // Watchdog de seguridad: si no llega comando reciente por el Serial,
  // detener motores -- protege contra cable desconectado, proceso Python
  // colgado, o pagina de control cerrada a mitad de un movimiento.
  if (millis() - lastCmdTime > CMD_TIMEOUT) {
    stopCar();
  }
}

// ===== Interpretar comando =====
void ejecutarComando(const String& move) {
  if (move == "F") forward();
  else if (move == "B") backward();
  else if (move == "SL") strafeLeft();
  else if (move == "SR") strafeRight();
  else if (move == "RL") rotateLeft();
  else if (move == "RR") rotateRight();
  else if (move == "FL") forwardLeft();
  else if (move == "FR") forwardRight();
  else if (move == "BL") backLeft();
  else if (move == "BR") backRight();
  // --- Comandos de diagnostico: una sola rueda, para aislar cableado ---
  else if (move == "T1") testWheel(IN1, IN2);  // Front Left
  else if (move == "T2") testWheel(IN3, IN4);  // Front Right
  else if (move == "T3") testWheel(IN5, IN6);  // Back Left
  else if (move == "T4") testWheel(IN7, IN8);  // Back Right
  else stopCar(); // incluye "S" y cualquier texto no reconocido
}

// Diagnostico: mueve UNA rueda hacia adelante y deja las otras tres
// paradas. Separa 'mapeo equivocado' de 'canal del driver muerto' de
// 'alimentacion insuficiente', que dan sintomas parecidos vistos desde
// afuera. Respeta el watchdog igual que el resto: hay que repetir el
// comando para que la rueda siga girando.
void testWheel(int a, int b){
  stopCar();
  digitalWrite(a, HIGH); digitalWrite(b, LOW);
}

// ===== Motor Control Logic =====
void stopCar(){
  digitalWrite(IN1,LOW); digitalWrite(IN2,LOW);
  digitalWrite(IN3,LOW); digitalWrite(IN4,LOW);
  digitalWrite(IN5,LOW); digitalWrite(IN6,LOW);
  digitalWrite(IN7,LOW); digitalWrite(IN8,LOW);
}

void forward(){
  digitalWrite(IN1,HIGH); digitalWrite(IN2,LOW);
  digitalWrite(IN3,HIGH); digitalWrite(IN4,LOW);
  digitalWrite(IN5,HIGH); digitalWrite(IN6,LOW);
  digitalWrite(IN7,HIGH); digitalWrite(IN8,LOW);
}

void backward(){
  digitalWrite(IN1,LOW); digitalWrite(IN2,HIGH);
  digitalWrite(IN3,LOW); digitalWrite(IN4,HIGH);
  digitalWrite(IN5,LOW); digitalWrite(IN6,HIGH);
  digitalWrite(IN7,LOW); digitalWrite(IN8,HIGH);
}

void strafeRight(){
  digitalWrite(IN1,HIGH); digitalWrite(IN2,LOW);  // FL
  digitalWrite(IN3,LOW);  digitalWrite(IN4,HIGH); // FR
  digitalWrite(IN5,LOW);  digitalWrite(IN6,HIGH); // BL
  digitalWrite(IN7,HIGH); digitalWrite(IN8,LOW);  // BR
}

void strafeLeft(){
  digitalWrite(IN1,LOW);  digitalWrite(IN2,HIGH);
  digitalWrite(IN3,HIGH); digitalWrite(IN4,LOW);
  digitalWrite(IN5,HIGH); digitalWrite(IN6,LOW);
  digitalWrite(IN7,LOW);  digitalWrite(IN8,HIGH);
}

void rotateRight(){
  digitalWrite(IN1,HIGH); digitalWrite(IN2,LOW);
  digitalWrite(IN3,LOW);  digitalWrite(IN4,HIGH);
  digitalWrite(IN5,HIGH); digitalWrite(IN6,LOW);
  digitalWrite(IN7,LOW);  digitalWrite(IN8,HIGH);
}

void rotateLeft(){
  digitalWrite(IN1,LOW);  digitalWrite(IN2,HIGH);
  digitalWrite(IN3,HIGH); digitalWrite(IN4,LOW);
  digitalWrite(IN5,LOW);  digitalWrite(IN6,HIGH);
  digitalWrite(IN7,HIGH); digitalWrite(IN8,LOW);
}

// --- Diagonales corregidas (antes estaban cruzadas izq/der) ---

void forwardLeft(){
  digitalWrite(IN1,HIGH); digitalWrite(IN2,LOW);  // FL forward
  digitalWrite(IN3,LOW);  digitalWrite(IN4,LOW);  // FR stop
  digitalWrite(IN5,LOW);  digitalWrite(IN6,LOW);  // BL stop
  digitalWrite(IN7,HIGH); digitalWrite(IN8,LOW);  // BR forward
}

void forwardRight(){
  digitalWrite(IN1,LOW);  digitalWrite(IN2,LOW);  // FL stop
  digitalWrite(IN3,HIGH); digitalWrite(IN4,LOW);  // FR forward
  digitalWrite(IN5,HIGH); digitalWrite(IN6,LOW);  // BL forward
  digitalWrite(IN7,LOW);  digitalWrite(IN8,LOW);  // BR stop
}

void backLeft(){
  digitalWrite(IN1,LOW);  digitalWrite(IN2,LOW);  // FL stop
  digitalWrite(IN3,LOW);  digitalWrite(IN4,HIGH); // FR backward
  digitalWrite(IN5,LOW);  digitalWrite(IN6,HIGH); // BL backward
  digitalWrite(IN7,LOW);  digitalWrite(IN8,LOW);  // BR stop
}

void backRight(){
  digitalWrite(IN1,LOW);  digitalWrite(IN2,HIGH); // FL backward
  digitalWrite(IN3,LOW);  digitalWrite(IN4,LOW);  // FR stop
  digitalWrite(IN5,LOW);  digitalWrite(IN6,LOW);  // BL stop
  digitalWrite(IN7,LOW);  digitalWrite(IN8,HIGH); // BR backward
}
