cmake_minimum_required(VERSION 3.8)
project(romeo_bringup)

find_package(ament_cmake REQUIRED)

# Paquete de solo-launch: ament_cmake alcanza (no hay código Python propio
# que instalar como librería, solo archivos de datos).
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

# scripts/ va aparte para que conserve el permiso de ejecución:
# install(DIRECTORY) no lo preserva, y volumen_placa.sh se ejecuta
# directamente (también como ExecStart de un servicio systemd).
install(PROGRAMS scripts/volumen_placa.sh
  DESTINATION share/${PROJECT_NAME}/scripts
)

ament_package()
