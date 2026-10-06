cmake_minimum_required(VERSION 3.8)
project(lora_bringup)

find_package(ament_cmake REQUIRED)

# Paquete de solo-launch: ament_cmake alcanza (no hay código Python propio
# que instalar como librería, solo archivos de datos).
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
