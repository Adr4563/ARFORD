cmake_minimum_required(VERSION 3.8)
project(lora_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Un mensaje/servicio por cada frontera real entre lora_brain y los nodos
# de hardware de lora_drivers -- ver el README de este workspace para el
# porqué de cada uno (qué reemplaza del proyecto monolítico original).
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/UserInput.msg"
  "msg/FaceCommand.msg"
  "msg/MotionCommand.msg"
  "srv/Speak.srv"
  "srv/PlayMusic.srv"
  "srv/DetectEmotion.srv"
  "srv/IsVoiceClientConnected.srv"
)

ament_export_dependencies(rosidl_default_runtime)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
