idf_component_register(
    SRCS "lora_audio_main.c"
    INCLUDE_DIRS "."
)

# Compilación de diagnóstico del micrófono:
#
#   idf.py -DSDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.diag" \
#          -DLORA_MODO_DIAG=1 build
#
# Sin esta línea, el -D de la orden se queda como variable de CMake y nunca
# llega al compilador de C, así que el #ifdef del código no se activaría y se
# construiría el firmware normal sin avisar.
if(LORA_MODO_DIAG)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE LORA_MODO_DIAG=1)
    message(STATUS "Lora: compilando en MODO DIAGNOSTICO (sin USB de audio)")
endif()
