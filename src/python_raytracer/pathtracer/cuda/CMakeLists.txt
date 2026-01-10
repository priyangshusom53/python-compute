cmake_minimum_required(VERSION 3.15)
project(pathtracer LANGUAGES CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES native)

set(PROJECT_SOURCES
   trace.cu
   intersect.cu
)

add_library(pathtracer_ptx OBJECT
    ${PROJECT_SOURCES}
)

target_include_directories(pathtracer_ptx PRIVATE include)

target_compile_definitions(pathtracer_ptx PRIVATE "$<$<CONFIG:DEBUG>:DEBUG>")

set_target_properties(pathtracer_ptx PROPERTIES
    CUDA_PTX_COMPILATION ON
)

add_custom_target(load_ptx ALL
COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/ptx"
COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_OBJECTS:pathtracer_ptx> "${CMAKE_BINARY_DIR}/$<CONFIG>/ptx"
COMMAND_EXPAND_LISTS
DEPENDS pathtracer_ptx)


