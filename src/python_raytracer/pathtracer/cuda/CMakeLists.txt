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

set(CUDA_FLAGS_DEBUG "-O0;-G;-lineinfo")
foreach(debug_flag IN ITEMS ${CUDA_FLAGS_DEBUG})
    target_compile_options(pathtracer_ptx PRIVATE 
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:${debug_flag}>
    )
endforeach()

target_compile_definitions(pathtracer_ptx PRIVATE "$<$<CONFIG:DEBUG>:DEBUG>")

set_target_properties(pathtracer_ptx PROPERTIES
    CUDA_PTX_COMPILATION ON
)

add_custom_target(load_ptx ALL
COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/ptx/$<CONFIG>"
COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_OBJECTS:pathtracer_ptx> "${CMAKE_BINARY_DIR}/ptx/$<CONFIG>"
COMMAND_EXPAND_LISTS
DEPENDS pathtracer_ptx)


