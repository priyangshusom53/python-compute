
cmake_minimum_required(VERSION 3.18)
project(pathtracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

set(PROJECT_SOURCES
   src/cudadefines.h
   src/mathconstants.h
   src/trace.cu
   src/vector.h
   src/vector.cu
   src/point.h
   src/point.cu
   src/normal.h
   src/normal.cu
   src/ray.h
   src/ray.cu
   src/bounds.h
   src/bounds.cu
   src/matrix.h
   src/transformation.h
   src/buffer.h
   src/mesh.h
   src/mesh.cpp
   src/triangle.h
   src/bvh.h
   src/intersection.cu
   src/utils.h
)

# By default nvcc can't compile .cpp .h file, here Language property forces compilation of .cpp files using nvcc
# set_source_files_properties(${PROJECT_SOURCES} PROPERTIES
#                             LANGUAGE CUDA)

add_library(pathtracer STATIC
   ${PROJECT_SOURCES}
)

target_include_directories(pathtracer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

set_target_properties(pathtracer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

set(CUDA_FLAGS_DEBUG "-O0;-G;-lineinfo")
foreach(debug_flag IN ITEMS ${CUDA_FLAGS_DEBUG})
    target_compile_options(pathtracer PRIVATE 
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:${debug_flag}>
    )
endforeach()

target_compile_definitions(pathtracer PRIVATE "$<$<CONFIG:DEBUG>:DEBUG>")
