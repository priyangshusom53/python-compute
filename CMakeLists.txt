cmake_minimum_required(VERSION 3.15)
project(pathtracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development)

find_package(pybind11 REQUIRED)

# c_python_types
add_library(cpython INTERFACE)

target_include_directories(cpython INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src/python_raytracer/c_functions/c_python)


# ---- BVH module ----
pybind11_add_module(_bvh
   src/python_raytracer/c_functions/bvh/_bvh.cpp
)

target_link_libraries(_bvh PRIVATE
   pybind11::module
   cpython
)

# ---- AABB module ----
pybind11_add_module(_aabb 
   src/python_raytracer/c_functions/aabb.cpp
)
target_link_libraries(_aabb PRIVATE 
   pybind11::module 
   cpython)

add_subdirectory(src/pathtracer ${CMAKE_BINARY_DIR}/pathtracer)
# target_link_libraries(pathtracer PRIVATE pybind11::module cpython)
pybind11_add_module(_pathtracer 
   src/pathtracer/src/python_bindings.cpp
)
# set_source_files_properties(
#     src/pathtracer/src/python_bindings.cpp
#     PROPERTIES LANGUAGE CUDA
# )

# set_target_properties(_pathtracer PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
# )
# find_package(CUDAToolkit REQUIRED)
target_link_libraries(_pathtracer PRIVATE pathtracer pybind11::module cpython )

install(TARGETS _pathtracer
   LIBRARY DESTINATION python_raytracer
   RUNTIME DESTINATION python_raytracer
)

# ---- Install into python package ----
install(TARGETS _bvh _aabb
   LIBRARY DESTINATION python_raytracer/c_functions
   RUNTIME DESTINATION python_raytracer/c_functions
)


