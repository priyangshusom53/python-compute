cmake_minimum_required(VERSION 3.15)
project(python_raytracer LANGUAGES CXX)

# ---- C++ standard ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Python ----
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# ---- pybind11 ----
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

# ---- Install into python package ----
install(TARGETS _bvh _aabb
   LIBRARY DESTINATION python_raytracer/c_functions
   RUNTIME DESTINATION python_raytracer/c_functions
)
