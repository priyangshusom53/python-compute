cmake_minimum_required(VERSION 3.15)
project(python_raytracer LANGUAGES CXX)

# ---- C++ standard ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Python ----
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# ---- pybind11 ----
find_package(pybind11 REQUIRED)


# ---- BVH module ----
pybind11_add_module(_bvh
   src/python_raytracer/c_functions/bvh/_bvh.cpp
)

target_link_libraries(_bvh PRIVATE
   pybind11::module
)

# ---- Install into python package ----
install(TARGETS _bvh
   LIBRARY DESTINATION python_raytracer/c_functions
   RUNTIME DESTINATION python_raytracer/c_functions
)
