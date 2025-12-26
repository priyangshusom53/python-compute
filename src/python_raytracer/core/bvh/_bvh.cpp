
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

PYBIND11_MODULE(_bvh, m)
{
   m.def("add", [](int i, int j)
         { return i + j; }, "A function which adds two numbers");
}