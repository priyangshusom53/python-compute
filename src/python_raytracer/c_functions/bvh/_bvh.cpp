
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <numpy_c.h>

namespace py = pybind11;

struct AABB {
    double min[3];
    double max[3];
};

auto build(c_numpy_arr<double> aabbs, int32_t maxTrisinNode) {

}



PYBIND11_MODULE(_bvh, mod)
{
    mod.def("add", [](int i, int j)
         { return i + j; }, "A function which adds two numbers");
}