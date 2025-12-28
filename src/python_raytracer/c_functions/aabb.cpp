
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <numpy_c.h>

#include <vector>

namespace py = pybind11;

py::tuple build_aabb_lines_numpy(py::array_t<double> aabbs)
{
    auto buf = aabbs.request();

    if (buf.ndim != 3)
        throw std::runtime_error("Expected 3D array");

    if (buf.ndim != 3 || buf.shape[1] != 2 || buf.shape[2] != 3)
        throw std::runtime_error("Expected shape (N, 2, 3)");

    if (buf.itemsize != sizeof(double))
        throw std::runtime_error("Expected float64");

    if (!(buf.strides[2] == sizeof(double) &&
        buf.strides[1] == 3 * sizeof(double)))
        throw std::runtime_error("Array must be C-contiguous");


    const py::ssize_t N = buf.shape[0];
    const double* aabb_ptr = static_cast<double*>(buf.ptr);

    // Allocate output NumPy arrays
    const int ndim = 2;
    ssize_t pointsShape[] = { N * 8, 3 };
    c_numpy_arr<double> points = create_numpy_arr<double>(ndim, pointsShape);

    ssize_t linesShape[] = { N * 12, 2 };
    c_numpy_arr<int32_t> lines = create_numpy_arr<int32_t>(ndim, linesShape);

    ssize_t colorsShape[] = { N * 12, 3 };
    c_numpy_arr<double> colors = create_numpy_arr<double>(ndim, colorsShape);

    auto p = points.mutable_unchecked<ndim>();
    auto l = lines.mutable_unchecked<ndim>();
    auto c = colors.mutable_unchecked<ndim>();

    py::ssize_t p_idx = 0;
    py::ssize_t l_idx = 0;

    static const int edges[12][2] = {
        {0,1},{1,2},{2,3},{3,0},
        {4,5},{5,6},{6,7},{7,4},
        {0,4},{1,5},{2,6},{3,7}
    };

    for (py::ssize_t i = 0; i < N; ++i) {
        const double* mn = aabb_ptr + i * 6;
        const double* mx = mn + 3;

        // corners
        double corners[8][3] = {
            {mn[0], mn[1], mn[2]},
            {mx[0], mn[1], mn[2]},
            {mx[0], mx[1], mn[2]},
            {mn[0], mx[1], mn[2]},
            {mn[0], mn[1], mx[2]},
            {mx[0], mn[1], mx[2]},
            {mx[0], mx[1], mx[2]},
            {mn[0], mx[1], mx[2]},
        };

        for (int k = 0; k < 8; ++k) {
            p(p_idx, 0) = corners[k][0];
            p(p_idx, 1) = corners[k][1];
            p(p_idx, 2) = corners[k][2];
            p_idx++;
        }

        int base = static_cast<int>(i * 8);

        for (int e = 0; e < 12; ++e) {
            l(l_idx, 0) = base + edges[e][0];
            l(l_idx, 1) = base + edges[e][1];

            c(l_idx, 0) = 1.0;
            c(l_idx, 1) = 0.0;
            c(l_idx, 2) = 0.0;

            l_idx++;
        }
    }

    return py::make_tuple(points, lines, colors);
}

PYBIND11_MODULE(_aabb, mod) {
    mod.def("build_aabb_wireframe_numpy", &build_aabb_lines_numpy,
        "Build LineSet arrays from AABB array (N,2,3)\
returns: (points(N*8, 3), lines(N*12,2), colors(N*12, 3))");
}

