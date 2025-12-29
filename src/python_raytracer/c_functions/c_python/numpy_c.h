#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>

namespace py = pybind11;

template<typename dtype>
using c_numpy_arr = py::array_t<dtype>;

using ssize_t = py::ssize_t;
using shape_t = std::vector<py::ssize_t>;
using stride_t = std::vector<py::ssize_t>;

template<typename dtype>
inline c_numpy_arr<dtype> create_numpy_arr(
	const int ndim, 
	ssize_t* shape) {

	shape_t arr_shape(ndim);
	stride_t arr_stride(ndim);
	py::ssize_t stride = sizeof(dtype);
	for (int i = ndim-1; i >=0; --i) {
		arr_shape[i] = shape[i];

		arr_stride[i] = stride;
		stride *= shape[i];
	}

	c_numpy_arr<dtype> arr(arr_shape, arr_stride);

	return arr;
}

template<typename dtype, py::ssize_t ndim=1>
inline auto get_mutable_reference(c_numpy_arr<dtype>& arr) {

	if (arr.ndim() != ndim)
		throw std::runtime_error("Dimension mismatch");
	auto ref = arr.mutable_unchecked<ndim>();
	return ref;
}

template <typename T>
py::array as_numpy_buffer(
    T* ptr,
    size_t count,
    py::object owner)
{
    static_assert(std::is_standard_layout_v<T>);
    static_assert(std::is_trivially_copyable_v<T>);

    // 2D byte array: (N, sizeof(T))
    std::vector<ssize_t> shape = {
        static_cast<ssize_t>(count),
        static_cast<ssize_t>(sizeof(T))
    };

    std::vector<ssize_t> strides = {
        static_cast<ssize_t>(sizeof(T)),
        1
    };

    return py::array(
        py::buffer_info(
            ptr,
            1,  // itemsize = 1 byte (uint8)
            py::format_descriptor<uint8_t>::format(),
            2,  // ndim
            shape,
            strides
        ),
        owner
    );
}


