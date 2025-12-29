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
py::array as_numpy_buffer(T* ptr, size_t count, py::object owner = py::none()) {
	static_assert(std::is_standard_layout_v<T>, "T must be standard layout");
	static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");

	return py::array(
		py::buffer_info(
			ptr,                              // pointer to data
			sizeof(T),                        // size of one element
			py::format_descriptor<uint8_t>::format(), // raw bytes
			1,                                // 1D array
			{ count * sizeof(T) },            // total bytes
			{ sizeof(T) }                     // stride = sizeof(T)
		),
		owner                                // keeps memory alive
	);
}


