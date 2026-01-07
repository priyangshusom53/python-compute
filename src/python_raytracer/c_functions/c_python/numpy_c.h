#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>
#include <initializer_list>

namespace py = pybind11;

using pyobject = py::object;

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

template<typename CType, typename NpType>
py::array as_numpy_buffer(CType* ptr, 
    std::initializer_list<ssize_t> shape, pyobject owner) {

    static_assert(std::is_standard_layout_v<CType>);
    static_assert(std::is_trivially_copyable_v<CType>);

    ssize_t _nDim = shape.size();
    shape_t _shape(shape);
    stride_t _strides(_nDim);
    ssize_t _pyElemSize = sizeof(NpType);

    _strides[_nDim-1] = _pyElemSize;
    for (size_t i = _nDim - 2; i >= 0; ++i) {
        _strides[i] = _shape[i+1] * _strides[i + 1];
    }

    return py::array(
        py::buffer_info(
            ptr,
            _pyElemSize,
            py::format_descriptor<NpType>::format(),
            _nDim,
            _shape,
            _strides), owner);
}

template<typename StructT>
py::array as_numpy_byte_buffer(StructT* ptr, ssize_t count, pyobject owner) {
    
    return as_numpy_buffer<StructT, uint8_t>(ptr, 
        {count* static_cast<ssize_t>(sizeof(StructT))}, owner);
}

