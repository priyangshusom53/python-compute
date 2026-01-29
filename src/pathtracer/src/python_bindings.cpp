#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "vector.h"
#include "point.h"
#include "normal.h"
#include "bounds.h"
#include "transformation.h"
#include "mesh.h"
#include "bvh.h"

#include<vector>
#include <stdexcept>

namespace py = pybind11;

template<typename AttrType, typename DType>
static std::vector<AttrType> to_type3_array(py::array_t<DType>& np_array);

template<typename AttrType, typename DType>
static std::vector<AttrType> to_type2_array(py::array_t<DType>& np_array);

template<typename DType>
static std::vector<DType> to_type_array(py::array_t<DType>& np_array);

static std::vector<Bounds3f> to_bounds3f_array(py::array_t<float>& np_array);

static TriangleMesh MakeTriangleMesh(
	int nTriangles,
	std::vector<int> indices,
	int nVertices,
	std::vector<Point3f> positions,
	std::vector<Normal3f> normals,
	std::vector<Vector2f> uv,
	std::vector<Bounds3f> triBounds,
	Transform ObjectToWorld,
	int materialIdx,
	int handedness
) {
	TriangleMesh mesh = TriangleMesh(
		nTriangles,
		indices,
		nVertices,
		positions,
		normals,
		uv,
		triBounds,
		ObjectToWorld,
		materialIdx,
		handedness);

	return mesh;
}

template<typename AttrType, typename DType>
static std::vector<AttrType> to_type3_array(py::array_t<DType>& np_array) {
	if (np_array.ndim() != 2 || np_array.shape(1) != 3 ||
		!(np_array.flags() & py::array::c_style)) {
		throw std::runtime_error("Expected numpy C_CONTIGUOUS array of shape (N, 3)");
	}

	const size_t size = np_array.shape(0);
	const DType* data = np_array.data();

	std::vector<AttrType> out;
	out.reserve(size);

	for (int i = 0; i < size; ++i) {
		out.emplace_back(
			data[3 * i + 0],
			data[3 * i + 1],
			data[3 * i + 2]
		);
	}

	return out;
}

template<typename AttrType, typename DType>
static std::vector<AttrType> to_type2_array(py::array_t<DType>& np_array) {
	if (np_array.ndim() != 2 || np_array.shape(1) != 2 ||
		!(np_array.flags() & py::array::c_style)) {
		throw std::runtime_error("Expected numpy C_CONTIGUOUS array of shape (N, 2)");
	}

	const size_t size = np_array.shape(0);
	const DType* data = np_array.data();

	std::vector<AttrType> out;
	out.reserve(size);

	for (int i = 0; i < size; ++i) {
		out.emplace_back(
			data[2 * i + 0],
			data[2 * i + 1]
		);
	}

	return out;
}

static std::vector<Bounds3f> to_bounds3f_array(py::array_t<float>& np_array) {
	if (np_array.ndim() != 3 || np_array.shape(1) != 2 || 
		np_array.shape(2) != 3 || !(np_array.flags() & py::array::c_style)) {
		throw std::runtime_error("Expected numpy C_CONTIGUOUS array of shape (N, 2, 3)");
	}

	const size_t size = np_array.shape(0);
	const float* data = np_array.data();

	std::vector<Bounds3f> out;
	out.reserve(size);

	for (int i = 0; i < size; ++i) {
		out.emplace_back(
			Point3f(
				data[6 * i + 0],
				data[6 * i + 1],
				data[6 * i + 2]
			),
			Point3f(
				data[6 * i + 3],
				data[6 * i + 4],
				data[6 * i + 5]
			)
		);
	}

	return out;
}

template<typename DType>
static std::vector<DType> to_type_array(py::array_t<DType>& np_array) {
	if (np_array.ndim() != 1 || !(np_array.flags() & py::array::c_style)) {
		throw std::runtime_error("Expected numpy C_CONTIGUOUS array of shape (N,)");
	}

	const size_t size = np_array.shape(0);
	const DType* data = np_array.data();

	std::vector<DType> out;
	out.reserve(size);

	for (int i = 0; i < size; ++i) {
		out.emplace_back(data[i]);
	}

	return out;
}

void bind_np_array(py::module_& m) {
	m.def("to_vec3f_array", &to_type3_array<Vector3f, float>);
	m.def("to_vec3i_array", &to_type3_array<Vector3i, int>);
	m.def("to_vec2f_array", &to_type2_array<Vector2f, float>);
	m.def("to_point3f_array", &to_type3_array<Point3f, float>);
	m.def("to_normal3f_array", &to_type3_array<Normal3f, float>);
	m.def("to_bounds3f_array", &to_bounds3f_array);
	m.def("to_int_array", &to_type_array<int>);
}
void bind_transform(py::module_& m) {
	py::class_<Matrix4f>(m, "Matrix4f")
		.def(py::init<const float*>())
		.def_static("Identity", &Matrix4f::Identity);

	py::class_<Transform>(m, "Transform")
		.def(py::init<const Matrix4f&>())
		.def_static("Identity", &Transform::Identity);
}
void bind_mesh(py::module_& m) {
	py::class_<TriangleMesh>(m, "TriangleMesh")
		.def(py::init(&MakeTriangleMesh))
		.def("SetPositions", &TriangleMesh::SetPositions)
		.def("SetNormals", &TriangleMesh::SetNormals)
		.def("SetTextureCoords", &TriangleMesh::SetTextureCoords)
		.def("HasTextureCoords", &TriangleMesh::HasTextureCoords)
		.def("SetTransform", &TriangleMesh::SetTransform)
		.def("TransformMeshObjectSpace", &TriangleMesh::TransformMeshObjectSpace)
		.def("TransformMeshWorldSpace", &TriangleMesh::TransformMeshWorldSpace);
}

void bind_bvh(py::module_& m) {
	py::class_<BVHAccel>(m, "BVHAccel")
		.def(py::init<std::vector<Triangle>,
			int, SplitMethod>());
}

PYBIND11_MODULE(pathtracer, m) {
	auto m_np = m.def_submodule("np", "NumPy conversion utilities");
	bind_np_array(m_np);

	auto m_transform = m.def_submodule("transform", "Math & transforms");
	bind_transform(m_transform);

	auto m_mesh = m.def_submodule("mesh", "Mesh data structures");
	bind_mesh(m_mesh);
}

