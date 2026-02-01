#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "buffer.h"
#include "vector.h"
#include "point.h"
#include "normal.h"
#include "bounds.h"
#include "transformation.h"
#include "mesh.h"
#include "bvh.h"
#include "soa.h"
#include "trace.h"

#include<vector>
#include <stdexcept>

namespace py = pybind11;

template<typename AttrType, typename DType>
static std::vector<AttrType> to_type3_array(py::array_t<DType, py::array::c_style | py::array::forcecast>& np_array);

template<typename AttrType, typename DType>
static std::vector<AttrType> to_type2_array(py::array_t<DType, py::array::c_style | py::array::forcecast>& np_array);

template<typename DType>
static std::vector<DType> to_type_array(py::array_t<DType, py::array::c_style | py::array::forcecast>& np_array);

static std::vector<Bounds3f> to_bounds3f_array(py::array_t<float, py::array::c_style | py::array::forcecast>& np_array);

template<typename AttrType, typename DType>
static std::vector<AttrType> to_type3_array(py::array_t<DType, py::array::c_style | py::array::forcecast>& np_array) {
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
static std::vector<AttrType> to_type2_array(py::array_t<DType, py::array::c_style | py::array::forcecast>& np_array) {
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

static std::vector<Bounds3f> to_bounds3f_array(py::array_t<float, py::array::c_style | py::array::forcecast>& np_array) {
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

template<typename SType>
static StructuredBuffer<SType, BufferType::GPU_BUFFER> to_struct_buffer_gpu(py::array_t<SType>& np_array) {
	if (np_array.ndim() != 1)
		throw std::runtime_error("Expected 1D numpy array of struct");

	size_t n = np_array.shape(0);
	const SType* src = np_array.data();

	StructuredBuffer<SType, BufferType::CPU_BUFFER> cpu(n);
	for (size_t i = 0; i < n; ++i)
		cpu[i] = src[i];

	StructuredBuffer<SType, BufferType::GPU_BUFFER> gpu(n);
	CopyData(cpu, gpu);

	return gpu;
}

template<typename SType>
static py::array_t<SType> from_struct_buffer_gpu(const StructuredBuffer<SType,GPU_BUFFER>& gpu_buff) {

	size_t n = gpu_buff.size();

	auto arr = py::array_t<SType>(n);
	SType* dst = arr.mutable_data();

	cudaError_t err = cudaMemcpy(
		(void*)dst,
		(void*)gpu_buff.data(),
		n * sizeof(SType),
		cudaMemcpyDeviceToHost
	);

	if (err != cudaSuccess)
		throw std::runtime_error(cudaGetErrorString(err));

	return arr;
}

template<typename DType>
static std::vector<DType> to_type_array(py::array_t<DType, py::array::c_style | py::array::forcecast>& np_array) {
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
	m.def("to_ray_buffer_gpu", &to_struct_buffer_gpu<Ray>);
	m.def("from_ray_buffer_gpu", &from_struct_buffer_gpu<Ray>);
	m.def("to_img_buffer_gpu", &to_struct_buffer_gpu<Vector4f>);
	m.def("from_img_buffer_gpu", &from_struct_buffer_gpu<Vector4f>);
}
void bind_transform(py::module_& m) {
	py::class_<Matrix4f>(m, "Matrix4f")
		.def(py::init<const float*>())
		.def_static("Identity", &Matrix4f::Identity);

	py::class_<Transform>(m, "Transform")
		.def(py::init<const Matrix4f&>())
		.def_static("Identity", &Transform::Identity);
}

void bind_struct_buffer(py::module_& m) {
	py::class_<StructuredBuffer<Vector4f, CPU_BUFFER>>(m, "StructBufV4f_CPU");
	py::class_<StructuredBuffer<Point3f, CPU_BUFFER>>(m, "StructBufP3f_CPU");
	py::class_<StructuredBuffer<Normal3f, CPU_BUFFER>>(m, "StructBufN3f_CPU");
	py::class_<StructuredBuffer<Vector2f, CPU_BUFFER>>(m, "StructBufV2f_CPU");

	py::class_<StructuredBuffer<Vector4f, GPU_BUFFER>>(m, "StructBufV4f_GPU");
	py::class_<StructuredBuffer<Point3f, GPU_BUFFER>>(m, "StructBufP3f_GPU");
	py::class_<StructuredBuffer<Normal3f, GPU_BUFFER>>(m, "StructBufN3f_GPU");
	py::class_<StructuredBuffer<Vector2f, GPU_BUFFER>>(m, "StructBufV2f_GPU");
}

void bind_mesh(py::module_& m) {
	py::class_<TriangleMesh, std::shared_ptr<TriangleMesh>>(m, "TriangleMesh")
		.def(py::init<
			int,
			const std::vector<Vector3i>&,
			int,
			const std::vector<Point3f>&,
			const Transform&,
			int,
			bool,
			const std::vector<Normal3f>&,
			const std::vector<Vector2f>&,
			int
		>())
		.def("HasNormals", &TriangleMesh::HasNormals)
		.def("SetNormals", &TriangleMesh::SetNormals)
		.def("SetTextureCoords", &TriangleMesh::SetTextureCoords)
		.def("HasTextureCoords", &TriangleMesh::HasTextureCoords)
		.def("SetTransform", &TriangleMesh::SetTransform)
		.def("TransformMeshObjectSpace", &TriangleMesh::TransformMeshObjectSpace)
		.def("TransformMeshWorldSpace", &TriangleMesh::TransformMeshWorldSpace);

	m.def("GetTriangles", &GetTriangles);
}

void bind_bvh(py::module_& m) {
	py::class_<BVHAccel>(m, "BVHAccel")
		.def(py::init<
			std::vector<std::shared_ptr<Triangle>>,
			int,
			SplitMethod>()
		)
		.def_readonly("triangles", &BVHAccel::triangles)
		.def_readonly("linearNodes", &BVHAccel::nodes);
}

void bind_soa(py::module_& m) {
	py::class_<SOA>(m, "SOA")
		.def(py::init<
			const std::vector<std::shared_ptr<TriangleMesh>>&,
			const std::vector<std::shared_ptr<Triangle>>&,
			const StructuredBuffer<LinearBVHNode, BufferType::CPU_BUFFER>&>()
		)
		.def_readonly("indices", &SOA::indices)
		.def_readonly("positions", &SOA::positions)
		.def_readonly("normals", &SOA::normals)
		.def_readonly("uvs", &SOA::uvs)
		.def_readonly("triangles", &SOA::triangles)
		.def_readonly("gpuMeshes", &SOA::meshes)
		.def_readonly("nodes", &SOA::nodes)
		.def_readonly("d_indices", &SOA::d_indices)
		.def_readonly("d_positions", &SOA::d_positions)
		.def_readonly("d_normals", &SOA::d_normals)
		.def_readonly("d_uvs", &SOA::d_uvs)
		.def_readonly("d_triangles", &SOA::d_triangles)
		.def_readonly("d_gpuMeshes", &SOA::d_meshes)
		.def_readonly("d_nodes", &SOA::d_nodes);
}

void bind_render(py::module_& m) {
	m.def("Render", &Render);
}

PYBIND11_MODULE(pathtracer, m) {
	//	1. Convert numpy arrays containing mesh data to C++ vector arrays
	auto m_np = m.def_submodule("np", "NumPy conversion utilities");
	bind_np_array(m_np);

	//	2. Transform operations C++
	auto m_transform = m.def_submodule("transform", "Math & transforms");
	bind_transform(m_transform);

	bind_struct_buffer(m);

	//	3. Construct TriangleMesh from vector arrays of mesh data
	auto m_mesh = m.def_submodule("mesh", "Mesh data structures");
	bind_mesh(m_mesh);

	//	4. Pass triangle data to make BVH
	auto m_bvh = m.def_submodule("accels");
	bind_bvh(m_bvh);

	//	5. Create SOA and upload to GPU
	auto m_soa = m.def_submodule("soa", "Creates SOA for GPU upload");
	bind_soa(m_soa);

	//	6. Call Render function with GPU buffers to render
	auto m_render = m.def_submodule("render");
	bind_render(m_render);
}

