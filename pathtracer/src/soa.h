#ifndef SOA_H
#define SOA_H

#include "cudadefines.h"
#include "buffer.h"
#include "vector.h"
#include "point.h"
#include "normal.h"
#include "triangle.h"
#include "mesh.h"
#include "bvh.h"

#include<vector>
#include<memory>

struct SOA {
	StructuredBuffer<Vector3i,BufferType::CPU_BUFFER> indices;
	StructuredBuffer<Point3f, BufferType::CPU_BUFFER> positions;
	StructuredBuffer<Normal3f, BufferType::CPU_BUFFER> normals;
	StructuredBuffer<Vector2f, BufferType::CPU_BUFFER> uvs;
	StructuredBuffer<Triangle, BufferType::CPU_BUFFER> triangles;
	StructuredBuffer<GPUTriangleMesh, BufferType::CPU_BUFFER> meshes;
	StructuredBuffer<LinearBVHNode, BufferType::CPU_BUFFER> nodes;

	StructuredBuffer<Vector3i, BufferType::GPU_BUFFER> d_indices;
	StructuredBuffer<Point3f, BufferType::GPU_BUFFER> d_positions;
	StructuredBuffer<Normal3f, BufferType::GPU_BUFFER> d_normals;
	StructuredBuffer<Vector2f, BufferType::GPU_BUFFER> d_uvs;
	StructuredBuffer<Triangle, BufferType::GPU_BUFFER> d_triangles;
	StructuredBuffer<GPUTriangleMesh, BufferType::GPU_BUFFER> d_meshes;
	StructuredBuffer<LinearBVHNode, BufferType::GPU_BUFFER> d_nodes;

	SOA(const std::vector<std::shared_ptr<TriangleMesh>>& triangleMeshes,
		const std::vector<std::shared_ptr<Triangle>>& triangles,
		const StructuredBuffer<LinearBVHNode, BufferType::CPU_BUFFER>& linearNodes);
private:
	void CopyToGPU();
};

INLINE SOA::SOA(const std::vector<std::shared_ptr<TriangleMesh>>& triMeshes, 
	const std::vector<std::shared_ptr<Triangle>>& _triangles,
	const StructuredBuffer<LinearBVHNode, BufferType::CPU_BUFFER>& linearNodes) {
	int nMeshes = triMeshes.size();
	if (nMeshes == 0)
		return;

	int triangleOffset = 0, positionsOffset = 0, 
		normalsOffset = 0, UVOffset = 0;

	meshes = StructuredBuffer<GPUTriangleMesh,BufferType::CPU_BUFFER>(nMeshes);
	for (int i = 0; i < nMeshes; ++i) {
		const TriangleMesh& triMesh = *triMeshes[i];

		GPUTriangleMesh& gpuMesh = meshes[i];

		gpuMesh.nTriangles = triMesh.nTriangles;
		gpuMesh.firstTriangleIdx = triangleOffset;
		triangleOffset += triMesh.nTriangles;

		gpuMesh.nVertices = triMesh.nVertices;
		gpuMesh.firstPositionIdx = positionsOffset;
		positionsOffset += triMesh.nVertices;

		if (triMesh.HasNormals()) {
			gpuMesh.hasNormals = 1;
			gpuMesh.firstNormalIdx = normalsOffset;
			normalsOffset += triMesh.nVertices;
		}

		if (triMesh.HasTextureCoords()) {
			gpuMesh.hasUV = 1;
			gpuMesh.firstUVIdx = UVOffset;
			UVOffset += triMesh.nVertices;
		}

		gpuMesh.ObjectToWorld = triMesh.ObjectToWorld;
		gpuMesh.materialIndex = triMesh.materialIndex;

	}

	indices = StructuredBuffer<Vector3i, BufferType::CPU_BUFFER>(triangleOffset);
	positions = StructuredBuffer<Point3f, BufferType::CPU_BUFFER>(positionsOffset);
	normals = StructuredBuffer<Normal3f, BufferType::CPU_BUFFER>(normalsOffset);
	uvs = StructuredBuffer<Vector2f, BufferType::CPU_BUFFER>(UVOffset);

	triangleOffset = 0, positionsOffset = 0, normalsOffset = 0, 
		UVOffset = 0;
	for (int i = 0; i < nMeshes; ++i) {
		const TriangleMesh& triMesh = *triMeshes[i];

		GPUTriangleMesh& gpuMesh = meshes[i];

		for(int j=0; j < triMesh.nTriangles; ++j)
			indices[triangleOffset+j] = triMesh.indices[j];
		triangleOffset += triMesh.nTriangles;

		for (int j = 0; j < triMesh.nVertices; ++j)
			positions[positionsOffset + j] = triMesh.positions[j];
		positionsOffset += triMesh.nVertices;

		if (triMesh.HasNormals()) {
			for (int j = 0; j < triMesh.nVertices; ++j)
				normals[normalsOffset + j] = triMesh.normals[j];
			normalsOffset += triMesh.nVertices;
		}

		if (triMesh.HasTextureCoords()) {
			for (int j = 0; j < triMesh.nVertices; ++j)
				uvs[UVOffset + j] = triMesh.uvs[j];
			UVOffset += triMesh.nVertices;
		}
	}

	triangles = StructuredBuffer<Triangle, BufferType::CPU_BUFFER>(_triangles.size());
	for (int i = 0; i < triangles.size(); ++i) {
		triangles[i] = 
			Triangle(_triangles[i]->meshIdx, _triangles[i]->triangleIdx, _triangles[i]->worldBounds);
	}

	nodes = StructuredBuffer<LinearBVHNode, BufferType::CPU_BUFFER>(linearNodes.size());
	for (int i = 0; i < nodes.size(); ++i) {
		nodes[i] = linearNodes[i];
	}

	CopyToGPU();
}

INLINE void SOA::CopyToGPU() {
	d_indices = StructuredBuffer<Vector3i, BufferType::GPU_BUFFER>(indices.size());
	CopyData<Vector3i>(indices, d_indices);

	d_positions = StructuredBuffer<Point3f, BufferType::GPU_BUFFER>(positions.size());
	CopyData<Point3f>(positions, d_positions);

	d_normals = StructuredBuffer<Normal3f, BufferType::GPU_BUFFER>(normals.size());
	CopyData<Normal3f>(normals, d_normals);

	d_uvs = StructuredBuffer<Vector2f, BufferType::GPU_BUFFER>(uvs.size());
	CopyData<Vector2f>(uvs, d_uvs);

	d_triangles = StructuredBuffer<Triangle, BufferType::GPU_BUFFER>(triangles.size());
	CopyData<Triangle>(triangles, d_triangles);

	d_meshes = StructuredBuffer<GPUTriangleMesh, BufferType::GPU_BUFFER>(meshes.size());
	CopyData<GPUTriangleMesh>(meshes, d_meshes);

	d_nodes = StructuredBuffer<LinearBVHNode, BufferType::GPU_BUFFER>(nodes.size());
	CopyData<LinearBVHNode>(nodes, d_nodes);
}

#endif