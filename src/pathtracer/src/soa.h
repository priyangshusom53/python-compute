#ifndef SOA_H
#define SOA_H

#include "cudadefines.h"
#include "buffer.h"
#include "vector.h"
#include "point.h"
#include "normal.h"
#include "triangle.h"
#include "mesh.h"

#include<vector>
#include<memory>

struct SOA {
	StructuredBuffer<Vector3i,BufferType::CPU_BUFFER> indices;
	StructuredBuffer<Point3f, BufferType::CPU_BUFFER> positions;
	StructuredBuffer<Normal3f, BufferType::CPU_BUFFER> normals;
	StructuredBuffer<Vector2f, BufferType::CPU_BUFFER> uvs;
	StructuredBuffer<Triangle, BufferType::CPU_BUFFER> triangles;
	StructuredBuffer<GPUTriangleMesh, BufferType::CPU_BUFFER> meshes;

	SOA(const std::vector<TriangleMesh>& triangleMeshes,
		const std::vector<std::shared_ptr<Triangle>>& triangles);
};

INLINE SOA::SOA(const std::vector<TriangleMesh>& triMeshes, 
	const std::vector<std::shared_ptr<Triangle>>& triangles) {
	int nMeshes = triMeshes.size();
	if (nMeshes == 0)
		return;

	int triangleOffset = 0, positionsOffset = 0, 
		normalsOffset = 0, UVOffset = 0;

	meshes = StructuredBuffer<GPUTriangleMesh,BufferType::CPU_BUFFER>(nMeshes);
	for (int i = 0; i < nMeshes; ++i) {
		const TriangleMesh& triMesh = triMeshes[i];

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
		const TriangleMesh& triMesh = triMeshes[i];

		GPUTriangleMesh& gpuMesh = meshes[i];

		for(int j=0; j < triMesh.nTriangles; ++j)
			indices[triangleOffset+j] = triMesh.indices[j];
		triangleOffset += triMesh.nTriangles;

		for (int j = 0; j < triMesh.nVertices; ++j)
			positions[positionsOffset + j] += triMesh.positions[j];
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
}

#endif