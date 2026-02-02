#ifndef MESH_H
#define MESH_H

#include "vector.h"
#include "normal.h"
#include "point.h"
#include "bounds.h"
#include "transformation.h"

#include<vector> // C++ vector array
#include<string>
#include<stdexcept>

constexpr const int LEFT_HANDED = 0;
constexpr const int RIGHT_HANDED = 1;

//class TriangleMesh {
//
//public:
//	int nTriangles;
//	std::vector<int> indices;
//	int nVertices;
//	std::vector<Point3f> positions;
//	std::vector<Normal3f> normals;
//	bool hasTexCoords;
//	std::vector<Vector2f> uvs;
//	std::vector<Bounds3f> triangleBounds; // world space triangle bounds
//	Transform ObjectToWorld;
//	int materialIndex;
//	int handedness;
//	TriangleMesh(
//		int nTriangles, 
//		const int indices[], 
//		int nVertices,
//		const float positions[],
//		const float ObjectToWorldMatrix[] = nullptr,
//		const float normals[] = nullptr,
//		const float uvs[] = nullptr, 
//		const float triangleBounds[] = nullptr, 
//		int materialIndex = 0, 
//		int handedness = LEFT_HANDED);
//	TriangleMesh(
//		int nTriangles,
//		const std::vector<int>& indices, 
//		int nVertices, 
//		const std::vector<Point3f>& positions,
//		const std::vector<Normal3f>& normals = std::vector<Normal3f>(),
//		const std::vector<Vector2f>& uvs = std::vector<Vector2f>(),
//		const std::vector<Bounds3f>& triangleBounds = std::vector<Bounds3f>(),
//		const Transform ObjectToWorld = Transform::Identity(),
//		int materialIndex = 0,  
//		int handedness = LEFT_HANDED);
//	TriangleMesh(const TriangleMesh& mesh) = delete;
//	TriangleMesh& operator=(const TriangleMesh& mesh) = delete;
//	TriangleMesh(TriangleMesh&& mesh) noexcept = default;
//	TriangleMesh& operator=(TriangleMesh&& mesh) noexcept = default;
//	void SetIndices(int nTriangles, const int indices[]);
//	void SetPositions(int nVertices, const float positions[]);
//	void CalculateNormals();
//	void SetNormals(int nVertices, const float normals[]);
//	bool HasTextureCoords() const;
//	void SetTextureCoords(int nVertices, const float uvs[]);
//	void SetTransform(int nVertices, const float localToWorldMatrix[]);
//	void SetMaterialIdx(int materialIndex);
//	// Bakes transform into vertex positions and updates triangle bounds
//	void TransformMeshObjectSpace(const Transform& t);
//	// Modifies ObjectToWorld transform
//	void TransformMeshWorldSpace(const Transform& t);
//	std::vector<Point3f> WorldSpacePositions() const;
//	std::vector<Normal3f> WorldSpaceNormals() const;
//	TriangleMesh& ChangeHandedness(int handedness);
//private:
//	void CalculateTriangleBounds();
//	void FlipWindingOrder();
//	void FlipZ();
//};

class TriangleMesh {

public:
	Transform ObjectToWorld;
	int nTriangles;
	std::vector<Vector3i> indices;
	int nVertices;
	std::vector<Point3f> positions;
	std::vector<Normal3f> normals;
	std::vector<Vector2f> uvs;
	int materialIndex;
	int handedness;
	TriangleMesh(
		const std::vector<Point3f>& positions,
		const std::vector<Vector3i>& indices = std::vector<Vector3i>(),
		const Transform& ObjectToWorld = Transform::Identity(),
		int handedness = LEFT_HANDED,
		const std::vector<Normal3f>& normals = std::vector<Normal3f>(),
		const std::vector<Vector2f>& uvs = std::vector<Vector2f>(),
		int materialIndex = 0);
	TriangleMesh(const TriangleMesh& mesh) = delete;
	TriangleMesh& operator=(const TriangleMesh& mesh) = delete;
	TriangleMesh(TriangleMesh&& mesh) noexcept = default;
	TriangleMesh& operator=(TriangleMesh&& mesh) noexcept = default;
	/*void SetIndices(int nTriangles, const int indices[]);*/
	/*void SetPositions(int nVertices, const std::vector<Point3f>& positions);*/
	std::vector<Bounds3f> GetTriangleObjectBounds() const;
	std::vector<Bounds3f> GetTriangleWorldBounds() const;
	void CalculateNormals();
	bool HasNormals() const ;
	void SetNormals(const std::vector<Normal3f>& normals);
	bool HasTextureCoords() const;
	void SetTextureCoords(const std::vector<Vector2f>& uvs);
	void SetTransform(const Transform& ObjectToWorld);
	void SetMaterialIdx(int materialIndex);
	// Bakes transform into vertex positions and updates triangle bounds
	void TransformMeshObjectSpace(const Transform& t);
	// Modifies ObjectToWorld transform
	void TransformMeshWorldSpace(const Transform& t);
	std::vector<Point3f> WorldSpacePositions() const;
	std::vector<Normal3f> WorldSpaceNormals() const;
	TriangleMesh& ChangeHandedness(int handedness);
private:
	void FlipWindingOrder();
	void FlipZ();
};

struct CUDA_ALIGN(16) GPUTriangleMesh {
	int nTriangles;
	int firstTriangleIdx;
	int nVertices;
	int firstPositionIdx;
	int hasNormals;
	int firstNormalIdx;
	int hasUV;
	int firstUVIdx;
	Transform ObjectToWorld;
	int materialIndex;
	int pad[3];
};

#endif