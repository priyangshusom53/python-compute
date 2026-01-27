#include "mesh.h"
#include <stdexcept>

TriangleMesh::TriangleMesh(
	int nTriangles,
	const int indices[],
	int nVertices,
	const float positions[],
	const float ObjectToWorldMatrix[],
	const float normals[],
	const float uvs[],
	const float triangleBounds[],
	int materialIndex,
	int handedness
) {

	
	this->handedness = handedness;
	this->hasTexCoords = false;
	this->nTriangles = nTriangles;
	this->indices.reserve((3 * nTriangles));
	for (int i = 0; i < 3 * nTriangles; ++i) {
		this->indices.push_back(indices[i]);
	}
	this->nVertices = nVertices;
	this->positions.reserve(nVertices);
	for (int i = 0; i < 3 * nVertices; i += 3) {
		this->positions.push_back(Point3f(positions[i], positions[i + 1], positions[i + 2]));
	}
	if (ObjectToWorldMatrix) {
		this->ObjectToWorld = Transform(Matrix4f(ObjectToWorldMatrix));
	}
	else {
		this->ObjectToWorld = Transform::Identity();
	}

	if (normals) {
		this->normals.reserve(nVertices);
		for (int i = 0; i < 3 * nVertices; i += 3) {
			this->normals.push_back(Normal3f(normals[i], normals[i + 1], normals[i + 2]));
		}
	}
	else {
		CalculateNormals();
	}
		
	if(triangleBounds) {
		this->triangleBounds.reserve(nTriangles);
		for(int i = 0; i < 6 * nTriangles; i += 6) {
			Point3f pMin(triangleBounds[i], triangleBounds[i + 1], triangleBounds[i + 2]);
			Point3f pMax(triangleBounds[i + 3], triangleBounds[i + 4], triangleBounds[i + 5]);
			this->triangleBounds.push_back(Bounds3f(pMin, pMax));
		}
	}
	else {
		CalculateTriangleBounds();
	}
	
	if (uvs) {
		this->uvs.reserve(nVertices);
		for (int i = 0; i < 2 * nVertices; i += 2) {
			this->uvs.push_back(Vector2f(uvs[i], uvs[i + 1]));
		}
		this->hasTexCoords = true;
	}
	else {
		this->hasTexCoords = false;
	}
	if (handedness == RIGHT_HANDED) {
		ChangeHandedness(LEFT_HANDED);
	}
	this->materialIndex = materialIndex;
}

TriangleMesh::TriangleMesh(
	int nTriangles,
	const std::vector<int>& indices,
	int nVertices,
	const std::vector<Point3f>& positions,
	const std::vector<Normal3f>& normals,
	const std::vector<Vector2f>& uvs,
	const std::vector<Bounds3f>& triangleBounds,
	const Transform ObjectToWorld,
	int materialIndex,
	int handedness) {

	this->handedness = handedness;
	this->hasTexCoords = false;
	this->nTriangles = nTriangles;
	this->indices.reserve((3 * nTriangles));
	for (int i = 0; i < 3 * nTriangles; ++i) {
		this->indices.push_back(indices[i]);
	}
	this->nVertices = nVertices;
	this->positions.reserve(nVertices);
	for (int i = 0; i < nVertices; ++i) {
		this->positions.push_back(positions[i]);
	}

	this->ObjectToWorld = ObjectToWorld;
	if (!normals.empty()) {
		for(int i = 0; i < nVertices; ++i) {
			this->normals.push_back(normals[i]);
		}
	}
	else {
		CalculateNormals();
	}

	if (!triangleBounds.empty()) {
		for(int i = 0; i < nTriangles; ++i) {
			this->triangleBounds.push_back(triangleBounds[i]);
		}
	}
	else {
		CalculateTriangleBounds();
	}
	if (!uvs.empty()) {
		for(int i = 0; i < nVertices; ++i) {
			this->uvs.push_back(uvs[i]);
		}
		this->hasTexCoords = true;
	}
	else {
		this->hasTexCoords = false;
	}
	if (handedness == RIGHT_HANDED) {
		ChangeHandedness(LEFT_HANDED);
	}
	this->materialIndex = materialIndex;

}

void TriangleMesh::SetIndices(int nTriangles, const int indices[]) {
	if (this->nTriangles != nTriangles) {
		throw std::invalid_argument("Number of triangles does not match existing mesh.");
	}
	for (int i = 0; i < 3 * nTriangles; ++i) {
		this->indices[i] = indices[i];
	}
}

void TriangleMesh::SetPositions(int nVertices, const float positions[]) {
	if(this->nVertices != nVertices) {
		throw std::invalid_argument("Number of vertices does not match existing mesh.");
	}
	for(int i = 0; i < nVertices; ++i) {
		this->positions[i] = Point3f(positions[3*i], positions[3*i + 1], positions[3*i + 2]);
	}
}

void TriangleMesh::CalculateNormals() {
	this->normals.clear();
	this->normals.reserve(this->nVertices);
	this->normals.resize(this->nVertices, Normal3f(0.f, 0.f, 0.f));
	const std::vector<int>& indices = this->indices;
	for (int i = 0; i < this->nTriangles; ++i) {
		int i0 = indices[3 * i];
		int i1 = indices[3 * i + 1];
		int i2 = indices[3 * i + 2];
		const Point3f& p0 = this->positions[i0];
		const Point3f& p1 = this->positions[i1];
		const Point3f& p2 = this->positions[i2];
		const Vector3f e01 = p1 - p0;
		const Vector3f e02 = p2 - p0;
		Normal3f n = Normal3f(Vector3f::Normalize(Vector3f::Cross(e01, e02)));
		this->normals[i0] = n;
		this->normals[i1] = n;
		this->normals[i2] = n;
	}
}

void TriangleMesh::SetNormals(int nVertices, const float normals[]) {
	for(int i = 0; i < 3 * nVertices; i += 3) {
		this->normals.push_back(Normal3f(normals[i], normals[i + 1], normals[i + 2]));
	}
}

bool TriangleMesh::HasTextureCoords() const {
	return this->hasTexCoords;
}

void TriangleMesh::SetTextureCoords(int nVertices, const float uvs[]) {
	for(int i = 0; i < 2 * nVertices; i += 2) {
		this->uvs.push_back(Vector2f(uvs[i], uvs[i + 1]));
	}
}

void TriangleMesh::SetTransform(int nVertices, const float localToWorldMatrix[]) {
	this->ObjectToWorld = Transform(Matrix4f(localToWorldMatrix));
}

void TriangleMesh::SetMaterialIdx(int materialIndex) {
	this->materialIndex = materialIndex;
}

// Bakes transform into vertex positions and updates triangle bounds
void TriangleMesh::TransformMeshObjectSpace(const Transform& t) {
	for(int i = 0; i < this->nVertices; ++i) {
		this->positions[i] = Point3f(t.TransformPoint(this->positions[i]));
		this->normals[i] = Normal3f(t.TransformNormal(this->normals[i]));
	}
	CalculateTriangleBounds();
}

// Modifies ObjectToWorld transform
void TriangleMesh::TransformMeshWorldSpace(const Transform& t) {

	const Matrix4f newObjToWorldMat = Matrix4f::MatMul(t.matrix, ObjectToWorld.matrix);
	const Matrix4f newInvObjToWorldMat = Matrix4f::MatMul(ObjectToWorld.invMatrix,t.matrix);
	this->ObjectToWorld = Transform(newObjToWorldMat, newInvObjToWorldMat);
}

std::vector<Point3f> TriangleMesh::WorldSpacePositions() const {
	std::vector<Point3f> worldPositions;
	for(int i = 0; i < this->nVertices; ++i) {
		worldPositions.push_back(Point3f(this->ObjectToWorld.TransformPoint(this->positions[i])));
	}
	return worldPositions;
}

std::vector<Normal3f> TriangleMesh::WorldSpaceNormals() const {
	std::vector<Normal3f> worldNormals;
	for(int i = 0; i < this->nVertices; ++i) {
		worldNormals.push_back(this->ObjectToWorld.TransformNormal(this->normals[i]));
	}
	return worldNormals;
}

TriangleMesh& TriangleMesh::ChangeHandedness(int handedness) {
	if (this->handedness != handedness) {
		FlipZ();
		FlipWindingOrder();
		auto swapHandMatrix = Transform::Scale(1.f, 1.f, -1.f).matrix;
		auto newObjToWorldMat = Matrix4f::MatMul(swapHandMatrix, Matrix4f::MatMul(this->ObjectToWorld.matrix, swapHandMatrix));
		auto newInvObjToWorldMat = Matrix4f::MatMul(swapHandMatrix, Matrix4f::MatMul(this->ObjectToWorld.invMatrix, swapHandMatrix));
		this->ObjectToWorld = Transform(newObjToWorldMat, newInvObjToWorldMat);
		CalculateTriangleBounds();
		this->handedness = handedness;
	}
	return *this;
}

void TriangleMesh::CalculateTriangleBounds() {
	triangleBounds.clear();
	triangleBounds.reserve(nTriangles);
	for (int i = 0; i < nTriangles; ++i) {
		const Point3f& p0 = Point3f(ObjectToWorld.TransformPoint(positions[indices[3 * i]]));
		const Point3f& p1 = Point3f(ObjectToWorld.TransformPoint(positions[indices[3 * i + 1]]));
		const Point3f& p2 = Point3f(ObjectToWorld.TransformPoint(positions[indices[3 * i + 2]]));
		Bounds3f b = Bounds3f::Union(Bounds3f::Union(Bounds3f(p0), p1), p2);
		this->triangleBounds.push_back(b);
	}
}

void TriangleMesh::FlipWindingOrder() {
	for (int i = 0; i < nTriangles; ++i) {
		std::swap(this->indices[3 * i + 1], this->indices[3 * i + 2]);
	}
}

void TriangleMesh::FlipZ() {
	for (int i = 0; i < nVertices; ++i) {
		positions[i].z = -positions[i].z;
		normals[i].z = -normals[i].z;
	}
}