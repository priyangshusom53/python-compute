#include "mesh.h"
#include <stdexcept>

//TriangleMesh::TriangleMesh(
//	int nTriangles,
//	const int indices[],
//	int nVertices,
//	const float positions[],
//	const float ObjectToWorldMatrix[],
//	const float normals[],
//	const float uvs[],
//	const float triangleBounds[],
//	int materialIndex,
//	int handedness
//) {
//
//	
//	this->handedness = handedness;
//	this->hasTexCoords = false;
//	this->nTriangles = nTriangles;
//	this->indices.reserve((3 * nTriangles));
//	for (int i = 0; i < 3 * nTriangles; ++i) {
//		this->indices.push_back(indices[i]);
//	}
//	this->nVertices = nVertices;
//	this->positions.reserve(nVertices);
//	for (int i = 0; i < 3 * nVertices; i += 3) {
//		this->positions.push_back(Point3f(positions[i], positions[i + 1], positions[i + 2]));
//	}
//	if (ObjectToWorldMatrix) {
//		this->ObjectToWorld = Transform(Matrix4f(ObjectToWorldMatrix));
//	}
//	else {
//		this->ObjectToWorld = Transform::Identity();
//	}
//
//	if (normals) {
//		this->normals.reserve(nVertices);
//		for (int i = 0; i < 3 * nVertices; i += 3) {
//			this->normals.push_back(Normal3f(normals[i], normals[i + 1], normals[i + 2]));
//		}
//	}
//	else {
//		CalculateNormals();
//	}
//		
//	if(triangleBounds) {
//		this->triangleBounds.reserve(nTriangles);
//		for(int i = 0; i < 6 * nTriangles; i += 6) {
//			Point3f pMin(triangleBounds[i], triangleBounds[i + 1], triangleBounds[i + 2]);
//			Point3f pMax(triangleBounds[i + 3], triangleBounds[i + 4], triangleBounds[i + 5]);
//			this->triangleBounds.push_back(Bounds3f(pMin, pMax));
//		}
//	}
//	else {
//		CalculateTriangleBounds();
//	}
//	
//	if (uvs) {
//		this->uvs.reserve(nVertices);
//		for (int i = 0; i < 2 * nVertices; i += 2) {
//			this->uvs.push_back(Vector2f(uvs[i], uvs[i + 1]));
//		}
//		this->hasTexCoords = true;
//	}
//	else {
//		this->hasTexCoords = false;
//	}
//	if (handedness == RIGHT_HANDED) {
//		ChangeHandedness(LEFT_HANDED);
//	}
//	this->materialIndex = materialIndex;
//}
//
//TriangleMesh::TriangleMesh(
//	int nTriangles,
//	const std::vector<int>& indices,
//	int nVertices,
//	const std::vector<Point3f>& positions,
//	const std::vector<Normal3f>& normals,
//	const std::vector<Vector2f>& uvs,
//	const std::vector<Bounds3f>& triangleBounds,
//	const Transform ObjectToWorld,
//	int materialIndex,
//	int handedness) {
//
//	this->handedness = handedness;
//	this->hasTexCoords = false;
//	this->nTriangles = nTriangles;
//	this->indices.reserve((3 * nTriangles));
//	for (int i = 0; i < 3 * nTriangles; ++i) {
//		this->indices.push_back(indices[i]);
//	}
//	this->nVertices = nVertices;
//	this->positions.reserve(nVertices);
//	for (int i = 0; i < nVertices; ++i) {
//		this->positions.push_back(positions[i]);
//	}
//
//	this->ObjectToWorld = ObjectToWorld;
//	if (!normals.empty()) {
//		for(int i = 0; i < nVertices; ++i) {
//			this->normals.push_back(normals[i]);
//		}
//	}
//	else {
//		CalculateNormals();
//	}
//
//	if (!triangleBounds.empty()) {
//		for(int i = 0; i < nTriangles; ++i) {
//			this->triangleBounds.push_back(triangleBounds[i]);
//		}
//	}
//	else {
//		CalculateTriangleBounds();
//	}
//	if (!uvs.empty()) {
//		for(int i = 0; i < nVertices; ++i) {
//			this->uvs.push_back(uvs[i]);
//		}
//		this->hasTexCoords = true;
//	}
//	else {
//		this->hasTexCoords = false;
//	}
//	if (handedness == RIGHT_HANDED) {
//		ChangeHandedness(LEFT_HANDED);
//	}
//	this->materialIndex = materialIndex;
//
//}

TriangleMesh::TriangleMesh(
	const std::vector<Point3f>& positions,
	const std::vector<Vector3i>& indices,
	int handedness,
	const std::vector<Normal3f>& normals,
	const std::vector<Vector2f>& uvs,
	int materialIndex
) : nTriangles(indices.size()), nVertices(positions.size()), handedness(LEFT_HANDED), 
materialIndex(materialIndex)
{
	TriangleMesh& _this = *this;
	_this.positions = std::vector<Point3f>(nVertices);
	for (int i = 0; i < _this.nVertices; ++i)
		_this.positions[i] = positions[i];

	if (indices.empty()) {
		nTriangles = nVertices / 3;
		_this.indices = std::vector<Vector3i>(nTriangles);
		for (int i = 0; i < nTriangles; ++i) {
			_this.indices[i] = Vector3i(3 * i, 3 * i + 1, 3 * i + 2);
		}
	}
	else {
		_this.indices = std::vector<Vector3i>(indices.size());
		for (int i = 0; i < _this.nTriangles; ++i)
			_this.indices[i] = indices[i];
	}
		
	if (!normals.empty()) {
		_this.normals = std::vector<Normal3f>(nVertices);
		for (int i = 0; i < _this.nVertices; ++i)
			_this.normals[i] = normals[i];
	}else {
		CalculateNormals();
	}

	if (!uvs.empty()) {
		_this.uvs = std::vector<Vector2f>(nVertices);
		for (int i = 0; i < _this.nVertices; ++i)
			_this.uvs[i] = uvs[i];
	}

	if (handedness == RIGHT_HANDED)
		ChangeHandedness(handedness);

}

//void TriangleMesh::SetIndices(int nTriangles, const int indices[]) {
//	if (this->nTriangles != nTriangles) {
//		throw std::invalid_argument("Number of triangles does not match existing mesh.");
//	}
//	for (int i = 0; i < 3 * nTriangles; ++i) {
//		this->indices[i] = indices[i];
//	}
//}
//
//void TriangleMesh::SetPositions(int nVertices, const float positions[]) {
//	if(this->nVertices != nVertices) {
//		throw std::invalid_argument("Number of vertices does not match existing mesh.");
//	}
//	for(int i = 0; i < nVertices; ++i) {
//		this->positions[i] = Point3f(positions[3*i], positions[3*i + 1], positions[3*i + 2]);
//	}
//}

void TriangleMesh::CalculateNormals() {
	TriangleMesh& _this = *this;
	_this.normals.clear();
	_this.normals.reserve(this->nVertices);
	_this.normals.resize(this->nVertices, Normal3f(0.f, 0.f, 0.f));
	const std::vector<Vector3i>& indices = _this.indices;
	for (int i = 0; i < this->nTriangles; ++i) {
		int i0 = indices[i].x;
		int i1 = indices[i].y;
		int i2 = indices[i].z;
		const Point3f& p0 = _this.positions[i0];
		const Point3f& p1 = _this.positions[i1];
		const Point3f& p2 = _this.positions[i2];
		const Vector3f e01 = p1 - p0;
		const Vector3f e02 = p2 - p0;
		Normal3f n = Normal3f(Vector3f::Normalize(Vector3f::Cross(e01, e02)));
		_this.normals[i0] = n;
		_this.normals[i1] = n;
		_this.normals[i2] = n;
	}
}

bool TriangleMesh::HasNormals() const {
	return (!normals.empty());
}
void TriangleMesh::SetNormals(const std::vector<Normal3f>& normals) {
	if (normals.size() != nVertices)
		throw std::runtime_error("Normals array should have same size as nVertices");
	TriangleMesh& _this = *this;
	_this.normals.clear();
	std::vector<Normal3f> _normals(nVertices);
	for (int i = 0; i < nVertices; ++i) {
		_normals[i] = normals[i];
	}
	_this.normals = std::move(_normals);
}
bool TriangleMesh::HasTextureCoords() const {
	return (!uvs.empty());
}
void TriangleMesh::SetTextureCoords(const std::vector<Vector2f>& uvs) {
	if (uvs.size() != nVertices)
		throw std::runtime_error("UV array should have same size as nVertices");
	TriangleMesh& _this = *this;
	_this.uvs.clear();
	std::vector<Vector2f> _uvs(nVertices);
	for (int i = 0; i < nVertices; ++i) {
		_uvs[i] = uvs[i];
	}
	_this.uvs = std::move(_uvs);
}

//void TriangleMesh::SetTransform(const Transform& ObjectToWorld) {
//	this->ObjectToWorld = ObjectToWorld;
//}

void TriangleMesh::SetMaterialIdx(int materialIndex) {
	this->materialIndex = materialIndex;
}

// Bakes transform into vertex positions and updates triangle bounds
void TriangleMesh::TransformMeshObjectSpace(const Transform& t) {
	for(int i = 0; i < nVertices; ++i) {
		positions[i] = Point3f(t.TransformPoint(positions[i]));
	}
	if (HasNormals()) {
		for (int i = 0; i < nVertices; ++i) {
			normals[i] = Normal3f(t.TransformNormal(normals[i]));
		}
	}
}

// Modifies ObjectToWorld transform
//void TriangleMesh::TransformMeshWorldSpace(const Transform& t) {
//
//	const Matrix4f newObjToWorldMat = Matrix4f::MatMul(t.matrix, ObjectToWorld.matrix);
//	const Matrix4f newInvObjToWorldMat = Matrix4f::MatMul(ObjectToWorld.invMatrix,t.matrix);
//	ObjectToWorld = Transform(newObjToWorldMat, newInvObjToWorldMat);
//}

std::vector<Point3f> TriangleMesh::WorldSpacePositions(const Transform& ObjectToWorld) const {
	std::vector<Point3f> worldPositions(nVertices);
	for(int i = 0; i < nVertices; ++i) {
		worldPositions.push_back(Point3f(ObjectToWorld.TransformPoint(positions[i])));
	}
	return worldPositions;
}

std::vector<Normal3f> TriangleMesh::WorldSpaceNormals(const Transform& ObjectToWorld) const {
	if (!HasNormals())
		return std::vector<Normal3f>();
	std::vector<Normal3f> worldNormals;
	for(int i = 0; i < nVertices; ++i) {
		worldNormals.push_back(ObjectToWorld.TransformNormal(normals[i]));
	}
	return worldNormals;
}

TriangleMesh& TriangleMesh::ChangeHandedness(int handedness) {
	TriangleMesh& _this = *this;
	if (_this.handedness != handedness) {
		FlipZ();
		FlipWindingOrder();
		_this.handedness = handedness;
	}
	return *this;
}
//
//void TriangleMesh::CalculateTriangleBounds() {
//	triangleBounds.clear();
//	triangleBounds.reserve(nTriangles);
//	for (int i = 0; i < nTriangles; ++i) {
//		const Point3f& p0 = Point3f(ObjectToWorld.TransformPoint(positions[indices[3 * i]]));
//		const Point3f& p1 = Point3f(ObjectToWorld.TransformPoint(positions[indices[3 * i + 1]]));
//		const Point3f& p2 = Point3f(ObjectToWorld.TransformPoint(positions[indices[3 * i + 2]]));
//		Bounds3f b = Bounds3f::Union(Bounds3f::Union(Bounds3f(p0), p1), p2);
//		this->triangleBounds.push_back(b);
//	}
//}


std::vector<Bounds3f> TriangleMesh::GetTriangleObjectBounds() const {
	std::vector<Bounds3f> triangleObjectBounds(nTriangles);
	for (int i = 0; i < nTriangles; ++i) {
		Vector3i idx = indices[i];
		Bounds3f bounds =
			Bounds3f::Union(Bounds3f(positions[idx[0]], positions[idx[1]]), positions[idx[2]]);
		triangleObjectBounds[i] = bounds;
	}
	return triangleObjectBounds;
}
std::vector<Bounds3f> TriangleMesh::GetTriangleWorldBounds(const Transform& ObjectToWorld) const {
	std::vector<Point3f> worldPositions = WorldSpacePositions(ObjectToWorld);
	std::vector<Bounds3f> triangleWorldBounds(nTriangles);
	for (int i = 0; i < nTriangles; ++i) {
		Vector3i idx = indices[i];
		Bounds3f bounds =
			Bounds3f::Union(Bounds3f(worldPositions[idx[0]], worldPositions[idx[1]]), worldPositions[idx[2]]);
		triangleWorldBounds[i] = bounds;
	}
	return triangleWorldBounds;
}

void TriangleMesh::FlipWindingOrder() {
	for (int i = 0; i < nTriangles; ++i) {
		std::swap(this->indices[i].y, this->indices[i].z);
	}
}

void TriangleMesh::FlipZ() {
	for (int i = 0; i < nVertices; ++i) {
		positions[i].z = -positions[i].z;
	}
	if (HasNormals()) {
		for (int i = 0; i < nVertices; ++i) {
			normals[i].z = -normals[i].z;
		}
	}
}