#include "mesh.h"
#include <stdexcept>

TriangleMesh::TriangleMesh(int nTriangles, const int indices[], int nVertices,
	const float positions[], const float normals[] = nullptr,
	const float uvs[] = nullptr, const float triangleBounds[] = nullptr,
	const float ObjectToWorldMatrix[] = nullptr, int materialIndex = 0,
	int handedness = LEFT_HANDED, bool generateNormals = false) {

	if (handedness == LEFT_HANDED) {
		this->handedness = LEFT_HANDED;
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
		if (normals) {
			this->normals.reserve(nVertices);
			for (int i = 0; i < 3 * nVertices; i += 3) {
				this->normals.push_back(Normal3f(normals[i], normals[i + 1], normals[i + 2]));
			}
			this->hasNormals = true;
		}
		else if (generateNormals) {
			CalculateNormals();
			this->hasNormals = true;
		}
		else
			this->hasNormals = false;
		
		if(ObjectToWorldMatrix) {
			this->ObjectToWorld = Transform(Matrix4f(ObjectToWorldMatrix));
		}
		else {
			this->ObjectToWorld = Transform::Identity();
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
	}
	else {
		this->handedness = RIGHT_HANDED;
		this->nTriangles = nTriangles;
		this->indices.reserve((3 * nTriangles));
		for (int i = 0; i < 3 * nTriangles; i+=3) {
			this->indices.push_back(indices[i]);
			this->indices.push_back(indices[i + 2]);
			this->indices.push_back(indices[i + 1]);
		}
		this->nVertices = nVertices;
		this->positions.reserve(nVertices);
		for (int i = 0; i < 3 * nVertices; i += 3) {
			this->positions.push_back(Point3f(positions[i], positions[i + 1], -positions[i + 2]));
		}
		if (normals) {
			this->normals.reserve(nVertices);
			for (int i = 0; i < 3 * nVertices; i += 3) {
				this->normals.push_back(Normal3f(normals[i], normals[i + 1], -normals[i + 2]));
			}
			this->hasNormals = true;
		}
		else if (generateNormals) {
			CalculateNormals();
			this->hasNormals = true;
		}
		else
			this->hasNormals = false;

		if (ObjectToWorldMatrix) {
			auto objToWorldRH = Matrix4f(ObjectToWorldMatrix);
			auto RH_to_LH = Transform::Scale(1.f, 1.f, -1.f).matrix;
			this->ObjectToWorld = Transform(Matrix4f::MatMul(RH_to_LH, Matrix4f::MatMul(objToWorldRH, RH_to_LH)));
		}
		else {
			this->ObjectToWorld = Transform::Identity();
		}

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
	
	this->materialIndex = materialIndex;
}

TriangleMesh::TriangleMesh(int nTriangles,
	const int indices[], int nVertices, const Point3f positions[],
	const Normal3f normals[] = nullptr, const Vector2f uvs[] = nullptr,
	const Bounds3f triangleBounds[] = nullptr,
	const Transform ObjectToWorld = Transform::Identity(),
	int materialIndex = 0, int handedness = LEFT_HANDED,
	bool generateNormals = false) {

	if (handedness == LEFT_HANDED) {
		this->handedness = LEFT_HANDED;
		this->nTriangles = nTriangles;
		this->indices.reserve((3 * nTriangles));
		for (int i = 0; i < 3 * nTriangles; ++i) {
			this->indices.push_back(indices[i]);
		}
		this->nVertices = nVertices;
		this->positions.reserve(nVertices);
		for (int i = 0; i < nVertices; i += 1) {
			this->positions.push_back(positions[i]);
		}
		if (normals) {
			this->normals.reserve(nVertices);
			for (int i = 0; i < nVertices; i += 1) {
				this->normals.push_back(normals[i]);
			}
			this->hasNormals = true;
		}
		else if (generateNormals) {
			CalculateNormals();
			this->hasNormals = true;
		}
		else
			this->hasNormals = false;

		this->ObjectToWorld = Transform(ObjectToWorld.matrix, ObjectToWorld.invMatrix);
		if (triangleBounds) {
			this->triangleBounds.reserve(nTriangles);
			for (int i = 0; i < nTriangles; i += 1) {
				this->triangleBounds.push_back(Bounds3f(triangleBounds[i].pMin, triangleBounds[i].pMax));
			}
		}
		else {
			CalculateTriangleBounds();
		}
	}
	else {
		this->handedness = RIGHT_HANDED;
		this->nTriangles = nTriangles;
		this->indices.reserve((3 * nTriangles));
		for (int i = 0; i < 3 * nTriangles; i += 3) {
			this->indices.push_back(indices[i]);
			this->indices.push_back(indices[i + 2]);
			this->indices.push_back(indices[i + 1]);
		}
		this->nVertices = nVertices;
		this->positions.reserve(nVertices);
		for (int i = 0; i < 3 * nVertices; i += 3) {
			this->positions.push_back(Point3f(positions[i].x, positions[i].y, -positions[i].z));
		}
		if (normals) {
			this->normals.reserve(nVertices);
			for (int i = 0; i < 3 * nVertices; i += 3) {
				this->normals.push_back(Normal3f(normals[i].x, normals[i].y, -normals[i].z));
			}
			this->hasNormals = true;
		}
		else if (generateNormals) {
			CalculateNormals();
			this->hasNormals = true;
		}
		else
			this->hasNormals = false;

		auto objToWorldRH = ObjectToWorld.matrix;
		auto RH_to_LH = Transform::Scale(1.f, 1.f, -1.f).matrix;
		this->ObjectToWorld = Transform(Matrix4f::MatMul(RH_to_LH, Matrix4f::MatMul(objToWorldRH, RH_to_LH)));

		CalculateTriangleBounds();
	}
	if (uvs) {
		this->uvs.reserve(nVertices);
		for (int i = 0; i < nVertices; i += 1) {
			this->uvs.push_back(uvs[i]);
		}
		this->hasTexCoords = true;
	}
	else {
		this->hasTexCoords = false;
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

bool TriangleMesh::HasNormals() const {
	return this->hasNormals;
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
	this->hasNormals = true;
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
		for (int i = 0; i < this->nVertices; ++i) {
			positions[i].z = -positions[i].z;
			if (this->hasNormals) {
				normals[i].z = -normals[i].z;
			}
		}
		for (int i = 0; i < this->nTriangles; ++i) {
			std::swap(this->indices[3 * i + 1], this->indices[3 * i + 2]);
		}
		auto swapHandMatrix = Transform::Scale(1.f, 1.f, -1.f).matrix;
		auto newObjToWorldMat = Matrix4f::MatMul(swapHandMatrix, Matrix4f::MatMul(this->ObjectToWorld.matrix, swapHandMatrix));
		auto newInvObjToWorldMat = Matrix4f::MatMul(swapHandMatrix, Matrix4f::MatMul(this->ObjectToWorld.invMatrix, swapHandMatrix));
		this->ObjectToWorld = Transform(newObjToWorldMat, newInvObjToWorldMat);
		CalculateTriangleBounds();
		this->handedness = handedness;
	}
}

TriangleMesh TriangleMesh::ChangeHandedness(const TriangleMesh& mesh, int handedness) {

}

void TriangleMesh::CalculateTriangleBounds() {
	for (int i = 0; i < this->nTriangles; ++i) {
		const Point3f& p0 = this->positions[this->indices[3 * i]];
		const Point3f& p1 = this->positions[this->indices[3 * i + 1]];
		const Point3f& p2 = this->positions[this->indices[3 * i + 2]];
		Bounds3f b = Bounds3f::Union(Bounds3f::Union(Bounds3f(p0), p1), p2);
		this->triangleBounds.push_back(b);
	}
}