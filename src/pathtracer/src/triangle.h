#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "buffer.h"
#include "cudadefines.h"
#include "vector.h"
#include "point.h"
#include "normal.h"
#include "bounds.h"
#include "mesh.h"
#include "interaction.h"

struct CUDA_ALIGN(4) Triangle{
	int meshIdx;
	int triangleIdx;
	Bounds3f worldBounds;
	CPU_GPU Triangle(int _meshIdx, int _triangleIdx, Bounds3f& _worldBounds);
	CPU_GPU bool Intersect(
		Ray& ray,
		const StructuredBufferView<GPUTriangleMesh>&meshes,
		const StructuredBufferView<Vector3i>&indexBuffer,
		const StructuredBufferView<Point3f>&positions,
		const StructuredBufferView<Normal3f>&normals,
		const StructuredBufferView<Vector2f>&uv,
		SurfaceInteraction * isect
	) const ;
	CPU_GPU bool Intersect(
		const Ray & ray,
		float* tHit,
		const StructuredBufferView<GPUTriangleMesh>&meshes,
		const StructuredBufferView<Vector3i>&indexBuffer,
		const StructuredBufferView<Point3f>&positions,
		const StructuredBufferView<Normal3f>&normals,
		const StructuredBufferView<Vector2f>&uv,
		SurfaceInteraction * isect
	) const;
};

CPU_GPU INLINE Triangle::Triangle(int _meshIdx, int _triangleIdx, 
	Bounds3f& _worldBounds) :
	meshIdx(_meshIdx), triangleIdx(_triangleIdx), 
	worldBounds(_worldBounds.pMin,_worldBounds.pMax)  {}

CPU_GPU INLINE bool Triangle::Intersect(
	Ray& ray,
	const StructuredBufferView<GPUTriangleMesh>& meshes,
	const StructuredBufferView<Vector3i>& indexBuffer,
	const StructuredBufferView<Point3f>& positions,
	const StructuredBufferView<Normal3f>& normals,
	const StructuredBufferView<Vector2f>& uv,
	SurfaceInteraction* isect
) const {

	float tHit = 0;
	if (Intersect(ray,&tHit,meshes,indexBuffer,positions,normals,uv,isect)) {
		ray.tMax = tHit;
		return true;
	}
	return false;
}

CPU_GPU INLINE bool Triangle::Intersect(
	const Ray& ray,
	float* tHit,
	const StructuredBufferView<GPUTriangleMesh>& meshes,
	const StructuredBufferView<Vector3i>& indexBuffer,
	const StructuredBufferView<Point3f>& positions,
	const StructuredBufferView<Normal3f>& normals,
	const StructuredBufferView<Vector2f>& uv,
	SurfaceInteraction* isect
) const {

	const Vector3i& indices = 
		indexBuffer[meshes[meshIdx].firstTriangleIdx+triangleIdx];
	const Point3f& p0 = positions[indices.x];
	const Point3f& p1 = positions[indices.y];
	const Point3f& p2 = positions[indices.z];
	Point3f p0t = p0 - Vector3f(ray.o);
	Point3f p1t = p1 - Vector3f(ray.o);
	Point3f p2t = p2 - Vector3f(ray.o);

	int kz = Vector3f::MaxDimension(Vector3f::Abs(ray.d));
	int kx = kz + 1; if (kx == 3)kx = 0;
	int ky = kx + 1; if (ky == 3)ky = 0;
	Vector3f d = Vector3f::Permute(ray.d, kx, ky, kz);
	p0t = Point3f::Permute(p0t, kx, ky, kz);
	p1t = Point3f::Permute(p1t, kx, ky, kz);
	p2t = Point3f::Permute(p2t, kx, ky, kz);

	float Sx = -d.x / d.z;
	float Sy = -d.y / d.z;
	float Sz = 1.f / d.z;
	p0t.x += Sx * p0t.z;
	p0t.y += Sy * p0t.z;
	p1t.x += Sx * p1t.z;
	p1t.y += Sy * p1t.z;
	p2t.x += Sx * p2t.z;
	p2t.y += Sy * p2t.z;

	float e0 = p1t.x * p2t.y - p1t.y * p2t.x;
	float e1 = p2t.x * p0t.y - p2t.y * p0t.x;
	float e2 = p0t.x * p1t.y - p0t.y * p1t.x;

	if ((e0 < 0 || e1 < 0 || e2 < 0) && (e0 > 0 || e1 > 0 || e2 > 0))
		return false;
	float det = e0 + e1 + e2;
	if (det == 0)
		return false;

	p0t.z *= Sz;
	p1t.z *= Sz;
	p2t.z *= Sz;
	float tScaled = e0 * p0t.z + e1 * p1t.z + e2 * p2t.z;
	if (det < 0 && (tScaled >= 0 || tScaled < ray.tMax * det))
		return false;
	else if (det > 0 && (tScaled <= 0 || tScaled > ray.tMax * det))
		return false;

	float invDet = 1 / det;
	float b0 = e0 * invDet;
	float b1 = e1 * invDet;
	float b2 = e2 * invDet;
	float t = tScaled * invDet;
	Point3f pHit = b0 * p0 + b1 * p1 + b2 * p2;
	*tHit = t;
	*isect = { pHit,Normal3f(),Vector2f(),Vector3f(),Vector3f() };
	return true;
}

#endif