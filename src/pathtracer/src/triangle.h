#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "cudadefines.h"
#include "mesh.h"
#include "interaction.h"

struct CUDA_ALIGN(4) Triangle{
	int meshIdx;
	int triangleIdx;
	Bounds3f worldBounds;
	CPU_GPU Triangle(int _meshIdx, int _triangleIdx, Bounds3f& _worldBounds);
	CPU_GPU bool Intersect(
		const StructuredBufferView<GPUTriangleMesh>&meshes,
		const StructuredBufferView<Vector3i>&indexBuffer,
		const StructuredBufferView<Point3f>&positions,
		const StructuredBufferView<Normal3f>&normals,
		const StructuredBufferView<Vector2f>&uv,
		SurfaceInteraction * isect
	) const ;
};

CPU_GPU INLINE Triangle::Triangle(int _meshIdx, int _triangleIdx, 
	Bounds3f& _worldBounds) :
	meshIdx(_meshIdx), triangleIdx(_triangleIdx), 
	worldBounds(_worldBounds.pMin,_worldBounds.pMax)  {}

CPU_GPU INLINE bool Triangle::Intersect(
	const StructuredBufferView<GPUTriangleMesh>& meshes,
	const StructuredBufferView<Vector3i>& indexBuffer,
	const StructuredBufferView<Point3f>& positions,
	const StructuredBufferView<Normal3f>& normals,
	const StructuredBufferView<Vector2f>& uv,
	SurfaceInteraction* isect
) const {

}

#endif