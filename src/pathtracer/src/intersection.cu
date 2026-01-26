
#include "cudadefines.h"
#include "ray.h"
#include "vector.h"
#include "point.h"
#include "bounds.h"
#include "transformation.h"
#include "utils.h"

struct CUDA_ALIGN(16) SurfaceInteraction
{
	Vector3f p;
	Vector3f n;
	Vector2f uv;
	Vector3f dpdu, dpdv;
	int pad[2];
	// 64 bytes total
};

GPU_ONLY CUDA_INLINE bool intersect_bound(const Ray& ray, const Bounds3f& b, float& hitt0, float& hitt1)
{
    float t0 = 0, t1 = ray.tMax;
    for (unsigned int i = 0; i < 3; ++i)
    {
        float invRayDir = 1 / ray.d[i];
        float tNear = (b.pMin[i] - ray.o[i]) * invRayDir;
        float tFar = (b.pMax[i] - ray.o[i]) * invRayDir;

        if (tNear > tFar)
            Swap<float>(tNear, tFar);
        t0 = tNear > t0 ? tNear : t0;
        t1 = tFar < t1 ? tFar : t1;
        if (t0 > t1)
            return false;
    }
    hitt0 = t0;
    hitt1 = t1;
    return true;
}

GPU_ONLY CUDA_INLINE bool intersect_bound(const Ray& ray, const Bounds3f& b, const Vector3f& invDir, const int dirIsNeg[3]) {
	float tMin = (b[dirIsNeg[0]].x - ray.o.x) * invDir.x;
	float tMax = (b[1 - dirIsNeg[0]].x - ray.o.x) * invDir.x;
	float tyMin = (b[dirIsNeg[1]].y - ray.o.y) * invDir.y;
	float tyMax = (b[1 - dirIsNeg[1]].y - ray.o.y) * invDir.y;
	if (tMin > tyMax || tyMin > tMax)
		return false;
	if (tyMin > tMin) tMin = tyMin;
	if (tyMax < tMax) tMax = tyMax;
	float tzMin = (b[dirIsNeg[2]].z - ray.o.z) * invDir.z;
	float tzMax = (b[1 - dirIsNeg[2]].z - ray.o.z) * invDir.z;
	if (tMin > tzMax || tzMin > tMax)
		return false;
	if (tzMin > tMin) tMin = tzMin;
	if (tzMax < tMax) tMax = tzMax;

	return (tMin < ray.tMax) && (tMax > 0);
}