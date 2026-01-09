#ifndef INTERSECTION_CUH
#define INTERSECTION_CUH

#include "bounds.cuh"
#include "ray.cuh"

#include "math/vector.cuh"
#include "algo.cuh"

struct SurfaceInteraction
{
   float4 p;
   float4 n;
};

__device__ __inline__ bool intersect_bound(const Bounds &b, const Ray &ray, float hitt0, float hitt1)
{
   float t0 = 0, t1 = ray.d.w;
   for (unsigned int i = 0; i < 3; ++i)
   {
      float invRayDir = 1 / value_at(ray.d, i);
      float tNear = (value_at(b.pMin, i) - value_at(ray.o, i)) * invRayDir;
      float tFar = (value_at(b.pMax, i) - value_at(ray.o, i)) * invRayDir;

      if (tNear > tFar)
         swap(tNear, tFar);
      t0 = tNear > t0 ? tNear : t0;
      t1 = tFar < t1 ? tFar : t1;
      if (t0 > t1)
         return false;
   }
   hitt0 = t0;
   hitt1 = t1;
   return true;
}

#endif
