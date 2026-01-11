#ifndef INTERSECTION_CUH
#define INTERSECTION_CUH

#include "bounds.cuh"
#include "ray.cuh"
#include "trianglemesh.cuh"

#include "math/vector.cuh"
#include "math/transform.cuh"
#include "algo.cuh"
#include "debug.cuh"

struct __align__(16) SurfaceInteraction
{
   float4 p;
   float4 n;
	float2 uv;
	float3 dpdu, dpdv;
	// 64 bytes total
};

__device__ __inline__ bool intersect_bound(const Bounds &b, const Ray &ray, float &hitt0, float &hitt1)
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

__device__ bool intersect_triangle(const Ray &ray, const TriangleMesh *meshes, const AttributeBuffers &buffers, const Triangle &triangle, float &tHit, SurfaceInteraction &isect)
{
   // transform triangle vertices to ray triangle intersection space

   // get vertex data from attribute buffers
   TriangleMesh mesh = meshes[triangle.meshIdx];
   int globalTriIdx = (mesh.firstTriangleIdx+triangle.triangleIdx);
	CUDA_ASSERT((buffers.indexBuffer[globalTriIdx].x < mesh.firstVertexIdx+mesh.numVertices),"indices should be less than numVertices, buffer overflow")
	CUDA_ASSERT((buffers.indexBuffer[globalTriIdx].y < mesh.firstVertexIdx+mesh.numVertices),"indices should be less than numVertices, buffer overflow")
	CUDA_ASSERT((buffers.indexBuffer[globalTriIdx].z < mesh.firstVertexIdx+mesh.numVertices),"indices should be less than numVertices, buffer overflow")

   float4 p0 = buffers.vertexBuffer[buffers.indexBuffer[globalTriIdx].x];
   float4 p1 = buffers.vertexBuffer[buffers.indexBuffer[globalTriIdx].y];
   float4 p2 = buffers.vertexBuffer[buffers.indexBuffer[globalTriIdx].z];

   // translate vertices with ray origin
   float4 p0t = p0 - ray.o;
   float4 p1t = p1 - ray.o;
   float4 p2t = p2 - ray.o;

   // make z axis the maxDim of ray.d
   int kz=maxDim(vecAbs(ray.d));
   int kx=kz +1;if(kx==3)kx=0;
   int ky=kx +1;if(ky==3)ky=0;
	
	float4 d = permute3(ray.d, kx, ky, kz);
	p0t=permute3(p0t,kx,ky,kz);
	p1t=permute3(p1t,kx,ky,kz);
	p2t=permute3(p2t,kx,ky,kz);

	// apply shear transformation to translated vertex positions
	float Sx=-d.x/d.z;
	float Sy=-d.y/d.z;
	float Sz=1.f/d.z;
	// coordinate permutation and sheer coefficients is only dependent on ray and independent of triangle
	// this info can be precomputed for each ray before testing for intersection
	p0t.x+=Sx *p0t.z;
	p0t.y+=Sy *p0t.z;
	p1t.x+=Sx *p1t.z;
	p1t.y+=Sy *p1t.z;
	p2t.x+=Sx *p2t.z;
	p2t.y+=Sy *p2t.z;
	// compute edge function for each edge and ray origin. if all 3 edge function values
	// have same sign(+ or -) ray intersects the triangle
	// edge functions are indexed p0->p1, p1->p2, p2->p0
	float e0 = p1t.x * p2t.y- p1t.y * p2t.x;
	float e1 = p2t.x * p0t.y- p2t.y * p0t.x;
	float e2 = p0t.x * p1t.y- p0t.y * p1t.x;  
	if((e0 < 0 || e1 < 0 || e2 < 0) && (e0 > 0 || e1 > 0 || e2 > 0))
		return false;
	float det = e0 + e1 + e2;
	if (det == 0)
		return false;
	// compute scaled hit distance to triangle and test against ray t range
	p0t.z *= Sz;
	p1t.z *= Sz;
	p2t.z *= Sz;
	float tScaled = e0 * p0t.z + e1 * p1t.z + e2 * p2t.z;
	if(det< 0 && tScaled >= 0)
		return false;
	else if (det > 0 && tScaled <= 0)
		return false;
	float invDet=1/det;
	float b0 = e0 * invDet;
	float b1 = e1 * invDet;
	float b2 = e2 * invDet;
	float t = tScaled * invDet;
	// compute partial derivatives dp/du, dp/dv 
	float3 dpdu,dpdv;
	float2 uv0,uv1,uv2;
	// get uvs of the vertices
	if(buffers.uvBuffer){
		uv0 = buffers.uvBuffer[buffers.indexBuffer[globalTriIdx].x];
		uv1 = buffers.uvBuffer[buffers.indexBuffer[globalTriIdx].y];
		uv2 = buffers.uvBuffer[buffers.indexBuffer[globalTriIdx].z];
	}else{
		uv0 = make_float2(0,0);
		uv1 = make_float2(0,1);
		uv2 = make_float2(1,1);
	}
	
	float2 duv02 = uv0 - uv2, duv12 = uv1 - uv2;
	// dp02 is vector p2->p0, dp12 is vector p2->p1
	float4 dp02 = p0 - p2, dp12 = p1 - p2;
	float determinant = duv02.x * duv12.y - duv02.y * duv12.x;

	dpdu = to_float3(( duv12.y * dp02- duv02.y * dp12) / determinant);
	dpdv = to_float3((-duv12.x * dp02 + duv02.x * dp12) / determinant);

	// Interpolate (u, v) parametric coordinates and hit point
	isect.p = b0 * p0 + b1 * p1 + b2 * p2;
	isect.uv =  b0 * uv0 + b1 * uv1 + b2 * uv2;
	isect.dpdu = dpdu;
	isect.dpdv = dpdv;
	isect.n = b0 * buffers.normalBuffer[buffers.indexBuffer[globalTriIdx].x] +
				 b1 * buffers.normalBuffer[buffers.indexBuffer[globalTriIdx].y] +
				 b2 * buffers.normalBuffer[buffers.indexBuffer[globalTriIdx].z];
	tHit = t;
	return true;
}

#endif
