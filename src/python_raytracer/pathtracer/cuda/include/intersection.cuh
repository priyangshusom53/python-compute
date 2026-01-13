#ifndef INTERSECTION_CUH
#define INTERSECTION_CUH

#include "bounds.cuh"
#include "ray.cuh"
#include "trianglemesh.cuh"
#include "bvh.cuh"

#include "math/vector.cuh"
#include "math/transform.cuh"
#include "algo.cuh"
#include "debug.cuh"

struct __align__(16) SurfaceInteraction
{
   float4 p;			 // float4 is 16 byte aligned
   float4 n;
	float2 uv;
	float3 dpdu, dpdv; // float3 is 16 byte aligned
	// 80 bytes total
};

__device__ __inline__ bool intersect_bound(const Ray &ray, const Bounds &b, float &hitt0, float &hitt1)
{
   float t0 = 0, t1 = ray.d.w; // tmax of ray
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

__device__ __inline__ bool intersect_bound(const Ray &ray, const Bounds &b, const float3 &invDir, const int dirIsNeg[3]){
	float tMin =  (b[  dirIsNeg[0]].x - ray.o.x) * invDir.x;
	float tMax =  (b[1-dirIsNeg[0]].x - ray.o.x) * invDir.x;
	float tyMin = (b[  dirIsNeg[1]].y - ray.o.y) * invDir.y;
	float tyMax = (b[1-dirIsNeg[1]].y - ray.o.y) * invDir.y;
	if (tMin > tyMax || tyMin > tMax) 
      return false;
	if (tyMin > tMin) tMin = tyMin; 
	if (tyMax < tMax) tMax = tyMax;
	float tzMin = (b[  dirIsNeg[2]].z - ray.o.z) * invDir.z; 
	float tzMax = (b[1-dirIsNeg[2]].z - ray.o.z) * invDir.z;
	if (tMin > tzMax || tzMin > tMax) 
		return false; 
	if (tzMin > tMin) tMin = tzMin; 
	if (tzMax < tMax) tMax = tzMax;
	// ray.d.w is tmax of ray
	return (tMin < ray.d.w) && (tMax > 0); 
}

// triangle comes under geometric primitive that updates Ray's tMax to intersection t
__device__ bool intersect_triangle(
	const Ray &ray, 
	const TriangleMesh *meshes, 
	const AttributeBuffers &buffers, 
	const Triangle &triangle, 
	float &tHit, 
	SurfaceInteraction &isect)
{
   // transform triangle vertices to ray triangle intersection space

   // get vertex data from attribute buffers
   TriangleMesh mesh = meshes[triangle.meshIdx];
   int globalTriIdx = (mesh.firstTriangleIdx+triangle.triangleIdx);
	CUDA_ASSERT((buffers.indexBuffer[globalTriIdx].x < mesh.firstVertexIdx+mesh.numVertices),"indices should be less than numVertices, buffer overflow");
	CUDA_ASSERT((buffers.indexBuffer[globalTriIdx].y < mesh.firstVertexIdx+mesh.numVertices),"indices should be less than numVertices, buffer overflow");
	CUDA_ASSERT((buffers.indexBuffer[globalTriIdx].z < mesh.firstVertexIdx+mesh.numVertices),"indices should be less than numVertices, buffer overflow");

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
	if(det< 0 && (tScaled >= 0 || tScaled < ray.d.w * det))
		return false;
	else if (det > 0 && (tScaled <= 0 || tScaled > ray.d.w * det))
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
	isect.n = normalize(isect.n);
	tHit = t;
	return true;
}

__device__ bool intersect_triangle(
	Ray &ray, 
	const TriangleMesh *meshes, 
	const AttributeBuffers &buffers, 
	const Triangle &triangle,  
	SurfaceInteraction &isect){
		float tHit;
		if(!(intersect_triangle(ray,meshes,buffers,triangle,tHit,isect)))
			return false;
		ray.d.w = tHit;
		return true;
}

__device__ bool intersect_bvh(
	Ray& ray,
	const LinearBVHNode *bvhNodes, 
	const TriangleMesh *meshes, 
	const AttributeBuffers &buffers, 
	const Triangle *orderedTriangles, 
	SurfaceInteraction &isect){
		bool hit = false;
		float3 invDir = make_float3(1/ray.d.x,1/ray.d.y,1/ray.d.z);
		int dirIsNeg[3] = { invDir.x < 0, invDir.y < 0, invDir.z<0};
		int toVisitOffset = 0, currentNodeIndex = 0;
		unsigned short nodesToVisit[64];
		while (true) {
			const LinearBVHNode node = bvhNodes[currentNodeIndex];
			if(intersect_bound(ray,node.b,invDir,dirIsNeg)){
				if(node.nTris > 0){
					CUDA_ASSERT(node.nTris < 255, "number of nodes exceed unsigned char limit");
					for(unsigned char i = 0; i < node.nTris; ++i){
						if(intersect_triangle(ray, meshes,buffers, orderedTriangles[node.offset + i],isect))
								hit = true;
					}
					if (toVisitOffset == 0) break;
					currentNodeIndex = nodesToVisit[--toVisitOffset];
				}else{
					if (dirIsNeg[node.axis]) {
						nodesToVisit[toVisitOffset++] = currentNodeIndex + 1;
						CUDA_ASSERT(toVisitOffset < 64, "BVH stack overflow");
						currentNodeIndex = node.offset;
					} else {
						nodesToVisit[toVisitOffset++] = node.offset;
						CUDA_ASSERT(toVisitOffset < 64, "BVH stack overflow");
						currentNodeIndex = currentNodeIndex + 1;
					}
				}
			}else{
				if(toVisitOffset == 0) break;
				currentNodeIndex = nodesToVisit[--toVisitOffset];
			}
		}	
		return hit;
}

#endif
