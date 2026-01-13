
#include "math/vector.cuh"
#include "debug.cuh"

#include "trianglemesh.cuh"
#include "ray.cuh"
#include "bounds.cuh"
#include "bvh.cuh"
#include "intersection.cuh"
#include "material.cuh"

/*    x axis
*     ------------->
*  y  |
*     |
*     |
*    \|/
*  block and thread axes
*/

extern "C" __global__ void trace_scene(
   const Ray *rays,
   const int w,
   const int h,
   const int3 *indexBuffer,
   const float4 *vertexBuffer, 
   const float4 *normalBuffer, 
   const float2 *uvBuffer, 
   const int numVertices, 
   const TriangleMesh *meshes, 
   const Triangle *triangles, // triangles are ordered to math bvh layout
   const int numTriangles,
   const LinearBVHNode *bvhNodes,
   const PBRMaterial *materials,
   const int numMaterials,
   float4 *output){

      int x = blockIdx.x * blockDim.x + threadIdx.x;
      int y = blockIdx.y * blockDim.y + threadIdx.y;
      if(x>=w || y>=h) return;
      Ray ray = rays[y*w+x]; 
      AttributeBuffers buffers{
         indexBuffer,
         vertexBuffer,
         normalBuffer,
         uvBuffer
      };
      SurfaceInteraction isect;
      bool hit = intersect_bvh(ray,bvhNodes,meshes,buffers,triangles,isect);
      // placeholder materials
      PBRMaterial material;
      static_assert(sizeof(PBRMaterial) == 32, "layout mismatch");
      if(hit){
         CUDA_ASSERT(numMaterials > 0, "Should have atleast one material");
         material = materials[(x+y) % numMaterials];
      }else
         material.baseColorFactor = make_float4(1,1,1,1);
      output[y*w+x] =  material.baseColorFactor; 
}