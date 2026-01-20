
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
   const unsigned char *meshes,      // TriangleMesh type
   const Triangle *triangles,        // triangles are ordered to math bvh layout
   const int numTriangles,
   const unsigned char *bvhNodes,    // LinearBVHNode type
   const unsigned char *materials,   // PBRMaterial type
   const int numMaterials,
   float4 *output){
      static_assert(sizeof(TriangleMesh) == 160, "TriangleMesh ABI mismatch");
      static_assert(alignof(TriangleMesh) == 16,  "TriangleMesh alignment mismatch");

      const TriangleMesh *_meshes = reinterpret_cast<const TriangleMesh*>(meshes);
      const LinearBVHNode *_bvhNodes = reinterpret_cast<const LinearBVHNode*>(bvhNodes);
      const PBRMaterial * _materials = reinterpret_cast<const PBRMaterial*>(materials);

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
      bool hit = intersect_bvh(ray,_bvhNodes,_meshes,buffers,triangles,isect);
      // placeholder materials
      PBRMaterial material;
      static_assert(sizeof(PBRMaterial) == 32, "layout mismatch");
      if(hit){
         CUDA_ASSERT(numMaterials > 0, "Should have atleast one material");
         // material = _materials[(x+y) % numMaterials];
         output[y*w+x] = make_float4(0,1,0,1);
      }else
         output[y*w+x] = make_float4(1,0,0,1);
      return;
      // material.baseColorFactor = make_float4(1,0,0,1);
      // output[y*w+x] =  material.baseColorFactor; 
}