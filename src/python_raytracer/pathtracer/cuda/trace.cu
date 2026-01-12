
#include "math/vector.cuh"
#include "debug.cuh"

#include "trianglemesh.cuh"
#include "ray.cuh"
#include "bounds.cuh"
#include "intersection.cuh"

extern "C" __global__ void trace_scene(const float4 *vertexBuffer, const float4 *normalBuffer, const float2 *uvBuffer, const int3 *indexBuffer,const int numVertices, const TriangleMesh *meshes, const Triangle *triangles){

}