#ifndef TRIANGLE_MESH_CUH
#define TRIANGLE_MESH_CUH

#include "math/transform.cuh"

struct __align__(16) TriangleMesh
{
   unsigned int firstTriangleIdx;
   unsigned int numTriangles;
   unsigned int numVertices;
   unsigned int firstVertexIdx;
   unsigned int materialIdx;
   Transform transform;
};

struct __align__(16) Triangle
{
   unsigned int meshIdx;
   unsigned int triangleIdx;
};

struct __align__(16) AttributeBuffers
{
   int3 *indexBuffer;    // 4 bytes
   float4 *vertexBuffer; // 4 bytes
   float4 *normalBuffer; // 4 bytes
   float2 *uvBuffer;     // 4 bytes
};

#endif