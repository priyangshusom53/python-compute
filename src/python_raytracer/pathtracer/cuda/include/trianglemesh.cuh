#ifndef TRIANGLE_MESH_CUH
#define TRIANGLE_MESH_CUH

#include "math/transform.cuh"

struct __align__(16) TriangleMesh
{
   int firstTriangleIdx;
   int numTriangles;
   int numVertices;
   int firstVertexIdx;
   int materialIdx;
   Transform transform;
};

struct __align__(16) Triangle
{
   int meshIdx;
   int triangleIdx;
};

struct __align__(16) AttributeBuffers
{
   int3 *indexBuffer;    // 4 bytes
   float4 *vertexBuffer; // 4 bytes
   float4 *normalBuffer; // 4 bytes
   float2 *uvBuffer;     // 4 bytes
};

#endif