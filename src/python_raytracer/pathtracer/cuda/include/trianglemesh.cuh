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
   int pad[3];
   Transform transform;
};

// aligned to largest member in struct 4 bytes
struct __align__(4) Triangle
{
   int meshIdx;
   int triangleIdx;
};

struct AttributeBuffers
{
   const int3 *indexBuffer;    // 8 bytes on 64bit bin
   const float4 *vertexBuffer; // 8 bytes
   const float4 *normalBuffer; // 8 bytes
   const float2 *uvBuffer;     // 8 bytes
};

#endif