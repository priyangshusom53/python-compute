#ifndef BVH_CUH
#define BVH_CUH

#include "bounds.cuh"

struct __align__(16) LinearBVHNode{
   Bounds b;
   int offset; // firstPrimitiveOffset for leaf nodes & second child offset for interior nodes
   unsigned short nTris;
   unsigned char axis;
   unsigned char pad[9];
};

static_assert(sizeof(LinearBVHNode) == 48, "BVH ABI mismatch");

#endif