#ifndef BVH_CUH
#define BVH_CUH

#include "bounds.cuh"

struct __align__(16) LinearBVHNode{
   Bounds b;
   int offset; // firstPrimitiveOffset for leaf nodes & second child offset for interior nodes
   unsigned short nTris;
   unsigned char axis;
};

#endif