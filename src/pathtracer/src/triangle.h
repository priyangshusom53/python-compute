#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "cudadefines.h"

struct CUDA_ALIGN(4) Triangle{
   int meshIdx;
   int triangleIdx;
   CPU_GPU Triangle(int _meshIdx, int _triangleIdx);
   CPU_GPU bool Intersect();
};

#endif