#ifndef ALGO_CUH
#define ALGO_CUH

__device__ __forceinline__ void swap(float &a, float &b)
{
   float tmp = a;
   a = b;
   b = tmp;
}

#endif