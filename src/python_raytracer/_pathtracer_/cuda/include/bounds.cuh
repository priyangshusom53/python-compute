#ifndef BOUNDS_CUH
#define BOUNDS_CUH

struct __align__(16) Bounds
{
   float4 pMin;
   float4 pMax;

   __device__ __inline__ float4 operator[](const int idx) const {
      int _idx = idx % 2;
      if(_idx==0) return pMin;
      else return pMax;
   }
};

#endif