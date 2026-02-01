#ifndef RAY_CUH
#define RAY_CUH

struct __align__(16) Ray
{
   float4 o;
   float4 d; // fourth component in d is tMax
};

#endif
