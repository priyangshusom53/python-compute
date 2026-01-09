#ifndef RAY_CUH
#define RAY_CUH

struct Ray
{
   float4 o;
   float4 d; // fourth component in d is tMax
};

#endif
