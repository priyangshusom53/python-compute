#ifndef MATERIAL_CUH
#define MATERIAL_CUH

struct __align__(16) PBRMaterial{
   float4 baseColorFactor;
   float metallicFactor;
   float roughnessFactor;
   float pad[2];
};

#endif