#ifndef MATERIAL_H
#define MATERIAL_H

#include "cudadefines.h"
#include "vector.h"


/*
*	Used both in CPU GPU
*/
struct CUDA_ALIGN(4) PBRMaterial{
   Vector4f baseColorFactor;
   float metallicFactor;
   float roughnessFactor;
   float emissiveFactor;
   int pad[1];

   CPU_GPU PBRMaterial() :baseColorFactor(Vector4f(1, 1, 1, 1)),
	   metallicFactor(0), roughnessFactor(0), emissiveFactor(0)
   {}

   CPU_GPU PBRMaterial(
	   const Vector4f& _baseColorFactor,
	   float _metallicFactor, 
	   float _roughnessFactor, 
	   float _emissiveFactor)
	   : baseColorFactor(_baseColorFactor),
		metallicFactor(_metallicFactor),
	    roughnessFactor(_roughnessFactor),
	   emissiveFactor(_emissiveFactor)
   {}
};

#endif