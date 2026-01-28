#ifndef INTERACTION_H
#define INTERACTION_H

#include "cudadefines.h"
#include "vector.h"
#include "point.h"

struct CUDA_ALIGN(16) SurfaceInteraction
{
	Vector3f p;
	Vector3f n;
	Vector2f uv;
	Vector3f dpdu, dpdv;
	int pad[2];
	// 64 bytes total
};

#endif