#ifndef INTERACTION_H
#define INTERACTION_H

#include "cudadefines.h"
#include "vector.h"
#include "point.h"
#include "normal.h"

struct CUDA_ALIGN(16) SurfaceInteraction
{
	Point3f p;
	Normal3f n;
	Vector2f uv;
	Vector3f dpdu, dpdv;
	int pad[2];
	// 64 bytes total
};

#endif