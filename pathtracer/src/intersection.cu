
#include "cudadefines.h"
#include "buffer.h"
#include "ray.h"
#include "vector.h"
#include "point.h"
#include "bounds.h"
#include "transformation.h"
#include "mesh.h"
#include "utils.h"

struct CUDA_ALIGN(16) SurfaceInteraction
{
	Vector3f p;
	Vector3f n;
	Vector2f uv;
	Vector3f dpdu, dpdv;
	int pad[2];
	// 64 bytes total
};



