#ifndef SOA_H
#define SOA_H

#include "cudadefines.h"
#include "buffer.h"
#include "vector.h"
#include "point.h"
#include "normal.h"
#include "triangle.h"
#include "mesh.h"

struct SOA {
	StructuredBuffer<Vector3i,BufferType::CPU_BUFFER> indices;
	StructuredBuffer<Point3f, BufferType::CPU_BUFFER> positions;
	StructuredBuffer<Normal3f, BufferType::CPU_BUFFER> normals;
	StructuredBuffer<Vector2f, BufferType::CPU_BUFFER> uvs;
};

#endif