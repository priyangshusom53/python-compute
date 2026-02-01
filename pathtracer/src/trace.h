#ifndef TRACE_H
#define TRACE_H

#include "cudadefines.h"
#include "mathconstants.h"
#include "buffer.h"
#include "vector.h"
#include "point.h"
#include "normal.h"
#include "bounds.h"
#include "transformation.h"
#include "ray.h"
#include "mesh.h"
#include "triangle.h"
#include "bvh.h"
#include "utils.h"

#include <cmath>

#include <device_launch_parameters.h>

void Render(
    StructuredBuffer<Ray, GPU_BUFFER>& rays,
    const int W,
    const int H,
    const StructuredBuffer<Vector3i, GPU_BUFFER>& indexBuffer,
    const StructuredBuffer<Point3f, GPU_BUFFER>& positionBuffer,
    const StructuredBuffer<Normal3f, GPU_BUFFER>& normalBuffer,
    const StructuredBuffer<Vector2f, GPU_BUFFER>& uvBuffer,
    const StructuredBuffer<GPUTriangleMesh, GPU_BUFFER>& meshes,  // GPUTriangleMesh type
    const StructuredBuffer<Triangle, GPU_BUFFER>& triangles,      // triangles are ordered to math bvh layout
    const StructuredBuffer<LinearBVHNode, GPU_BUFFER>& bvhNodes,  // LinearBVHNode type
    const int numMaterials,
    StructuredBuffer<Vector4f, GPU_BUFFER>& output
);

#endif