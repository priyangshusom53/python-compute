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

#include <device_launch_parameters.h>


extern "C" __global__ void trace_scene(
    StructuredBufferView<Ray> rays,
    const int W,
    const int H,
    const StructuredBufferView<Vector3i> indexBuffer,
    const StructuredBufferView<Point3f> positionBuffer,
    const StructuredBufferView<Normal3f> normalBuffer,
    const StructuredBufferView<Vector2f> uvBuffer,
    const StructuredBufferView<GPUTriangleMesh> meshes,  // GPUTriangleMesh type
    const StructuredBufferView<Triangle> triangles,      // triangles are ordered to math bvh layout
    const StructuredBufferView<LinearBVHNode> bvhNodes,  // LinearBVHNode type
    const int numMaterials,
    StructuredBufferView<Vector4f> output) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    Ray ray = rays[y * W + x];
    
    SurfaceInteraction isect;
    bool hit = false;
    hit = IntersectBVH(
        ray,
        bvhNodes,
        triangles,
        meshes,
        indexBuffer,
        positionBuffer,
        normalBuffer,
        uvBuffer,
        &isect
    );
    //for (int i = 0; i < numTriangles; ++i) {
    //    hit = intersect_triangle(ray, _meshes, buffers, triangles[i], isect);
    //    if (hit) break;
    //}

    //// placeholder materials
    //PBRMaterial material;
    //static_assert(sizeof(PBRMaterial) == 32, "layout mismatch");
    if (hit) {
        /*CUDA_ASSERT(numMaterials > 0, "Should have atleast one material");*/
        // material = _materials[(x+y) % numMaterials];
        output[y * W + x] = Vector4f(0, 1, 0, 1);
    }
    else {
        output[y * W + x] = Vector4f(1, 0, 0, 1);
    }
    return;
    // material.baseColorFactor = make_float4(1,0,0,1);
    // output[y*w+x] =  material.baseColorFactor; 
}