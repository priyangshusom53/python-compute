#include "trace.h"

extern "C" __global__ void trace_scene(
    StructuredBufferView<Ray>& rays,
    const int W,
    const int H,
    const StructuredBufferView<Vector3i>& indexBuffer,
    const StructuredBufferView<Point3f>& positionBuffer,
    const StructuredBufferView<Normal3f>& normalBuffer,
    const StructuredBufferView<Vector2f>& uvBuffer,
    const StructuredBufferView<GPUTriangleMesh>& meshes,  // GPUTriangleMesh type
    const StructuredBufferView<Triangle>& triangles,      // triangles are ordered to math bvh layout
    const StructuredBufferView<LinearBVHNode>& bvhNodes,  // LinearBVHNode type
    const int numMaterials,
    StructuredBufferView<Vector4f>& output) {

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
) {

    StructuredBufferView<Ray> _rays = rays.view();
    const StructuredBufferView<Vector3i> _indexBuffer = indexBuffer.view();
    const StructuredBufferView<Point3f> _positionBuffer = positionBuffer.view();
    const StructuredBufferView<Normal3f> _normalBuffer = normalBuffer.view();
    const StructuredBufferView<Vector2f> _uvBuffer = uvBuffer.view();
    const StructuredBufferView<GPUTriangleMesh> _meshes = meshes.view();
    const StructuredBufferView<Triangle> _triangles = triangles.view();
    const StructuredBufferView<LinearBVHNode> _bvhNodes = bvhNodes.view();

    StructuredBufferView<Vector4f> _output = output.view();

    dim3 blockDim = dim3(16, 16, 1);
    uint32_t gridX = std::ceil(W / blockDim.x);
    uint32_t gridY = std::ceil(H / blockDim.y);
    dim3 gridDim = dim3(gridX, gridY, 1);
    trace_scene<<<gridDim, blockDim>>>(
        _rays,
        W,
        H,
        _indexBuffer,
        _positionBuffer,
        _normalBuffer,
        _uvBuffer,
        _meshes,
        _triangles,
        _bvhNodes,
        1,
        _output
        );
}