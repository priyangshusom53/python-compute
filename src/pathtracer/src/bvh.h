#ifndef BVH_H
#define BVH_H

#include "cudadefines.h"
#include "vector.h"
#include "point.h"
#include "buffer.h"
#include "triangle.h"
#include "ray.h"
#include "interaction.h"

#include<vector>
#include<algorithm>
#include<memory>

enum SplitMethod { SAH, Middle, EqualCounts };

struct BVHTriangleInfo {
	int triangleNumber;
	Bounds3f bounds;
	Point3f centroid;
	BVHTriangleInfo(int triangleNumber, Bounds3f bounds)
		: triangleNumber(triangleNumber), bounds(bounds),
		centroid((bounds.pMin + bounds.pMax)*0.5f) {}
};

struct BVHBuildNode {
	Bounds3f bounds;
	BVHBuildNode* children[2];
	int splitAxis, firstTriangleOffset, nTriangles;
	void InitLeaf(int first, int n, const Bounds3f& b) {
		firstTriangleOffset = first;
		nTriangles = n;
		bounds = b;
		children[0] = children[1] = nullptr;
	}
	void InitInterior(int axis, BVHBuildNode* c0, BVHBuildNode* c1) {
		children[0] = c0;
		children[1] = c1;
		bounds = Bounds3f::Union(c0->bounds, c1->bounds);
		splitAxis = axis;
		nTriangles = 0;
	}
};

struct LinearBVHNode {
	Bounds3f bounds;
	int offset;		// firstTriangleOffset for leaf
					// secondChildOffset for interior
	unsigned short nTriangles;
	unsigned char axis;
	unsigned char pad[1];
};

class BVHAccel {
	const int maxTrisInNode;
	const SplitMethod splitMethod;
	std::vector<std::shared_ptr<Triangle>> triangles;
	StructuredBuffer<LinearBVHNode, BufferType::CPU_BUFFER> nodes;
public:
	BVHAccel(const std::vector<std::shared_ptr<Triangle>>& triangles, 
		int maxTrisInNode, SplitMethod splitMethod = SplitMethod::SAH);
	BVHBuildNode* RecusiveBuild(
		std::vector<BVHBuildNode*>& bvhNodes,
		std::vector<BVHTriangleInfo>& triangleInfos,
		int start, int end, int *totalNodes,
		std::vector<std::shared_ptr<Triangle>>& orderedTriangles);
	int FlattenBVHTree(BVHBuildNode* node, int *offset);
};

CPU_GPU INLINE bool IntersectBVH(const Ray& ray, 
	const StructuredBufferView<LinearBVHNode>& linearNodes,
	const StructuredBufferView<Triangle>& orderedTriangles,
	const StructuredBufferView<GPUTriangleMesh>& meshes,
	const StructuredBufferView<Vector3i>& indexBuffer,
	const StructuredBufferView<Point3f>& positions,
	const StructuredBufferView<Normal3f>& normals,
	const StructuredBufferView<Vector2f>& uv,
	SurfaceInteraction* isect) {

	bool hit = false;
	Vector3f invDir(1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z);
	int dirIsNeg[3] = { invDir.x < 0, invDir.y < 0, invDir.z < 0 };
	int toVisitOffset = 0, currentNodeIndex = 0;
	int nodesToVisitStack[64];
	while (true) {
		const LinearBVHNode* node = &linearNodes[currentNodeIndex];
		if (node->bounds.Intersect(ray, invDir, dirIsNeg)) {
			if (node->nTriangles > 0) {
				int offset = node->offset;
				int nTriangles = node->nTriangles;
				for (int i = offset; i < offset + nTriangles; ++i) {
					if (orderedTriangles[i].Intersect(
						meshes,
						indexBuffer,
						positions,
						normals,
						uv,
						isect
					)) {
						hit = true;
					}
				}
				if (toVisitOffset == 0) break;
				currentNodeIndex = nodesToVisitStack[--toVisitOffset];
			}
			else {
				if (dirIsNeg[node->axis]) {
					nodesToVisitStack[toVisitOffset++] = currentNodeIndex + 1;
					currentNodeIndex = node->offset;
				}
				else {
					nodesToVisitStack[toVisitOffset++] = node->offset;
					currentNodeIndex = currentNodeIndex + 1;
				}
			}
		}
		else {
			if (toVisitOffset == 0) break;
			currentNodeIndex = nodesToVisitStack[--toVisitOffset];
		}
	}
	return hit;
}

#endif