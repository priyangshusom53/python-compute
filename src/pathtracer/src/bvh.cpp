#include "bvh.h"

/*
*	Make BVHTriangleInfo from Triangle array and call RecursiveBuild
*/
BVHAccel::BVHAccel(
	const std::vector<std::shared_ptr<Triangle>>& triangles,
	int maxTrisInNode, SplitMethod splitMethod) 
	: maxTrisInNode(std::min(255, maxTrisInNode)), splitMethod(splitMethod), triangles(triangles) {

	std::vector<BVHTriangleInfo> triangleInfos(triangles.size());
	for (int i = 0; i < triangles.size(); ++i) {
		triangleInfos[i] = { i, triangles[i]->worldBounds };
	}
	int totalNodes = 0;
	std::vector<std::shared_ptr<Triangle>> orderedTriangles;
	std::vector<BVHBuildNode*> bvhNodes;
	BVHBuildNode* root = RecusiveBuild(bvhNodes, triangleInfos, 0,
		triangleInfos.size(), &totalNodes, orderedTriangles);
	this->triangles.swap(orderedTriangles);

	nodes = 
		StructuredBuffer<LinearBVHNode, BufferType::CPU_BUFFER>(totalNodes);
	nodes.Allocate();
	int offset = 0;
	FlattenBVHTree(root, &offset);
}

/*
*	Idea:
*		1. Loop through given array of BVHTriangleInfo,
*		2. Compute Union of triangle bounds from "start" to "end",
*		3. Select splitting criteria along which
		   total bound of all triangle is maximum and divide the
		   bound in two children bounds,
*		4. Perform step 1,2 and 3 recursively until 
*		   end-start <= maxTrisInNode
*		5. Make leaf node containing triangles end-start
*	Algorithms:
*		1. SAH:
*				Surface Area Heuristic is based on the cost function:
*				c(A, B) = t_trav + P_a(SUM(t_isect(a_i),1 < i < N_a) +
*								P_b(SUM(t_isect(b_i),1 < i < N_b)
*				t_trav = cost of traversing node containing A and B
*				P_a, P_b = probability of ray passing through node A and B
*				t_isect = cost of computing intersection with a triangle
*				N_a, N_b = number of triangles in A and B
*				
*				goal is to minimize this cost
*				only when ray has greater probability of passing through
*				node that contains less triangles
*				probability P(A|B) = S_A/S_B
*				i.e bigger surface area has greater probability
*		2. Middle:
*				1. Compute midpoint of the centroids of all the triangles
*				   along the partitioning axis,
*				2. Then seperate the triangles between two groups, 
*				   triangles that come before midpoint and after midpoint
*				   using std::partition(),
*				3. perform step 1 and 2 recursively
*		3. EqualCounts:
*				Splits total bound into two bounds with equal triangle
*				count and arranges them with std::nth_element()
*/
BVHBuildNode* BVHAccel::RecusiveBuild(
	std::vector<BVHBuildNode*>& bvhNodes,
	std::vector<BVHTriangleInfo>& triangleInfos,
	int start, int end, int* totalNodes,
	std::vector<std::shared_ptr<Triangle>>& orderedTriangles){
	
	BVHBuildNode* node = new BVHBuildNode();
	bvhNodes.push_back(node);
	(*totalNodes)++;
	Bounds3f bounds;
	for (int i = start; i < end; ++i) {
		bounds = Bounds3f::Union(bounds, triangleInfos[i].bounds);
	}
	int nTriangles = end - start;
	if (nTriangles == 1) {
		int firstTriangleOffset = orderedTriangles.size();
		for (int i = start; i < end; ++i) {
			int triangleNumber = triangleInfos[i].triangleNumber;
			orderedTriangles.push_back(triangles[triangleNumber]);
		}
		node->InitLeaf(firstTriangleOffset, nTriangles, bounds);
		return node;
	}
	else {
		Bounds3f centroidBounds;
		for (int i = start; i < end; ++i)
			centroidBounds = Bounds3f::Union(centroidBounds, triangleInfos[i].centroid);
		int dim = centroidBounds.MaximumExtent();
		
		int mid = (start + end) / 2;
		if (centroidBounds.pMin[dim] == centroidBounds.pMax[dim]) { // degenerate case
			int firstTriangleOffset = orderedTriangles.size();
			for (int i = start; i < end; ++i) {
				int triangleNumber = triangleInfos[i].triangleNumber;
				orderedTriangles.push_back(triangles[triangleNumber]);
			}
			node->InitLeaf(firstTriangleOffset, nTriangles, bounds);
			return node;
		}
		else {
			// Partition triangles based on SplitMethod
			switch (splitMethod) {
			case SplitMethod::Middle:{
				float midPoint = (centroidBounds.pMin[dim] + 
					centroidBounds.pMax[dim])*0.5f;
				BVHTriangleInfo* midPtr =
					std::partition(&triangleInfos[start], &triangleInfos[end - 1] + 1,
						[dim, midPoint](const BVHTriangleInfo& pi) {
							return pi.centroid[dim] < midPoint;
						});
				mid = midPtr - &triangleInfos[0];
				if (mid != start && mid != end)
					break;
			}
			case SplitMethod::EqualCounts: {
				mid = (start + end) / 2;
				std::nth_element(&triangleInfos[start], &triangleInfos[mid],
					&triangleInfos[end - 1] + 1,
					[dim](const BVHTriangleInfo& a, const BVHTriangleInfo& b) {
						return a.centroid[dim] < b.centroid[dim];
					});

				break;
			}
			case SplitMethod::SAH: {

				if (nTriangles <= 4) {
						mid = (start + end) / 2;
					std::nth_element(&triangleInfos[start], &triangleInfos[mid],
						&triangleInfos[end - 1] + 1,
						[dim](const BVHTriangleInfo& a, const BVHTriangleInfo& b) {
							return a.centroid[dim] < b.centroid[dim];
						});
				}
				else {
					constexpr int nBuckets = 12;
					struct BucketInfo {
						int count = 0;
						Bounds3f bounds;
					};

					BucketInfo buckets[nBuckets];
					for (int i = start; i < end; ++i) {
						int b = nBuckets *
							centroidBounds.Offset(triangleInfos[i].centroid)[dim];
						if (b == nBuckets) b = nBuckets - 1;
						buckets[b].count++;
						buckets[b].bounds = Bounds3f::Union(buckets[b].bounds, 
							triangleInfos[i].bounds);
					}

					float cost[nBuckets - 1];
					for (int i = 0; i < nBuckets - 1; ++i) {
						Bounds3f b0, b1;
						int count0 = 0, count1 = 0;
						for (int j = 0; j <= i; ++j) {
							b0 = Bounds3f::Union(b0, buckets[j].bounds);
							count0 += buckets[j].count;
						}
						for (int j = i + 1; j < nBuckets; ++j) {
							b1 = Bounds3f::Union(b1, buckets[j].bounds);
							count1 += buckets[j].count;
						}
						cost[i] = .125f + (count0 * b0.SurfaceArea() +
							count1 * b1.SurfaceArea()) / bounds.SurfaceArea();
					}

					float minCost = cost[0];
					int minCostSplitBucket = 0;
					for (int i = 1; i < nBuckets - 1; ++i) {
						if (cost[i] < minCost) {
							minCost = cost[i];
							minCostSplitBucket = i;
						}
					}

					float leafCost = nTriangles;
					if (nTriangles > maxTrisInNode || minCost < leafCost) {
						BVHTriangleInfo* pmid = std::partition(&triangleInfos[start],
							&triangleInfos[end - 1] + 1,
							[=](const BVHTriangleInfo& pi) {
								int b = nBuckets * centroidBounds.Offset(pi.centroid)[dim];
								if (b == nBuckets) b = nBuckets - 1;
								return b <= minCostSplitBucket;
							});
						mid = pmid - &triangleInfos[0];
					}
					else {
						int firstPrimOffset = orderedTriangles.size();
						for (int i = start; i < end; ++i) {
							int triangleNumber = triangleInfos[i].triangleNumber;
							orderedTriangles.push_back(triangles[triangleNumber]);
						}
						node->InitLeaf(firstPrimOffset, nTriangles, bounds);
						return node;
					}
				}

				break;
			}
			}

			node->InitInterior(dim,
				RecusiveBuild(bvhNodes, triangleInfos, start, mid,
					totalNodes, orderedTriangles),
				RecusiveBuild(bvhNodes, triangleInfos, mid, end,
					totalNodes, orderedTriangles));
		}

	}
	return node;
}

int BVHAccel::FlattenBVHTree(BVHBuildNode* node, int* offset) {
	LinearBVHNode* linearNode = &nodes[*offset];
	linearNode->bounds = node->bounds;
	int myOffset = (*offset)++;
	if (node->nTriangles > 0) {
		linearNode->offset = node->firstTriangleOffset;
		linearNode->nTriangles = node->nTriangles;
	}
	else {
		linearNode->axis = node->splitAxis;
		linearNode->nTriangles = 0;
		FlattenBVHTree(node->children[0], offset);
		linearNode->offset = FlattenBVHTree(node->children[1], offset);
	}
	return myOffset;
}