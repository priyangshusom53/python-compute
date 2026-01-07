
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <numpy_c.h>

#include <cassert>
#include <type_traits>
#include <algorithm>
#include <array>
#include <limits>


namespace py = pybind11;


inline std::array<double, 3> vecmin(const double* vec1, const double* vec2) {
    
    std::array<double, 3> result;
    result[0] = std::min(vec1[0], vec2[0]);
    result[1] = std::min(vec1[1], vec2[1]);
    result[2] = std::min(vec1[2], vec2[2]);

    return result;
}

inline std::array<double, 3> vecmax(const double* vec1, const double* vec2) {

    std::array<double, 3> result;
    result[0] = std::max(vec1[0], vec2[0]);
    result[1] = std::max(vec1[1], vec2[1]);
    result[2] = std::max(vec1[2], vec2[2]);

    return result;
}

// AABB and it's methods
struct AABB {
    double min[3];
    double max[3];
};

AABB Empty() {

    AABB b;
    b.min[0] = b.min[1] = b.min[2] = 
        std::numeric_limits<double>::infinity();
    b.max[0] = b.max[1] = b.max[2] = 
        -std::numeric_limits<double>::infinity();

    return b;
}

std::array<double, 3>Diagonal(const AABB& bounds) {
    std::array<double, 3> result;
    result[0] = bounds.max[0] - bounds.min[0];
    result[1] = bounds.max[1] - bounds.min[1];
    result[2] = bounds.max[2] - bounds.min[2];

    return result;
}

int MaxExtent(const AABB& bounds) {

    double x = bounds.max[0] - bounds.min[0];
    double y = bounds.max[1] - bounds.min[1];
    double z = bounds.max[2] - bounds.min[2];

    if (x > y && x > z)
        return 0;
    else if (y > z)
        return 1;
    else
        return 2;
}

std::array<double, 3> Offset(const AABB& bounds, 
    const double* point) {

    const double *pMin = bounds.min;
    const double *pMax = bounds.max;

    std::array<double, 3> o;
    o[0] = point[0] - pMin[0];
    o[1] = point[1] - pMin[1];
    o[2] = point[2] - pMin[2];

    if (pMax[0] > pMin[0]) o[0] /= pMax[0] - pMin[0];
    if (pMax[1] > pMin[1]) o[1] /= pMax[1] - pMin[1];
    if (pMax[2] > pMin[2]) o[2] /= pMax[2] - pMin[2];

    return o;
}

double SurfaceArea(const AABB& bounds) {

    auto diag = Diagonal(bounds);
    return 2 * 
        (diag[0] * diag[1] + 
            diag[0] * diag[2] + diag[1] * diag[2]);
}

AABB Union(const AABB& bounds, const double* point) {
    
    AABB result;
    auto _min = vecmin(bounds.min, point);
    auto _max = vecmax(bounds.max, point);

    result.min[0] = _min[0];
    result.min[1] = _min[1];
    result.min[2] = _min[2];

    result.max[0] = _max[0];
    result.max[1] = _max[1];
    result.max[2] = _max[2];

    return result;
}
AABB Union(const AABB& bounds1, const AABB& bounds2) {


    AABB result;
    auto _min = vecmin(bounds1.min, bounds2.min);
    auto _max = vecmax(bounds1.max, bounds2.max);

    result.min[0] = _min[0];
    result.min[1] = _min[1];
    result.min[2] = _min[2];

    result.max[0] = _max[0];
    result.max[1] = _max[1];
    result.max[2] = _max[2];

    return result;
}
// --------

struct BVHTriangleInfo {
    size_t triNumber; // triangle number is index to indices array
    AABB bounds;
    double centroid[3];
};

struct BVHBuildNode {
    AABB bounds;
    BVHBuildNode* children[2];
    int splitAxis, firstTriOffset, nTris;

    BVHBuildNode(): bounds(Empty()), splitAxis(-1), 
        firstTriOffset(-1), nTris(-1){
        children[0] = children[1] = nullptr;
    }

    void InitLeaf(int first, int n, AABB& bounds) {
        firstTriOffset = first;
        nTris = n;
        this->bounds = bounds;
        children[0] = children[1] = nullptr;
    }

    void InitInterior(int axis, BVHBuildNode* c0, BVHBuildNode* c1) {
        children[0] = c0;
        children[1] = c1;
        bounds = Union(c0->bounds, c1->bounds);
        splitAxis = axis;
        nTris = 0;
    }
};

struct LinearBVHNode {
    AABB bounds;    // 48 bytes
    int32_t offset; // triangleOffset or secondChildOffset
    uint16_t nTris; // 0 for interior nodes 
    uint8_t axis;
    uint8_t pad;        
};

static_assert(sizeof(LinearBVHNode) == 56, "Must be 56 bytes");
static_assert(std::is_standard_layout_v<LinearBVHNode>);
static_assert(std::is_trivially_copyable_v<LinearBVHNode>);

/* Functions for building BVH */

int flatten_bvh_tree(LinearBVHNode* linearNodes, BVHBuildNode* node, int* offset) {
    
    LinearBVHNode* linearNode = &linearNodes[*offset];
    linearNode->bounds = node->bounds;
    int myoffset = (*offset)++;
    if (node->nTris > 0) { // leaf node
        linearNode->offset = node->firstTriOffset;
        linearNode->nTris = node->nTris;
        linearNode->axis = 0;
        linearNode->pad = 0;
    }
    else { // interior node
        linearNode->axis = node->splitAxis;
        linearNode->nTris = 0;
        flatten_bvh_tree(linearNodes, node->children[0], offset);
        linearNode->offset =
            flatten_bvh_tree(linearNodes, node->children[1], offset);
    }
    return myoffset;
}

BVHBuildNode* recursive_build(
    std::vector<BVHBuildNode*>& nodePtrs,
    std::vector<BVHTriangleInfo>& bvhTriangleInfos,
    int start, int end, const int maxTrisInNode, int* totalNodes, 
    std::vector<int32_t>& orderedTris) {

    // create node
    BVHBuildNode* node = new BVHBuildNode();
    nodePtrs.push_back(node);
    (*totalNodes)++;

    // compute total bounds of all triangles
    AABB nodeBounds = Empty();
    for (int i = start; i < end; ++i) {
        nodeBounds = Union(nodeBounds, bvhTriangleInfos[i].bounds);
    }


    int nTriangles = end - start;
    if (nTriangles == 1) // create leaf node 
    {
        int firstTriOffset = orderedTris.size();
        for (int i = start; i < end; ++i) {
            int triNumber = bvhTriangleInfos[i].triNumber;
            orderedTris.push_back(triNumber);
        }
        node->InitLeaf(firstTriOffset, nTriangles, nodeBounds);
        return node;
    }

    // compute bounds of triangle bounds centroids and 
    // choose max dim as split axis
    AABB centroidBounds = Empty();
    for (int i = start; i < end; ++i) {
        centroidBounds = Union(centroidBounds, bvhTriangleInfos[i].centroid);
    }
    int dim = MaxExtent(centroidBounds);
    int mid = (start + end) / 2;
    /*
    * If all of the centroid points are at the same position 
    * (i.e., the centroid bounds have zero volume), then 
    * recursion stops and a leaf node is created with the primitives; 
    * none of the splitting methods here is effective in that (unusual) case.
    */
    if (centroidBounds.max[dim] == centroidBounds.min[dim]) { // create leaf
        int firstTriOffset = orderedTris.size();
        for (int i = start; i < end; ++i) {
            int triNumber = bvhTriangleInfos[i].triNumber;
            orderedTris.push_back(triNumber);
        }
        node->InitLeaf(firstTriOffset, nTriangles, nodeBounds);
        return node;
    }
    else {
        // Partition primitives using approximate SAH

        // Allocate BucketInfo for SAH partition buckets
        constexpr int nBuckets = 12;
        struct BucketInfo {
            int count = 0;
            AABB bounds = Empty();
        };
        BucketInfo buckets[nBuckets];

        // Initialize BucketInfo for SAH partition buckets
        for (int i = start; i < end; ++i) {
            int b = nBuckets * 
                Offset(centroidBounds, bvhTriangleInfos[i].centroid)[dim];
            if (b == nBuckets) b = nBuckets - 1;
            buckets[b].count++;
            buckets[b].bounds =
                Union(buckets[b].bounds, bvhTriangleInfos[i].bounds);
        }

        // Compute costs for splitting after each bucket
        double cost[nBuckets - 1] = { 0 };
        for (int i = 0; i < nBuckets - 1; ++i) {
            AABB b0 = Empty(), b1 = Empty();
            int count0 = 0, count1 = 0;
            for (int j = 0; j <= i; ++j) {
                b0 = Union(b0, buckets[j].bounds);
                count0 += buckets[j].count;
            }
            for (int j = i + 1; j < nBuckets; ++j) {
                b1 = Union(b1, buckets[j].bounds);
                count1 += buckets[j].count;
            }

            double parentArea = SurfaceArea(nodeBounds);
            if (parentArea <= 0) parentArea = 1;
            // calculate SAH
            cost[i] = .125 + (count0 * SurfaceArea(b0) +
                count1 * SurfaceArea(b1)) / parentArea;
        }

        // Find bucket to split at that minimizes SAH metric
        double minCost = cost[0];
        int minCostSplitBucket = 0;
        for (int i = 1; i < nBuckets - 1; ++i) {
            if (cost[i] < minCost) {
                minCost = cost[i];
                minCostSplitBucket = i;
            }
        }

        // Either create leaf or split primitives at selected SAH bucket
        double leafCost = nTriangles;
        if (nTriangles > maxTrisInNode || minCost < leafCost) {
            BVHTriangleInfo* pmid = std::partition(&bvhTriangleInfos[start],
                &bvhTriangleInfos[end - 1] + 1,
                [=](const BVHTriangleInfo& pi) {
                    int b = nBuckets * Offset(centroidBounds, pi.centroid)[dim];
                    if (b == nBuckets) b = nBuckets - 1;
                    return b <= minCostSplitBucket;
                });
            mid = pmid - &bvhTriangleInfos[0];
        }
        else {
            // Create leaf node
            int firstTriOffset = orderedTris.size();
            for (int i = start; i < end; ++i) {
                int triNumber = bvhTriangleInfos[i].triNumber;
                orderedTris.push_back(triNumber);
            }
            node->InitLeaf(firstTriOffset, nTriangles, nodeBounds);
            return node;
        }

        node->InitInterior(dim,
            recursive_build(nodePtrs, bvhTriangleInfos, 
                start, mid, maxTrisInNode, totalNodes, orderedTris),
            recursive_build(nodePtrs, bvhTriangleInfos, 
                mid, end, maxTrisInNode, totalNodes, orderedTris));
    }
    return node;
}

std::vector<BVHTriangleInfo> make_bvhTriangleInfos_from_aabbs(const AABB* aabbs, ssize_t triCount) {

    std::vector<BVHTriangleInfo> infos;
    infos.reserve(triCount);

    for (size_t i = 0; i < triCount; ++i) {
        BVHTriangleInfo info;
        info.triNumber = i;
        info.bounds = aabbs[i];

        for (int k = 0; k < 3; ++k)
            info.centroid[k] = 0.5 * (aabbs[i].min[k] + aabbs[i].max[k]);

        infos.push_back(info);
    }
    return infos;
}

struct BVH {
    std::vector<LinearBVHNode> linearNodes;
    std::vector<int32_t> orderedTriangles;


    auto build(c_numpy_arr<double> world_aabbs, int32_t maxTrisinNode) {

        auto buf = world_aabbs.request();

        if (buf.ndim != 3)
            throw std::runtime_error("Expected 3D array");

        if (buf.ndim != 3 || buf.shape[1] != 2 || buf.shape[2] != 3)
            throw std::runtime_error("Expected shape (N, 2, 3)");

        if (buf.itemsize != sizeof(double))
            throw std::runtime_error("Expected float64");

        if (!(buf.strides[2] == sizeof(double) &&
            buf.strides[1] == 3 * sizeof(double)))
            throw std::runtime_error("Array must be C-contiguous");

        const py::ssize_t triCount = buf.shape[0];

        static_assert(sizeof(AABB) == 6 * sizeof(double),
            "AABB layout mismatch");

        static_assert(alignof(AABB) == alignof(double),
            "Unexpected AABB alignment");

        const AABB* aabbs = reinterpret_cast<const AABB*>(buf.ptr);
        auto infos = make_bvhTriangleInfos_from_aabbs(aabbs, triCount);

        std::vector<BVHBuildNode*> nodePtrs;
        int totalNodes = 0;

        auto bvhRoot = recursive_build(nodePtrs, infos,
            0, triCount, maxTrisinNode, &totalNodes, orderedTriangles);

        
        linearNodes.resize(totalNodes);
        int offset = 0;
        flatten_bvh_tree(&linearNodes[0], nodePtrs[0], &offset);

        auto nodes_buf = as_numpy_byte_buffer<LinearBVHNode>
            (&linearNodes[0], totalNodes, py::cast(this));

        py::ssize_t nTriangles = static_cast<py::ssize_t>(orderedTriangles.size())/3;
        auto tris_buf = as_numpy_buffer<int32_t, int32_t>(
            &orderedTriangles[0],{ nTriangles, 3},py::cast(this));


        for (auto* n : nodePtrs) delete n;
        return py::make_tuple(nodes_buf, tris_buf);
    }
};




PYBIND11_MODULE(_bvh, mod)
{
    py::class_<BVH>(mod, "BVH")
        .def(py::init<>())
        .def("build", &BVH::build);
}