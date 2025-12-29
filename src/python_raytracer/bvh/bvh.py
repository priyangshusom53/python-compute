import numpy as np

from python_raytracer.c_functions import _bvh


def calculate_bvh(world_aabbs, max_tris_in_node=1):
   bvh_c_instance = _bvh.BVH()
   nodes_raw, ordered_tris = bvh_c_instance.build(world_aabbs, max_tris_in_node)

   assert nodes_raw.base is None or nodes_raw.base is bvh_c_instance
   assert nodes.dtype.itemsize == 56


   LinearBVHNode_dtype = np.dtype([
    ("bounds_min", np.float64, (3,)),
    ("bounds_max", np.float64, (3,)),
    ("offset",     np.int32),
    ("nTris",      np.uint16),
    ("axis",       np.uint8),
    ("pad",        np.uint8),
   ], align=True)

   nodes = nodes_raw.view(LinearBVHNode_dtype)

   return(nodes, ordered_tris)


