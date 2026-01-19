import numpy as np
import math
from PIL import Image
from python_raytracer.core.geometry.transformation import Transform
from python_raytracer.bvh.bvh import calculate_bvh
from python_raytracer.loader.gltf_loader import GLTFLoader
from python_raytracer.core.geometry.triangle_mesh import TriangleMesh

n = 1e-10
f = 1000
fov = 90
loader = GLTFLoader()
meshes, materials = loader.load(r"D:\3D Models\sponza_gltf\scene.gltf")
all_world_bounds = np.ascontiguousarray(np.concatenate([mesh.world_bounds for mesh in meshes],axis=0,dtype=np.float32))
bvh_nodes,tris = calculate_bvh(all_world_bounds,4)
leafs = bvh_nodes[bvh_nodes['nTris']>0]['nTris']
nleafs = len(leafs)
print(nleafs)
print(tris.shape)
print(np.max(tris))
total_tris = [mesh.n_triangles for mesh in meshes]
print(np.sum(total_tris))
