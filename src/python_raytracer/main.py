
import numpy as np

from python_raytracer.loader.gltf_loader import GLTFLoader

from python_raytracer.core.geometry.triangle_mesh import TriangleMesh
from python_raytracer.plots.o3dplots import plot_mesh_data
from python_raytracer.bvh import bvh

def main():

   # load the scene
   loader = GLTFLoader()
   path = r"D:\3D Models\sponza_gltf\scene.gltf"
   meshes, materials = loader.load(path)

   # prepare arrays for collecting vertex attribute data 
   # into single arrays
   all_vertices:np.ndarray = None
   all_normals:np.ndarray = None
   all_uvs:np.ndarray = None
   all_indices:np.ndarray = None
   all_world_bounds:np.ndarray = None
   transforms:list[np.ndarray] = None
   vertex_offset = 0

   all_vertices = meshes[0].positions.array
   all_normals = meshes[0].normals.array if meshes[0].normals is not None else None
   all_uvs = meshes[0].uv.array if meshes[0].uv is not None else None
   all_indices = meshes[0].vertex_indices
   vertex_offset += meshes[0].n_vertices
   all_world_bounds = meshes[0].world_bounds if meshes[0].world_bounds is not None else None
   transforms = [meshes[0].transform.matrix]

   # collect all mesh data into single arrays, 
   # adjust indices array by indices += mesh vertex count
   mesh_count = len(meshes)
   tri_count = sum(mesh.n_triangles for mesh in meshes)
   for mesh in meshes[1:]:

      all_vertices = np.concatenate([all_vertices, mesh.positions.array], axis=0)

      all_normals = np.concatenate([all_normals, mesh.normals.array], axis=0) if all_normals is not None and mesh.normals is not None else all_normals

      all_uvs = np.concatenate([all_uvs, mesh.uv.array], axis=0) if all_uvs is not None and mesh.uv is not None else all_uvs

      # adjust indices offset
      adjusted_indices = mesh.vertex_indices + vertex_offset
      all_indices = np.concatenate([all_indices, adjusted_indices], axis=0)

      all_world_bounds = np.concatenate([all_world_bounds, mesh.world_bounds], axis=0) if all_world_bounds is not None and mesh.world_bounds is not None else all_world_bounds

      # store local to world transforms
      transforms.append(mesh.transform.matrix)

      vertex_offset += mesh.n_vertices



   # store contagious arrays
   all_obj_vertices_cont = np.ascontiguousarray(all_vertices,dtype=np.float64)
   all_normals_cont:np.ndarray = np.ascontiguousarray(all_normals, dtype=np.float64)
   all_uvs_cont:np.ndarray = np.ascontiguousarray(all_uvs, dtype=np.float64)
   all_indices_cont:np.ndarray = np.ascontiguousarray(all_indices, dtype=np.int32)
   all_world_bounds_cont:np.ndarray = np.ascontiguousarray(all_world_bounds, dtype=np.float64)
   transforms:list[np.ndarray] = transforms

   world_vertices_list = [mesh.positions.array @ mesh.transform.matrix.T for mesh in meshes]
   world_vertices = np.concatenate(world_vertices_list, axis=0)

   # calculate bvh
   bvh_nodes_arr, ordered_triangles = bvh.calculate_bvh(all_world_bounds_cont,4)


   # visualize only interior nodes
   leaf_filter = bvh_nodes_arr["nTris"] == 0
   bounds_min = bvh_nodes_arr["bounds_min"][leaf_filter]
   bounds_max = bvh_nodes_arr["bounds_max"][leaf_filter]
   
   bvh_bounds = np.ascontiguousarray(
    np.stack([bounds_min, bounds_max], axis=1)
   )



   plot_mesh_data(world_vertices,
                  all_indices,
                  all_normals,
                  all_uvs, 
                  bvh_bounds)

   


if __name__ == "__main__":
   main()


   
