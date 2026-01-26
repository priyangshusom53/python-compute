import argparse
import logging

import numpy as np

from python_raytracer.core.log import logger
from python_raytracer.core import debugging
from python_raytracer.gltf_loader import GLTFLoader
from python_raytracer.transformation import Transform
from python_raytracer.triangle_mesh import TriangleMesh
from python_raytracer.core.camera.simplecamera import (PerspectiveCamera,Film)
from python_raytracer.pathtracer.pathtracer import PathTracer
from python_raytracer.plots.o3dplots import plot_mesh_data
from python_raytracer.bvh import bvh
from python_raytracer.plots.vtkvisualizer import Visualizer

# -debug or --debug flag to enable debugging
def main(debug:bool):

   # setup log
   logger.log_config()
   log = logging.getLogger(__name__)

   # load the scene
   loader = GLTFLoader()
   path = r"D:\3D Models\\gltf_sample_scenes\\glTF-Sample-Assets\\Models\ABeautifulGame\\glTF\ABeautifulGame.gltf"
   meshes, materials = loader.load(path)
   for mesh in meshes:
      mesh.set_positions(mesh.positions.array * 100.0)
   # # prepare arrays for collecting vertex attribute data 
   # # into single arrays
   # all_vertices:np.ndarray = None
   # all_normals:np.ndarray = None
   # all_uvs:np.ndarray = None
   # all_indices:np.ndarray = None
   # all_world_bounds:np.ndarray = None
   # transforms:list[np.ndarray] = None
   # vertex_offset = 0

   # all_vertices = meshes[0].positions.array
   # all_normals = meshes[0].normals.array if meshes[0].normals is not None else None
   # all_uvs = meshes[0].uv.array if meshes[0].uv is not None else None
   # all_indices = meshes[0].vertex_indices
   # vertex_offset += meshes[0].n_vertices
   # all_world_bounds = meshes[0].world_bounds if meshes[0].world_bounds is not None else None
   # transforms = [meshes[0].transform.matrix]

   # # collect all mesh data into single arrays, 
   # # adjust indices array by indices += mesh vertex count
   # mesh_count = len(meshes)
   # tri_count = sum(mesh.n_triangles for mesh in meshes)
   # for mesh in meshes[1:]:

   #    all_vertices = np.concatenate([all_vertices, mesh.positions.array], axis=0)

   #    all_normals = np.concatenate([all_normals, mesh.normals.array], axis=0) if all_normals is not None and mesh.normals is not None else all_normals

   #    all_uvs = np.concatenate([all_uvs, mesh.uv.array], axis=0) if all_uvs is not None and mesh.uv is not None else all_uvs

   #    # adjust indices offset
   #    adjusted_indices = mesh.vertex_indices + vertex_offset
   #    all_indices = np.concatenate([all_indices, adjusted_indices], axis=0)

   #    all_world_bounds = np.concatenate([all_world_bounds, mesh.world_bounds], axis=0) if all_world_bounds is not None and mesh.world_bounds is not None else all_world_bounds

   #    # store local to world transforms
   #    transforms.append(mesh.transform.matrix)

   #    vertex_offset += mesh.n_vertices



   # # store contagious arrays
   # all_obj_vertices_cont = np.ascontiguousarray(all_vertices,dtype=np.float32)
   # all_normals_cont:np.ndarray = np.ascontiguousarray(all_normals, dtype=np.float32)
   # all_uvs_cont:np.ndarray = np.ascontiguousarray(all_uvs, dtype=np.float32)
   # all_indices_cont:np.ndarray = np.ascontiguousarray(all_indices, dtype=np.int32)
   # all_world_bounds_cont:np.ndarray = np.ascontiguousarray(all_world_bounds, dtype=np.float32)
   # transforms:list[np.ndarray] = transforms

   # world_vertices_list = [mesh.positions.array @ mesh.transform.matrix.T for mesh in meshes]
   # world_vertices = np.concatenate(world_vertices_list, axis=0)

   # print("vertices:", world_vertices.shape)
   # print("indices:", all_indices.shape)
   # print("max index:", all_indices.max())


   # # calculate bvh
   # bvh_nodes_arr, ordered_triangles = bvh.calculate_bvh(all_world_bounds_cont,4)


   # print(bvh_nodes_arr.shape)
   # # visualize only interior nodes
   # leaf_filter = bvh_nodes_arr["nTris"] == 0
   # bounds_min = bvh_nodes_arr["bounds_min"][leaf_filter]
   # bounds_max = bvh_nodes_arr["bounds_max"][leaf_filter]
   
   # bvh_bounds = np.ascontiguousarray(
   #  np.stack([bounds_min, bounds_max], axis=1)
   # )

   # plot_mesh_data(world_vertices,
   #                all_indices,
   #                all_normals,
   #                all_uvs, 
   #                bvh_bounds)

   tracer = PathTracer(debug,meshes)
   vs = Visualizer(meshes)
   # setup camera
   cam = PerspectiveCamera(Transform.identity(),120,film=Film(1920,1080))

   def render_on_keypress():
      cam.set_world_transform(vs.get_camera_transform())
      print(vs.get_camera_transform().matrix)
      rays = cam.generate_camera_rays()
      tracer.render(debug,rays)
      
   vs.add_key_callback('t',render_on_keypress)

   # debug rays and bounds
   cam.set_world_transform(Transform(Transform.translate(0,0,30).matrix @ Transform.scale(1,1,-1).matrix))
   rays = cam.generate_camera_rays()
   indices = np.indices((1080,1920))
   indices = np.stack([indices[0],indices[1]],axis=2)
   mask = ((indices[:,:,0] % 100 == 0) & (indices[:,:,1] % 100 == 0))
   ray_os = rays[mask][:,0:3].copy().reshape((-1,3))
   ray_ds = rays[mask][:,4:7].copy().reshape((-1,3))
   ray_actor = debugging.Gizmo().draw_ray_vtk(ray_os,ray_ds)
   del ray_os
   del ray_ds
   vs.add_actors([ray_actor])
   def show_debug_rays_per_frame():
      log.debug("VTK StartEvent:=======")
      vs.follow_scene_cam(ray_actor)
   vs.add_key_callback('t',show_debug_rays_per_frame)

   all_world_bounds = np.ascontiguousarray(np.concatenate([mesh.world_bounds for mesh in meshes],axis=0,dtype=np.float32))
   bvh_nodes,_ = bvh.calculate_bvh(all_world_bounds,4)
   bounds_min = bvh_nodes["bounds_min"][:, :3]
   bounds_max = bvh_nodes["bounds_max"][:, :3]
   bvh_bounds = np.stack([bounds_min, bounds_max], axis=1)

   del all_world_bounds
   del bvh_nodes
   bounds_actors = debugging.Gizmo().draw_aabb3_vtk(bvh_bounds,max_count=50)
   vs.add_actors(bounds_actors)
   vs.start()


if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("-debug","--debug",action="store_true",dest="debug")
   args = parser.parse_args()
   main(args.debug)


   
