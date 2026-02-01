import argparse
import logging

import numpy as np

from python_raytracer.core.log import logger
# from python_raytracer.core import debugging
from python_raytracer.gltf_loader import load_gltf_as_triangle_meshes
# from python_raytracer.transformation import Transform
from python_raytracer.triangle_mesh import (TriangleMesh, TriangleMesh4)

# from python_raytracer.pathtracer.pathtracer import PathTracer
from python_raytracer.plots.o3dplots import plot_mesh_data
from python_raytracer.bvh import bvh
from python_raytracer.plots.vtkvisualizer import Visualizer

from python_raytracer import _pathtracer_

# -debug or --debug flag to enable debugging
def main(debug:bool):

   # setup log
   logger.log_config()
   log = logging.getLogger(__name__)

   # load the scene
   path = r"D:\3D Models\\gltf_sample_scenes\\glTF-Sample-Assets\\Models\ABeautifulGame\\glTF\ABeautifulGame.gltf"
   path = r"D:\3D Models\sponza_gltf\scene.gltf"
   # meshes, materials = loader.load(path)
   # for mesh in meshes:
   #    mesh.set_positions(mesh.positions.array * 100.0)
   

   # tracer = PathTracer(debug,meshes)
   # vs = Visualizer(meshes)
   # # setup camera
   # cam = PerspectiveCamera(Transform.identity(),120,film=Film(1920,1080))

   # def render_on_keypress():
   #    cam.set_world_transform(vs.get_camera_transform())
   #    print(vs.get_camera_transform().matrix)
   #    rays = cam.generate_camera_rays()
   #    tracer.render(debug,rays)
      
   # vs.add_key_callback('t',render_on_keypress)

   # # debug rays and bounds
   # cam.set_world_transform(Transform(Transform.translate(0,0,30).matrix @ Transform.scale(1,1,-1).matrix))
   # rays = cam.generate_camera_rays()
   # indices = np.indices((1080,1920))
   # indices = np.stack([indices[0],indices[1]],axis=2)
   # mask = ((indices[:,:,0] % 100 == 0) & (indices[:,:,1] % 100 == 0))
   # ray_os = rays[mask][:,0:3].copy().reshape((-1,3))
   # ray_ds = rays[mask][:,4:7].copy().reshape((-1,3))
   # ray_actor = debugging.Gizmo().draw_ray_vtk(ray_os,ray_ds)
   # del ray_os
   # del ray_ds
   # vs.add_actors([ray_actor])
   # def show_debug_rays_per_frame():
   #    log.debug("VTK StartEvent:=======")
   #    vs.follow_scene_cam(ray_actor)
   # vs.add_key_callback('t',show_debug_rays_per_frame)

   # all_world_bounds = np.ascontiguousarray(np.concatenate([mesh.world_bounds for mesh in meshes],axis=0,dtype=np.float32))
   # bvh_nodes,_ = bvh.calculate_bvh(all_world_bounds,4)
   # bounds_min = bvh_nodes["bounds_min"][:, :3]
   # bounds_max = bvh_nodes["bounds_max"][:, :3]
   # bvh_bounds = np.stack([bounds_min, bounds_max], axis=1)

   # del all_world_bounds
   # del bvh_nodes
   # bounds_actors = debugging.Gizmo().draw_aabb3_vtk(bvh_bounds,max_count=50)
   # vs.add_actors(bounds_actors)
   # vs.start()

   meshes = load_gltf_as_triangle_meshes(path,0,True)
   hands = [mesh.handedness for mesh in meshes]
   print(hands)
   meshes4 = [TriangleMesh4(mesh) for mesh in meshes]
   vtk_vis = Visualizer(meshes=meshes4)
   vtk_vis.start()


if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("-debug","--debug",action="store_true",dest="debug")
   args = parser.parse_args()
   main(args.debug)


   
