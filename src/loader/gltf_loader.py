
from scene_loader import Loader
from ..core.geometry.triangle_mesh import TriangleMesh

import trimesh 
import open3d as o3d
import numpy as np

class GLTFLoader(Loader):

   def load(self, path:str):

      scene = trimesh.load(path,force='scene')

      geomtries = []
      if isinstance(scene,trimesh.Trimesh):
         meshes = [scene]
      else:
         meshes = []
         for node_name in scene.graph.nodes:
            transform, geometry_name = scene.graph[node_name]
            if geometry_name is None:
                continue
            geometry = scene.geometry[geometry_name]
            if isinstance(geometry, trimesh.Trimesh):
                # Apply world transform from scene graph
                transformed_mesh = geometry.copy()
                transformed_mesh.apply_transform(transform)
                meshes.append(transformed_mesh)
      return scene
   
print(GLTFLoader().load('D:\\3D Models\\sponza_gltf\\scene.gltf'))
