import numpy as np
import cupy as cp
import logging
import random

from python_raytracer.core.renderer.renderer import Renderer
from python_raytracer.core.material.pbr_material import PBRMaterial
from python_raytracer.core.geometry.triangle_mesh import TriangleMesh
from python_raytracer.bvh import bvh
from python_raytracer.core.camera import simplecamera


class PathTracer(Renderer):

   def _make_bvh_from_meshes(meshes:list[TriangleMesh]):
      all_world_bounds = meshes[0].world_bounds
      for mesh in meshes[1:]:
         all_world_bounds = np.concatenate([all_world_bounds, mesh.world_bounds], axis=0)
      all_world_bounds_cont:np.ndarray = np.ascontiguousarray(all_world_bounds, dtype=np.float32)


   def render(self,
              debug:bool,
              index_buff:np.ndarray,
              vertex_buff:np.ndarray,
              normal_buff:np.ndarray,
              uv_buff:np.ndarray,
              num_triangles:int,
              num_vertices:np.ndarray,
              meshes:list[TriangleMesh],
              materials:np.ndarray=None):
      if debug:
         assert index_buff.dtype == np.int32
         assert index_buff.shape == (num_triangles,3)
         assert vertex_buff.dtype == np.float32
         assert vertex_buff.shape == (num_vertices,4)
         assert normal_buff.shape == (num_vertices,4)
         assert uv_buff.shape == (num_vertices,2)

      d_index_buff = cp.asarray(index_buff)
      d_vertex_buff = cp.asarray(vertex_buff)
      d_normal_buff = cp.asarray(normal_buff)
      d_uv_buff = cp.asarray(uv_buff)

      # make place holder materials
      num_materials = 10
      material_dtype = np.dtype([
         ("baseColorFactor", np.float32,4),
         ("metallic", np.float32),
         ("roughness", np.float32),
         ("_pad", np.float32, 2),
      ])
      assert material_dtype.itemsize == 32
      materials = np.zeros(num_materials,dtype=material_dtype)
      for i in range(0,10):
         materials[i]["baseColorFactor"] = [random.random(),random.random(),random.random(),1]


   def render_screen_extent(self,scene,extent):
      raise NotImplementedError