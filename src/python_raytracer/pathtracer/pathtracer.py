import numpy as np
import cupy as cp
import logging
import random

from python_raytracer.core.geometry.transformation import Transform
from python_raytracer.core.renderer.renderer import Renderer
from python_raytracer.core.material.pbr_material import PBRMaterial
from python_raytracer.core.geometry.triangle_mesh import TriangleMesh
from python_raytracer.bvh import bvh
from python_raytracer.core.camera.simplecamera import (PerspectiveCamera,Film)
from python_raytracer.plots.imageplot import save_to_image


class PathTracer(Renderer):

   def _make_bvh_from_meshes(meshes:list[TriangleMesh]):
      all_world_bounds = meshes[0].world_bounds
      for mesh in meshes[1:]:
         all_world_bounds = np.concatenate([all_world_bounds, mesh.world_bounds], axis=0)
      all_world_bounds_cont:np.ndarray = np.ascontiguousarray(all_world_bounds, dtype=np.float32)
      bvh_nodes, ordered_triangles = bvh.calculate_bvh(all_world_bounds_cont,4)
      return (bvh_nodes,ordered_triangles)

   # buffer numpy arrays should be C_CONTIGUOUS and contagious array
   def render(self,
              debug:bool,
              cam_transform:Transform,
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

      index_buff = np.ascontiguousarray(index_buff)
      vertex_buff = np.ascontiguousarray(vertex_buff)
      normal_buff = np.ascontiguousarray(normal_buff)
      uv_buff = np.ascontiguousarray(uv_buff)
      d_index_buff = cp.asarray(index_buff)
      d_vertex_buff = cp.asarray(vertex_buff)
      d_normal_buff = cp.asarray(normal_buff)
      d_uv_buff = cp.asarray(uv_buff)
      # calculate bvh and ordered triangles
      bvh_nodes, ordered_triangles = self._make_bvh_from_meshes(meshes)
      if debug:
         assert ordered_triangles.dtype == np.int32
      # <Make Triangle array from ordered triangles>
      d_triangles_of_mesh = ordered_triangles[ordered_triangles>=0 and ordered_triangles < meshes[0].n_triangles]
      d_triangles_mesh_idx = np.repeat(d_triangles_of_mesh.shape[0],0)
      d_triangles = np.concatenate((d_triangles_mesh_idx,d_triangles_of_mesh),axis=-1,dtype=np.int32)
      for mesh_idx, mesh in enumerate(meshes,start=1):
         d_triangles_of_mesh = ordered_triangles[ordered_triangles>=meshes[mesh_idx-1] and ordered_triangles < mesh.n_triangles]
         d_triangles_mesh_idx = np.repeat(d_triangles_of_mesh.shape[0],mesh_idx)
         d_triangles = np.concatenate((d_triangles,np.concatenate((d_triangles_mesh_idx,d_triangles_of_mesh),axis=-1,dtype=np.int32)),axis=0)
      d_triangles = cp.asarray(np.ascontiguousarray(d_triangles,dtype=np.int32))

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

      # get camera rays
      cam = PerspectiveCamera(cam_transform,120,film=Film(1920,1080))
      rays = cam.generate_camera_rays()
      # set output image
      

   def render_screen_extent(self,scene,extent):
      raise NotImplementedError