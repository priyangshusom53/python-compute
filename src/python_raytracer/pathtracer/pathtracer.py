import numpy as np
import cupy as cp
import logging
import random
from pathlib import Path

from python_raytracer.core.geometry.transformation import Transform
from python_raytracer.core.renderer.renderer import Renderer
from python_raytracer.core.material.pbr_material import PBRMaterial
from python_raytracer.core.geometry.triangle_mesh import TriangleMesh
from python_raytracer.bvh import bvh
from python_raytracer.core.camera.simplecamera import (PerspectiveCamera,Film)
from python_raytracer.plots.imageplot import save_to_image


class PathTracer(Renderer):

   __slots__ = ['n_tris',
                'n_verts',
                'triangles',
                'index_buff',
                'vert_buff',
                'norm_buff',
                'uv_buff',
                'np_gpu_meshes',
                'bvh_nodes',

                'd_index_buff',
                'd_vert_buff',
                'd_norm_buff',
                'd_uv_buff',
                'd_triangles',
                'd_meshes',
                'd_bvh_nodes',
                'd_materials']

   def _create_attribute_buffers(self,meshes:list[TriangleMesh]):
      index_buff = np.concatenate([mesh.vertex_indices for mesh in meshes],axis=0,dtype=np.int32)
      n_tris = index_buff.shape[0]
      vertex_buff = np.concatenate([mesh.positions.array @ mesh.transform.matrix.T for mesh in meshes],axis=0,dtype=np.float32)
      n_verts = vertex_buff.shape[0]
      normal_buff = np.concatenate([mesh.normals.array for mesh in meshes],axis=0,dtype=np.float32)
      uv_buff = np.concatenate([mesh.uv.array for mesh in meshes],axis=0,dtype=np.float32)
      return (n_tris,
              n_verts,
              index_buff,
              vertex_buff,
              normal_buff,
              uv_buff)

   def _create_gpu_trimesh_buffer(self,meshes:list[TriangleMesh]):
      num_meshes = len(meshes)

      # corresponds to numTriangles in gpu TriangleMesh
      ntris_arr = np.array([mesh.n_triangles for mesh in meshes],dtype=np.int32)
      # corresponds to firstTriangleIdx in gpu TriangleMesh
      ftidx_arr = np.concatenate(([0], np.cumsum(ntris_arr[:-1], dtype=np.int32)))
      # corresponds to numVertices in gpu TriangleMesh
      nverts_arr = np.array([mesh.n_vertices for mesh in meshes],dtype=np.int32)
      # corresponds to firstVertexIdx in gpu TriangleMesh
      fvertidx_arr = np.concatenate(([0], np.cumsum(nverts_arr[:-1], dtype=np.int32)))
      # materialIdx in gpu TriangleMesh
      matidx_arr = np.array([mesh.material_idx for mesh in meshes],dtype=np.int32)
      # make padding array of length 3 ints
      pad_arr = np.zeros((num_meshes,3),dtype=np.int32)
      # make Transform array of 2 mat4x4 shape(N,8,4)
      t_arr = np.stack(
         [np.concatenate((mesh.transform.matrix, mesh.transform.inverse_matrix), 
         axis=0)for mesh in meshes],
         axis=0).astype(np.float32)
      return (num_meshes, 
              ftidx_arr, 
              ntris_arr, 
              nverts_arr, 
              fvertidx_arr, 
              matidx_arr, 
              pad_arr, 
              t_arr)

   def _make_bvh_from_meshes(self,meshes:list[TriangleMesh]):
      all_world_bounds = np.concatenate([mesh.world_bounds for mesh in meshes],axis=0)
      
      all_world_bounds_cont:np.ndarray = np.ascontiguousarray(all_world_bounds, dtype=np.float32)
      bvh_nodes, ordered_triangles = bvh.calculate_bvh(all_world_bounds_cont,4)
      return (bvh_nodes,ordered_triangles)
   
   def _compute_gpu_triangles_from_ordered_triangles(self,ordered_tris:np.ndarray):
      
      num_meshes = len(self.np_gpu_meshes)
      triangle_blocks = []
      for mesh_idx,mesh in enumerate(self.np_gpu_meshes):
         start = mesh['firstTriangleIdx']
         end = start + mesh['numTriangles'] # exclusive [start,end)
         triangles_of_mesh = ordered_tris[(ordered_tris >= start) & (ordered_tris < end)]
      
         triangles_mesh_idx = np.full(shape=(triangles_of_mesh.shape[0],),fill_value=mesh_idx,dtype=np.int32)

         triangles = np.stack((triangles_mesh_idx, triangles_of_mesh),axis=1).astype(np.int32)

         triangle_blocks.append(triangles)
      triangles = np.concatenate(triangle_blocks,axis=0,dtype=np.int32)
      return triangles

   # buffer numpy arrays should be C_CONTIGUOUS and contagious array
   def render(self,
              debug:bool,
              cam_transform:Transform,
              materials:np.ndarray=None):
      logger = logging.getLogger(__name__)
      if debug:
         logger.info(f"{__name__} running in debug mode")

      # # <Make cupy arrays>
      # d_index_buff = cp.asarray(self.index_buff)
      # d_vertex_buff = cp.asarray(self.vert_buff)
      # d_normal_buff = cp.asarray(self.norm_buff)
      # d_uv_buff = cp.asarray(self.uv_buff)
      # d_triangles = cp.asarray(self.triangles)
      # d_meshes = cp.asarray(self.np_gpu_meshes)
      # d_bvh_nodes = cp.asarray(self.bvh_nodes)

      # make place holder materials
      # num_materials = 10
      # material_dtype = np.dtype([
      #    ("baseColorFactor", np.float32,4),
      #    ("metallic", np.float32),
      #    ("roughness", np.float32),
      #    ("_pad", np.float32, 2),
      # ])
      # assert material_dtype.itemsize == 32
      # materials = np.zeros(num_materials,dtype=material_dtype)
      # for i in range(0,10):
      #    materials[i]["baseColorFactor"] = [random.random(),random.random(),random.random(),1]
      # d_materials = cp.asarray(np.ascontiguousarray(materials))

      # get camera rays
      cam = PerspectiveCamera(cam_transform,120,film=Film(1920,1080))
      rays = cam.generate_camera_rays()
      rays[:,:,7] = np.float32(np.inf) # set tmax
      d_rays = cp.asarray(np.ascontiguousarray(rays))
      # set output image
      W, H = 1920, 1080
      d_output = cp.zeros(
         (H, W, 4),
         dtype=cp.float32
      )

      # path from project root top
      kernel_path = ""
      if debug:
         kernel_path = Path(Path(__file__).parent / "cuda/build/ptx/Debug/trace.ptx").resolve()
      else:
         kernel_path = Path(Path(__file__).parent / "cuda/build/ptx/Release/trace.ptx").resolve()

      module = cp.RawModule(
         path=str(kernel_path),
         options=("--std=c++14",),
      )
      trace_kernel = module.get_function("trace_scene")
      threads = (16, 16, 1)
      blocks = (
         (W + threads[0] - 1) // threads[0],
         (H + threads[1] - 1) // threads[1],
         1
      )

      logger.info(f"Invoking kernel: trace_scene")
      trace_kernel(
         blocks,
         threads,
         (
            d_rays,                     # Ray*
            np.int32(W),
            np.int32(H),
            self.d_index_buff,          # int3*
            self.d_vert_buff,           # float4*
            self.d_norm_buff,           # float4*
            self.d_uv_buff,             # float2*
            np.int32(self.n_verts),
            self.d_meshes,              # TriangleMesh*
            self.d_triangles,           # Triangle*
            np.int32(self.n_tris),
            self.d_bvh_nodes,           # LinearBVHNode*
            self.d_materials,           # PBRMaterial*
            np.int32(10),               # numMaterials
            d_output                    # float4*
         )
      )
      cp.cuda.runtime.deviceSynchronize()
      err = cp.cuda.runtime.getLastError()
      assert err == 0

      # save result to image
      image = cp.asnumpy(d_output)
      save_to_image(image, "output.png")
      logger.info("image saved to output.png")
      
   def __init__(self,meshes:list[TriangleMesh],materials:np.ndarray=None):
      # <Compute mesh attributes>
      n_tris, n_verts, idx_buff, vert_buff, norm_buff, uv_buff = self._create_attribute_buffers(meshes)
      
      self.n_tris = n_tris
      self.n_verts = n_verts
      self.index_buff = np.ascontiguousarray(idx_buff,dtype=np.int32)
      self.vert_buff = np.ascontiguousarray(vert_buff,dtype=np.float32)
      self.norm_buff = np.ascontiguousarray(norm_buff,dtype=np.float32)
      self.uv_buff = np.ascontiguousarray(uv_buff,dtype=np.float32)

      # <Create gpu TriangleMesh array>
      triangle_mesh_dtype = np.dtype([
      ("firstTriangleIdx", np.int32),
      ("numTriangles",     np.int32),
      ("numVertices",      np.int32),
      ("firstVertexIdx",   np.int32),
      ("materialIdx",      np.int32),
      ("pad",              np.int32, (3,)),
      ("transform",        np.float32, (8, 4)),
      ], align=True)
      assert triangle_mesh_dtype.itemsize == 160

      num_meshes,ftidx_arr,ntris_arr,nverts_arr,fvertidx_arr,matidx_arr,pad_arr,t_arr = self._create_gpu_trimesh_buffer(meshes)

      mesh_buffer = np.zeros(num_meshes, dtype=triangle_mesh_dtype)

      mesh_buffer["firstTriangleIdx"] = ftidx_arr
      mesh_buffer["numTriangles"]     = ntris_arr
      mesh_buffer["numVertices"]      = nverts_arr
      mesh_buffer["firstVertexIdx"]   = fvertidx_arr
      mesh_buffer["materialIdx"]      = matidx_arr
      mesh_buffer["pad"]              = pad_arr
      mesh_buffer["transform"]        = t_arr
      self.np_gpu_meshes = mesh_buffer
      assert mesh_buffer.flags["C_CONTIGUOUS"]
      assert mesh_buffer.dtype.itemsize == 160

      # <Compute bvh and ordered triangles>
      bvh_nodes, ordered_triangles = self._make_bvh_from_meshes(meshes)
      assert ordered_triangles.dtype == np.int32
      self.bvh_nodes = bvh_nodes

      # <Make Triangle array from ordered triangles>
      self.triangles = np.ascontiguousarray(self._compute_gpu_triangles_from_ordered_triangles(ordered_triangles))

      # <Make cupy arrays>
      self.d_index_buff = cp.asarray(self.index_buff)
      self.d_vert_buff = cp.asarray(self.vert_buff)
      self.d_norm_buff = cp.asarray(self.norm_buff)
      self.d_uv_buff = cp.asarray(self.uv_buff)
      self.d_triangles = cp.asarray(self.triangles)
      self.d_meshes = cp.asarray(self.np_gpu_meshes.view(np.uint8))
      self.d_bvh_nodes = cp.asarray(self.bvh_nodes)

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
      self.d_materials = cp.asarray(np.ascontiguousarray(materials))

   def render_screen_extent(self,scene,extent):
      raise NotImplementedError