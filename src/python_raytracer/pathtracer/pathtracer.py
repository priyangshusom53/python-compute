import numpy as np
import cupy as cp

from python_raytracer.core.renderer.renderer import Renderer

class PathTracer(Renderer):

   def render(self,scene,index_buff:np.ndarray,vertex_buff:np.ndarray,normal_buff:np.ndarray,uv_buff:np.ndarray,num_triangles:int,num_vertices:np.ndarray,debug:bool):
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


   def render_screen_extent(self,scene,extent):
      raise NotImplementedError