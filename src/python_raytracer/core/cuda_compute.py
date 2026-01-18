import numpy as np
import cupy as cp

class Texture:
   __slots__ = ['H','W','array']
   def __init__(self,h,w):
      self.H = h
      self.W = w
      self.array = cp.zeros(shape=(h,w,4),dtype=cp.float32)
   def set_array(self,np_arr:np.ndarray):
      assert np_arr.ndim == 3
      assert np_arr.shape == (self.H,self.W,4)
      assert np_arr.dtype == np.float32
      self.array = cp.array(np_arr,dtype=cp.float32)

class CudaKernel:

   __slots__ = ['out_tex']

   # def __init__(self,path,tex_h,tex_w):
   #    self.out_tex = Texture(tex_h,tex_w)
   # def run(self)