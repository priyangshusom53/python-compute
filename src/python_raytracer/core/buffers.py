
from typing import Union

import numpy as np

from python_raytracer.transformation import Transform

class Vec3Buffer:
   """ uses numpy array with shape (N,3) and dtype=float32 as buffer for 3D vectors """

   __slots__ = ["_buffer"]

   def __init__(self, size:int, data:np.ndarray=None):
      if data is None:
            self._buffer = np.zeros((size, 3), dtype=np.float32)

      elif isinstance(data, np.ndarray):
         if data.ndim != 2 or data.shape[1] != 3:
               raise ValueError("Array must have shape (N, 3).")

         self._buffer = np.ascontiguousarray(data, dtype=np.float32)

   def __len__(self):
      return self._buffer.shape[0]
      
   @property
   def array(self):
      return self._buffer
   
   def flatten_array(self):
      """Return a flattened view of the buffer."""
      return self._buffer.ravel()


class Vec2Buffer:
   """ uses numpy array with shape (N,2) and dtype=float32 as buffer for 2D vectors """

   __slots__ = ["_buffer"]

   def __init__(self, size:int, data:np.ndarray=None):
      if data is None:
            self._buffer = np.zeros((size, 2), dtype=np.float32)

      elif isinstance(data, np.ndarray):
         if data.ndim != 2 or data.shape[1] != 2:
               raise ValueError("Array must have shape (N, 2).")

         self._buffer = np.ascontiguousarray(data, dtype=np.float32)

   def __len__(self):
      return self._buffer.shape[0]
      
   @property
   def array(self):
      return self._buffer
   
   def flatten_array(self):
      """Return a flattened view of the buffer."""
      return self._buffer.ravel()


class Vec4Buffer:
   """ uses numpy contagious array with shape (N,4) and dtype=float32 as buffer for 4D vectors """

   __slots__ = ["_buffer"]

   def __init__(self, size:int, data:np.ndarray=None):
      if data is None:
            self._buffer = np.zeros((size, 4), dtype=np.float32)

      elif isinstance(data, np.ndarray):
         if data.ndim != 2 or data.shape[1] != 4:
               raise ValueError("Array must have shape (N, 4).")

         self._buffer = np.ascontiguousarray(data, dtype=np.float32)

   def __len__(self):
      return self._buffer.shape[0]
      
   @property
   def array(self):
      return self._buffer
   
   def flatten_array(self):
      """Return a flattened view of the buffer."""
      return self._buffer.ravel()
   
   def transform(self, matrix4x4:Union[np.ndarray,Transform]):
      if isinstance(matrix4x4, Transform):
         matrix4x4 = matrix4x4.matrix

      if matrix4x4.shape != (4, 4):
         raise ValueError("Matrix must have shape (4, 4).")
      self._buffer = self._buffer @ matrix4x4.T