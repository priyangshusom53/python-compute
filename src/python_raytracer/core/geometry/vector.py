
from typing import Union
import numpy as np

from .transformation import Transform


class Vec4:

   __slots__ = ["_v"]

   def __init__(self,x:float=None, y:float=None, z:float=None, w:float=None):
      if None in (x, y, z, w):
         self._v = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
      else:
         self._v =  np.array([x, y, z, w], dtype=np.float32)

   def from_array(self, arr:list[float]):
      if len(arr) != 4:
         raise ValueError("Array must have exactly 4 elements.")
      self._v = np.array(arr, dtype=np.float32)

   def __len__(self):
      return 4
   
   def __iter__(self):
      return iter(self._v)

   def __getitem__(self, index:int) -> float:
      return self._v[index]
   
   def __setitem__(self, index:int, value:float):
      self._v[index] = value

   # -----------------------
   # Attribute access
   # -----------------------
   @property
   def x(self): return self._v[0]

   @x.setter
   def x(self, value): self._v[0] = value

   @property
   def y(self): return self._v[1]

   @y.setter
   def y(self, value): self._v[1] = value

   @property
   def z(self): return self._v[2]

   @z.setter
   def z(self, value): self._v[2] = value

   @property
   def w(self): return self._v[3]

   @w.setter
   def w(self, value): self._v[3] = value

   # -----------------------
   # NumPy interoperability
   # -----------------------
   def as_numpy(self):
      """Explicit NumPy view (no copy)."""
      return self._v

   def __array__(self, dtype=None):
      """Allows np.array(v) to work."""
      return self._v.astype(dtype, copy=False) if dtype else self._v

   # -----------------------
   # Math ops
   # -----------------------
   def __add__(self, other):
      return Vec4(*(self._v + other))

   def __sub__(self, other):
      return Vec4(*(self._v - other))

   def __mul__(self, scalar):
      return Vec4(*(self._v * scalar))

   __rmul__ = __mul__

   def __neg__(self):
      return Vec4(*(-self._v))
   
   @staticmethod
   def dot(v1:'Vec4', v2:'Vec4') -> float:
      return float(np.dot(v1._v, v2._v))
   
   @staticmethod
   def cross(v1:'Vec4', v2:'Vec4') -> 'Vec4':
      cross_prod = np.cross(v1._v[:3], v2._v[:3])
      return Vec4(cross_prod[0], cross_prod[1], cross_prod[2], 0.0)
   
   @staticmethod
   def normalize(v:'Vec4') -> 'Vec4':
      norm = np.linalg.norm(v._v[:3])
      if norm == 0:
         raise ValueError("Cannot normalize zero-length vector.")
      normalized = v._v[:3] / norm
      return Vec4(normalized[0], normalized[1], normalized[2], v._v[3])


class Vec3:

   __slots__ = ["_v"]

   def __init__(self,x:float=None, y:float=None, z:float=None):
      if None in (x, y, z):
         self._v = np.array([0.0, 0.0, 0.0], dtype=np.float32)
      else:
         self._v =  np.array([x, y, z], dtype=np.float32)

   def from_array(self, arr:list[float]):
      if len(arr) != 3:
         raise ValueError("Array must have exactly 3 elements.")
      self._v = np.array(arr, dtype=np.float32)

   def __len__(self):
      return 3
   
   def __iter__(self):
      return iter(self._v)

   def __getitem__(self, index:int) -> float:
      return self._v[index]
   
   def __setitem__(self, index:int, value:float):
      self._v[index] = value

   # -----------------------
   # Attribute access
   # -----------------------
   @property
   def x(self): return self._v[0]

   @x.setter
   def x(self, value): self._v[0] = value

   @property
   def y(self): return self._v[1]

   @y.setter
   def y(self, value): self._v[1] = value

   @property
   def z(self): return self._v[2]

   @z.setter
   def z(self, value): self._v[2] = value

   # -----------------------
   # NumPy interoperability
   # -----------------------
   def as_numpy(self):
      """Explicit NumPy view (no copy)."""
      return self._v

   def __array__(self, dtype=None):
      """Allows np.array(v) to work."""
      return self._v.astype(dtype, copy=False) if dtype else self._v

   # -----------------------
   # Math ops
   # -----------------------
   def __add__(self, other):
      return Vec3(*(self._v + other))

   def __sub__(self, other):
      return Vec3(*(self._v - other))

   def __mul__(self, scalar):
      return Vec3(*(self._v * scalar))

   __rmul__ = __mul__

   def __neg__(self):
      return Vec3(*(-self._v))
   
   @staticmethod
   def dot(v1:'Vec3', v2:'Vec3') -> float:
      return float(np.dot(v1._v, v2._v))
   
   @staticmethod
   def cross(v1:'Vec3', v2:'Vec3') -> 'Vec3':
      cross_prod = np.cross(v1._v[:3], v2._v[:3])
      return Vec3(cross_prod[0], cross_prod[1], cross_prod[2])
   
   @staticmethod
   def normalize(v:'Vec3') -> 'Vec3':
      norm = np.linalg.norm(v._v[:3])
      if norm == 0:
         raise ValueError("Cannot normalize zero-length vector.")
      normalized = v._v[:3] / norm
      return Vec3(normalized[0], normalized[1], normalized[2])


class Vec3Buffer:
   """ uses numpy array with shape (N,3) as buffer for 3D vectors """

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
   
   def flatten_array(self) -> np.ndarray:
      """Return a flattened view of the buffer."""
      return self._buffer.flatten()


class Vec2Buffer:
   """ uses numpy array with shape (N,2) as buffer for 2D vectors """

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
   
   def flatten_array(self) -> np.ndarray:
      """Return a flattened view of the buffer."""
      return self._buffer.flatten()


class Vec4Buffer:
   """ uses numpy array with shape (N,4) as buffer for 4D vectors """

   __slots__ = ["_buffer"]

   def __init__(self, size:int, data:np.ndarray=None):
      if data is None:
            self._buffer = np.zeros((size, 4), dtype=np.float32)
            self._buffer[:, 3] = 1.0

      elif isinstance(data, np.ndarray):
         if data.ndim != 2 or data.shape[1] != 4:
               raise ValueError("Array must have shape (N, 4).")

         self._buffer = np.ascontiguousarray(data, dtype=np.float32)

   def __len__(self):
      return self._buffer.shape[0]
      
   @property
   def array(self):
      return self._buffer
   
   def flatten_array(self) -> np.ndarray:
      """Return a flattened view of the buffer."""
      return self._buffer.flatten()
   
   def transform(self, matrix4x4:Union[np.ndarray,Transform]):

      if isinstance(matrix4x4, Transform):
         matrix4x4 = matrix4x4._matrix

      if matrix4x4.shape != (4, 4):
         raise ValueError("Matrix must have shape (4, 4).")
      self._buffer = self._buffer @ matrix4x4.T
   



