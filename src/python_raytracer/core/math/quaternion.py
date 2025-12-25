

import numpy as np
from typing import Union

class Quaternion:

   __slots__ = ("_q")

   def __init__(self, w:float=1.0, x:float=0.0, y:float=0.0, z:float=0.0):
      self._q = np.array([w, x, y, z], dtype=np.float32)

   def from_array(self, arr: Union[list[float],np.ndarray]):
      if len(arr) != 4:
         raise ValueError("Array must have exactly 4 elements.")
      self._q = np.array(arr, dtype=np.float32)

   def __mul__(self, other: 'Quaternion') -> 'Quaternion':
      if isinstance(other, Quaternion):
         w1, x1, y1, z1 = self._q
         w2, x2, y2, z2 = other._q
         w = w1*w2 - x1*x2 - y1*y2 - z1*z2
         x = w1*x2 + x1*w2 + y1*z2 - z1*y2
         y = w1*y2 - x1*z2 + y1*w2 + z1*x2
         z = w1*z2 + x1*y2 - y1*x2 + z1*w2
         return Quaternion(w, x, y, z)
      return NotImplemented
   
   def __rmul__(self, other: 'Quaternion'):
      return NotImplemented
   
   def __imul__(self, other: 'Quaternion'):
      result = self * other
      self.w, self.x, self.y, self.z = result.w, result.x, result.y, result.z
      return self
   
   def conjugate(self) -> 'Quaternion':
      w, x, y, z = self._q
      return Quaternion(w, -x, -y, -z)
   
   def inverse(self) -> 'Quaternion':
      conj = self.conjugate()
      norm_sq = np.dot(self._q, self._q)
      if norm_sq == 0:
         raise ValueError("Cannot invert a zero-length quaternion.")
      inv_q = conj._q / norm_sq
      return Quaternion(inv_q[0], inv_q[1], inv_q[2], inv_q[3])

   @staticmethod
   def mul(q1:'Quaternion', q2: 'Quaternion') -> 'Quaternion':
      """Left multiply two quaternions."""
      return q1 * q2

   @staticmethod
   def dot(q1:'Quaternion', q2: 'Quaternion') -> float:
      return float(np.dot(q1._q, q2._q))
   
   @staticmethod
   def normalize(q:'Quaternion'):
      norm = np.linalg.norm(q._q)
      if norm == 0:
         raise ValueError("Cannot normalize a zero-length quaternion.")
      q._q /= norm
      return q
   
   @staticmethod
   def mat4x4(q:'Quaternion') -> np.ndarray:
      """Convert quaternion to 4x4 rotation matrix."""
      w, x, y, z = q._q
      return np.array([
         [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w),     0],
         [2*(x*y + z*w),     1 - 2*(x**2 + z**2), 2*(y*z - x*w),     0],
         [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2 + y**2), 0],
         [0,                 0,                 0,                 1]
      ], dtype=np.float32)
   
   @staticmethod
   def slerp(q1:'Quaternion', q2:'Quaternion', t:float) -> 'Quaternion':
      """Spherical linear interpolation between two quaternions."""
      dot_product = Quaternion.dot(q1, q2)
      if dot_product < 0.0:
         q2 = Quaternion(-q2.w, -q2.x, -q2.y, -q2.z)
         dot_product = -dot_product
      if dot_product > 0.9995:
         result = Quaternion(
            q1.w + t*(q2.w - q1.w),
            q1.x + t*(q2.x - q1.x),
            q1.y + t*(q2.y - q1.y),
            q1.z + t*(q2.z - q1.z)
         )
         return Quaternion.normalize(result)
      theta = np.arccos(dot_product)
      thetap = theta * t
      sin_thetap = np.sin(thetap)
      sin_theta = np.sin(theta)
      s0 = np.cos(thetap) - dot_product * sin_thetap / sin_theta
      s1 = sin_thetap / sin_theta
      return Quaternion(
         (s0 * q1.w) + (s1 * q2.w),
         (s0 * q1.x) + (s1 * q2.x),
         (s0 * q1.y) + (s1 * q2.y),
         (s0 * q1.z) + (s1 * q2.z)
      )
   
   @staticmethod
   def rotate(v:'Quaternion', q:'Quaternion'):
      """Rotate vector v by quaternion q."""
      q_vec = v
      q_conj = q.conjugate()
      q_res = q * q_vec * q_conj
      return Quaternion(q_res.x, q_res.y, q_res.z)

   
