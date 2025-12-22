

from dataclasses import dataclass
import numpy as np
from typing import Union

from .vector import Vec3

@dataclass(slots=True)
class Bounds3:
   pMin:Vec3
   pMax:Vec3

   def __init__(self, pMin:Union[np.ndarray,Vec3,list]=None, pMax:Union[np.ndarray,Vec3,list]=None):
      if isinstance(pMin, Vec3) and isinstance(pMax, Vec3):
         self.pMin = pMin
         self.pMax = pMax
         return
      
      if pMin is None:
         self.pMin = Vec3(np.inf, np.inf, np.inf)
      else:
         self.pMin = Vec3(pMin[0], pMin[1], pMin[2])
      
      if pMax is None:
         self.pMax = Vec3(-np.inf, -np.inf, -np.inf)
      else:
         self.pMax = Vec3(pMax[0], pMax[1], pMax[2])

def union(b1:Bounds3, b2:Bounds3) -> Bounds3:
   """Compute the union of two Bounds3 objects."""
   new_pMin = np.minimum(b1.pMin.as_numpy(), b2.pMin.as_numpy())
   new_pMax = np.maximum(b1.pMax.as_numpy(), b2.pMax.as_numpy())
   return Bounds3(new_pMin, new_pMax)

def union(b1:Bounds3, p:Union[np.ndarray,Vec3])-> Bounds3:
   """Compute the union of a Bounds3 object and a point."""
   if isinstance(p, Vec3):
      p = p.as_numpy()
   new_pMin = np.minimum(b1.pMin.as_numpy(), p)
   new_pMax = np.maximum(b1.pMax.as_numpy(), p)
   return Bounds3(new_pMin, new_pMax)

def intersect(b1:Bounds3, b2:Bounds3) -> Bounds3:
   """Compute the intersection of two Bounds3 objects."""
   new_pMin = np.maximum(b1.pMin.as_numpy(), b2.pMin.as_numpy())
   new_pMax = np.minimum(b1.pMax.as_numpy(), b2.pMax.as_numpy())
   return Bounds3(new_pMin, new_pMax)

def overlaps(b1:Bounds3, b2:Bounds3) -> bool:
   """Check if two Bounds3 objects overlap."""
   x = (b1.pMax[0] >= b2.pMin[0]) and (b1.pMin[0] <= b2.pMax[0])
   y = (b1.pMax[1] >= b2.pMin[1]) and (b1.pMin[1] <= b2.pMax[1])
   z = (b1.pMax[2] >= b2.pMin[2]) and (b1.pMin[2] <= b2.pMax[2])
   return x and y and z

def diagonal(b:Bounds3):
   """Compute the diagonal of a Bounds3 object."""
   return b.pMax - b.pMin

def surface_area(b:Bounds3) -> float:
   """Compute the surface area of a Bounds3 object."""
   d = diagonal(b)
   return 2 * (d[0]*d[1] + d[0]*d[2] + d[1]*d[2])

def volume(b:Bounds3) -> float:
   """Compute the volume of a Bounds3 object."""
   d = diagonal(b)
   return d[0] * d[1] * d[2]

def maximum_extent(b:Bounds3) -> int:
   """Return the axis with the maximum extent of a Bounds3 object."""
   d = diagonal(b)
   if d[0] > d[1] and d[0] > d[2]:
      return 0
   elif d[1] > d[2]:
      return 1
   else:
      return 2