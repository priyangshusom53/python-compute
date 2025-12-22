
from dataclasses import dataclass
import numpy as np

from .bounds import Bounds3

@dataclass
class Triangle:
   mesh_id: int
   vertex_indices_id: int
   material_id: int
   vectex_attribute_offset: int

def bound(triangle:Triangle, positions:np.ndarray) -> Bounds3:
   v0 = positions[triangle.vertex_indices_id + 0][:3]
   v1 = positions[triangle.vertex_indices_id + 1][:3]
   v2 = positions[triangle.vertex_indices_id + 2][:3]

   pMin = np.minimum(np.minimum(v0, v1), v2)
   pMax = np.maximum(np.maximum(v0, v1), v2)

   return Bounds3(pMin, pMax)

def triangle_intersection(triangle:Triangle, ray_origin, ray_direction):
   # Placeholder for actual intersection logic
   pass



