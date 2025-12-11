
import numpy as np

from geometry.triangle_mesh import TriangleMesh
from camera.camera import Camera

import enum

class NodeType(enum.Enum):
   PARENT = 0
   MESH = 1
   CAMERA = 2
   EMPTY = 3

class Node:

   def __init__(
         self, 
         local_to_global:np.ndarray,
         type:NodeType,
         node:'Node' = None, 
         mesh:TriangleMesh = None,
         cam:Camera = None
   ):
      self.local_to_global = local_to_global
      self.type = type
      self.node = node
      self.mesh = mesh
      self.cam = cam



class Scene:
   
   def __init__(self, root:Node):
      self.root = root

      
      