
import numpy as np


from ..core.geometry.transformation import Transform

class Node:
   
   def __init__(
         self,
         name:str,
         parent:'Node'=None,
         children:list['Node']=None,
         mesh_indices:list[int]=None,
         camera:Camera=None,
         light:Light=None,
         local_to_world_transform: np.ndarray = None,
   ):
      self.name = name if name is not None else "Node"
      self.parent = parent
      self.children = children if children is not None else []
      self.mesh_indices = mesh_indices if mesh_indices is not None else []
      self.camera = camera
      self.light = light
      self.local_to_world_transform = Transform(local_to_world_transform) if local_to_world_transform is not None else Transform()

   def add_child(self, child:'Node'):
      self.children.append(child)

   def add_mesh_index(self, mesh_index:int):
      self.mesh_indices.append(mesh_index)

   def set_camera(self, camera:Camera):
      self.camera = camera

   def set_local_to_world_transform(self, transform:np.ndarray):
      self.local_to_world_transform = Transform(transform)

   def set_parent(self, parent:'Node'):
      parent.add_child(self)
      local_to_parent = np.matmul(np.linalg.inv(parent.local_to_world_transform._matrix), self.local_to_world_transform._matrix)

      self.parent = parent

      parent_transform = parent.local_to_world_transform._matrix
      child_transform = np.matmul(parent_transform, local_to_parent)

      self.local_to_world_transform = Transform(child_transform)

   @property
   def has_mesh(self) -> bool:
      return len(self.mesh_indices) > 0
   
   @property
   def mesh_count(self) -> int:
      return len(self.mesh_indices)

   @property
   def has_children(self) -> bool:
      return len(self.children) > 0
   
   @property
   def children_count(self) -> int:
      return len(self.children)
   


