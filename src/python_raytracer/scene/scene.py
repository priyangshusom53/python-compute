
import numpy as np
import enum

from .node import Node


class NodeType(enum.Enum):
   PARENT = 0
   MESH = 1
   CAMERA = 2
   EMPTY = 3


class Scene:
   
   def __init__(self, name:str=None, rootnodes:list[Node]=None):
      self.name = name if name is not None else "Scene"
      self.rootnodes = rootnodes if rootnodes is not None else []

   def add_root_node(self, node:Node):
      self.rootnodes.append(node)

   

      
      