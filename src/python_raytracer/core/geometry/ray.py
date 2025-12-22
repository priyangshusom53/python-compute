
import numpy as np
from .vector import Vec4

class Ray:

   def __init__(self, origin: Vec4, direction: Vec4):
      self.origin = origin
      self.direction = direction
      self.t = 0.0


