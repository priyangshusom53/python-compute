
import numpy as np

class Camera:

   def __init__(self, position:np.ndarray, rotation:np.ndarray, fov:float):
      self.position = position
      self.rotation = rotation
      self.fov = fov