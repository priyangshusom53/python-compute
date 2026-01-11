import numpy as np

from python_raytracer.core.renderer.renderer import Renderer

class PathTracer(Renderer):

   def render(self,scene):
      pass

   def render_screen_extent(self,scene,extent):
      raise NotImplementedError