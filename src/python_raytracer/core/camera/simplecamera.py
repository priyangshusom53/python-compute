
import numpy as np
import math

from python_raytracer.core.geometry.transformation import Transform

class ScreenWindow:
   __slots__ = ['p_min', 'p_max']

   # pmin and pmax are 2d vector coordinate of pmin and pmax of screen extent in screen space 
   def __init__(self,pmin:list[float], pmax:list[float]):
      assert len(pmin) == 2
      assert len(pmax) == 2
      self.p_min = pmin
      self.p_max = pmax

class Film:
   __slots__ = ['res_x','res_y']

   def __init__(self, x:int, y:int):
      self.res_x = x
      self.res_y = y

class PerspectiveCamera:

   __slots__ = ['fov','film','camera_to_world','camera_to_screen','screen_to_raster', 'raster_to_screen','raster_to_camera']

   def perspective_transform(fov:float, n:float, f:float):
      # <Transfom to perspective view volume>
      persp = Transform(
         matrix=np.array([[1, 0, 0,              0],
                          [0, 1, 0,              0],
                          [0, 0, f/f-n, -f*n/(f-n)],
                          [0, 0, 1,              0]]))
      # <Scale with fov>
      inv_tan_ang = 1 / math.tan(math.radians(fov) / 2)
      scale = Transform.scale(inv_tan_ang, inv_tan_ang, 1)
      return Transform(matrix=np.linalg.matmul(scale.matrix, persp.matrix))

   def raster_to_screen_transform(self,screen:ScreenWindow,film:Film):
      screen_to_ndc = Transform.translate(-screen.p_min[0], -screen.p_max[1],0)
      scale = Transform.scale(1/(screen.p_max[0]-screen.p_min[0]), 1/(screen.p_min[1]-screen.p_max[1]), 1)
      ndc_to_raster = Transform.scale(film.res_x,film.res_y,1)
      screen_to_raster = Transform(matrix=np.linalg.matmul(ndc_to_raster.matrix,(np.linalg.matmul(scale.matrix,screen_to_ndc.matrix))))
      raster_to_screen = Transform.inverse(screen_to_raster)
      return raster_to_screen

   def __init__(self, 
                cam_to_world:Transform, 
                fov:float,
                film:Film):
      self.camera_to_world = cam_to_world
      self.fov = fov
      self.film = film

      # <Camera to screen transform>
      self.camera_to_screen:Transform = self.perspective_transform(fov, 2, 1000)

      # <Make ScreenWindow from Film aspect ratio>
      screen = None
      aspect = film.res_x/film.res_y
      if aspect > 1:
         scr_min_y = -1.0
         scr_max_y = 1.0
         scr_min_x = -aspect
         scr_max_x = aspect
         screen = ScreenWindow([scr_min_x,scr_min_y],[scr_max_x,scr_max_y])
      else:
         scr_min_x = -1.0
         scr_max_x = 1.0
         scr_min_y = -aspect
         scr_max_y = aspect
         screen = ScreenWindow([scr_min_x,scr_min_y],[scr_max_x,scr_max_y])
      
      # <Compute raster to screen transform>
      screen_to_ndc = Transform.translate(-screen.p_min[0], -screen.p_max[1],0)
      scale = Transform.scale(1/(screen.p_max[0]-screen.p_min[0]), 1/(screen.p_min[1]-screen.p_max[1]), 1)
      ndc_to_raster = Transform.scale(film.res_x,film.res_y,1)
      self.screen_to_raster = Transform(matrix=np.linalg.matmul(ndc_to_raster.matrix,(np.linalg.matmul(scale.matrix,screen_to_ndc.matrix))))
      self.raster_to_screen = Transform.inverse(self.screen_to_raster)

      # <Compute raster to camera transform>
      self.raster_to_camera = Transform(matrix=np.linalg.matmul(Transform.inverse(self.camera_to_screen).matrix,self.raster_to_screen.matrix))

   def generate_camera_rays(self):
      pix_x = np.indices((1,self.film.res_x))[1]
   