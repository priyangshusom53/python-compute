
import numpy as np
import math

from python_raytracer.core.geometry.transformation import Transform
from python_raytracer.core.math.quaternion import Quaternion

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

class ProjectiveCamera:

   __slots__ = ['film','camera_to_world','camera_to_screen','screen_to_raster', 'raster_to_screen','raster_to_camera']

   def __init__(
         self, 
         film:Film,
         screen:ScreenWindow, 
         cam_to_world:Transform,
         cam_to_screen:Transform):
      
      self.film = film
      self.camera_to_world = cam_to_world
      self.camera_to_screen = cam_to_screen
      # <Compute raster_to_screen Transform>
      screen_to_ndc = Transform.translate(-screen.p_min[0], -screen.p_max[1],0)
      scale = Transform.scale(1/(screen.p_max[0]-screen.p_min[0]), 1/(screen.p_min[1]-screen.p_max[1]), 1)
      ndc_to_raster = Transform.scale(film.res_x,film.res_y,1)
      self.screen_to_raster = Transform(matrix=np.linalg.matmul(ndc_to_raster.matrix,(np.linalg.matmul(scale.matrix,screen_to_ndc.matrix))))
      self.raster_to_screen = Transform.inverse(self.screen_to_raster)
      # <Compute raster_to_camera Transform>
      self.raster_to_camera = Transform(matrix=(Transform.inverse(self.camera_to_screen).matrix @ self.raster_to_screen.matrix))

   @staticmethod
   def raster_to_screen_transform(screen:ScreenWindow,film:Film):
      screen_to_ndc = Transform.translate(-screen.p_min[0], -screen.p_max[1],0)
      scale = Transform.scale(1/(screen.p_max[0]-screen.p_min[0]), 1/(screen.p_min[1]-screen.p_max[1]), 1)
      ndc_to_raster = Transform.scale(film.res_x,film.res_y,1)
      screen_to_raster = Transform(matrix=np.linalg.matmul(ndc_to_raster.matrix,(np.linalg.matmul(scale.matrix,screen_to_ndc.matrix))))
      raster_to_screen = Transform.inverse(screen_to_raster)
      return raster_to_screen

   @staticmethod
   def place_cam(pos:np.ndarray,rot_axis:np.ndarray,angle:float):
      """Returns cam to world transform"""
      rot_axis /= np.linalg.norm(rot_axis)
      q = Quaternion.angle_axis(angle,rot_axis)
      t = Transform.translate(pos[0],pos[1],pos[2])
      r = Quaternion.mat4x4(q)
      cam_to_world = Transform(t.matrix @ r)
      return cam_to_world

class PerspectiveCamera(ProjectiveCamera):

   __slots__ = ['fov']

   def perspective_transform(fov:float, n:float, f:float):
      # <Transfom to perspective view volume>
      persp = Transform(
         matrix=np.array([[1, 0, 0,              0],
                          [0, 1, 0,              0],
                          [0, 0, f/(f-n), -f*n/(f-n)],
                          [0, 0, 1,              0]]))
      # <Scale with fov>
      inv_tan_ang = 1 / math.tan(math.radians(fov) / 2)
      scale = Transform.scale(inv_tan_ang, inv_tan_ang, 1)
      return Transform(matrix=np.linalg.matmul(scale.matrix, persp.matrix))

   def __init__(self, 
                cam_to_world:Transform, 
                fov:float,
                film:Film):

      self.fov = fov
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

      super().__init__(film,screen,cam_to_world,self.perspective_transform(fov,2,1000))

   def generate_camera_rays(self):
      W = self.film.res_x
      H = self.film.res_y
      pix_x = np.indices((W,))[0] + 0.5
      pix_y = np.indices((H,))[0] + 0.5
      pix_grid_x, pix_grid_y = np.meshgrid(pix_x,pix_y)
      pix_grid = np.stack((pix_grid_x,pix_grid_y),axis=-1)
      # <Transform raster coordinates to camera space>
      zeros = np.zeros((H,W,1),dtype=np.float32)
      ones = np.ones((H,W,1),dtype=np.float32)
      pix_grid_cam = np.concatenate((pix_grid,zeros,ones),axis=-1)
      pix_grid_cam = pix_grid_cam @ self.raster_to_camera.matrix.T
      # <Create Rays>
      ray_os = np.zeros(shape=pix_grid_cam.shape,dtype=np.float32)
      ray_os[:,:,3] = 1.0
      ray_ds = pix_grid_cam.astype(np.float32)
      ray_ds[:,:,3] = 0
      ray_ds_xyz = ray_ds[:,:,:3]
      ray_ds_xyz /= np.linalg.norm(ray_ds_xyz,axis=-1,keepdims=True)
      ray_ds[:,:,:3] = ray_ds_xyz
      # <Convert to world space>
      ray_os = ray_os @ self.camera_to_world.matrix.T
      ray_ds = ray_ds @ self.camera_to_world.matrix.T
      rays = np.concatenate((ray_os,ray_ds),axis=2,dtype=np.float32)
      return rays # shape=(H,W,8)
   