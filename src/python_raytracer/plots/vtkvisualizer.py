from typing import Callable

import numpy as np
import vtk
from vtkmodules.util import numpy_support

from python_raytracer.triangle_mesh import(TriangleMesh, TriangleMesh4)
from python_raytracer.transformation import Transform

class Visualizer:

   __slots__ = ['vtk_renderer', 'vtk_interactor','vtk_render_window']

   def __init__(self,meshes:list[TriangleMesh4]):
      world_positions3 = [(mesh.positions @ mesh.ObjectToWorld.T)[:,:3] for mesh in meshes]
      all_world_pos3 = np.concatenate(world_positions3,axis=0)
      vtk_points_data = numpy_support.numpy_to_vtk(
            all_world_pos3,
            deep=1,
            array_type=vtk.VTK_FLOAT
      )
      del all_world_pos3
      vtk_points = vtk.vtkPoints()
      vtk_points.SetData(vtk_points_data)

      indices = []
      vertex_offset:int = 0
      for mesh in meshes:
         indices.append((mesh.indices + vertex_offset))
         vertex_offset += mesh.nVertices
      all_indices = np.concatenate(indices,axis=0)
      cell_counts = np.full((all_indices.shape[0],1),3)
      vtk_cells_data = numpy_support.numpy_to_vtkIdTypeArray(
         np.hstack((cell_counts,all_indices)).astype(np.int64).ravel(),
         deep=1
      )
      del all_indices
      del cell_counts
      vtk_cells = vtk.vtkCellArray()
      vtk_cells.SetCells(mesh.nTriangles,vtk_cells_data)

      vtk_polydata = vtk.vtkPolyData()
      vtk_polydata.SetPoints(vtk_points)
      vtk_polydata.SetPolys(vtk_cells)
      
      vtk_mapper = vtk.vtkPolyDataMapper()
      vtk_mapper.SetInputData(vtk_polydata)

      vtk_actor = vtk.vtkActor()
      vtk_actor.SetMapper(vtk_mapper)

      vtk_renderer = vtk.vtkRenderer()
      vtk_renderer.AddActor(vtk_actor)
      vtk_renderer.SetBackground(0.1, 0.2, 0.4)
      self.vtk_renderer = vtk_renderer

      render_window = vtk.vtkRenderWindow()
      render_window.SetSize(800, 600)
      render_window.AddRenderer(vtk_renderer)
      self.vtk_render_window = render_window

      interactor = vtk.vtkRenderWindowInteractor()
      interactor.SetRenderWindow(render_window)
      interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
      self.vtk_interactor = interactor

   def start(self):
      #self.vtk_renderer.ResetCamera()

      self.vtk_render_window.Render()
      self.vtk_interactor.Initialize()
      self.vtk_interactor.Start()

   def add_actors(self,actors:list[vtk.vtkActor]):
      for actor in actors:
         self.vtk_renderer.AddActor(actor)

   def get_scene_cam(self):
      return self.vtk_renderer.GetActiveCamera()
   
   def follow_scene_cam(self,actor:vtk.vtkActor):
      camera = self.vtk_renderer.GetActiveCamera()  
      view_matrix = camera.GetViewTransformMatrix()
      world_matrix = vtk.vtkMatrix4x4()
      world_matrix.DeepCopy(view_matrix)
      world_matrix.Invert()
      actor.SetUserMatrix(world_matrix)

   def get_camera_transform(self):
      rh_to_lh = Transform.scale(1,1,-1)
      vtk_camera = self.vtk_renderer.GetActiveCamera()
      vtk_view_mat = vtk.vtkMatrix4x4()
      vtk_view_mat.DeepCopy(vtk_camera.GetViewTransformMatrix())
      world_to_cam = np.zeros((4,4),dtype=np.float32)
      for i in range(4):
         for j in range(4):
            world_to_cam[i,j] = vtk_view_mat.GetElement(i,j)
      cam_to_world = np.linalg.inv(world_to_cam)
      return Transform(rh_to_lh.matrix @ (cam_to_world @ rh_to_lh.matrix))
   
   def add_key_callback(self,key:str,callback:Callable):
      def on_key_press(obj,event):
         _key = obj.GetKeySym()
         if _key == key:
            callback()
      self.vtk_interactor.AddObserver("KeyPressEvent", on_key_press)
   
   def add_render_event_callback(self,callback:Callable):
      self.vtk_renderer.AddObserver("RenderEvent", callback)