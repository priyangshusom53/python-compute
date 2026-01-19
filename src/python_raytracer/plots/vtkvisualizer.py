from typing import Callable

import numpy as np
import vtk
from vtkmodules.util import numpy_support

from python_raytracer.core.geometry.triangle_mesh import(TriangleMesh)
from python_raytracer.core.geometry.transformation import Transform

class Visualizer:

   __slots__ = ['vtk_renderer', 'vtk_interactor','vtk_render_window']

   def __init__(self,meshes:list[TriangleMesh]):
      world_positions3 = [(mesh.positions.array @ mesh.transform.matrix.T)[:,:3] for mesh in meshes]
      all_world_pos3 = np.concatenate(world_positions3,axis=0)
      vtk_points_data = numpy_support.numpy_to_vtk(
            all_world_pos3,
            deep=1,
            array_type=vtk.VTK_FLOAT
      )
      vtk_points = vtk.vtkPoints()
      vtk_points.SetData(vtk_points_data)

      indices = []
      vertex_offset:int = 0
      for mesh in meshes:
         indices.append((mesh.vertex_indices + vertex_offset))
         vertex_offset += mesh.n_vertices
      all_indices = np.concatenate(indices,axis=0)
      cell_counts = np.full((all_indices.shape[0],1),3)
      vtk_cells_data = numpy_support.numpy_to_vtkIdTypeArray(
         np.hstack((cell_counts,all_indices)).astype(np.int64).ravel(),
         deep=1
      )
      vtk_cells = vtk.vtkCellArray()
      vtk_cells.SetCells(mesh.n_triangles,vtk_cells_data)

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