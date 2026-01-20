import numpy as np
import vtk
from vtkmodules.util import numpy_support

# GIZMO_COLOR:list[float] = None
class Gizmo:

   @staticmethod
   def draw_ray(r_os:np.ndarray, r_ds:np.ndarray, t:float=10):
      r_from = r_os.copy().reshape((-1,1,3))
      r_to = (r_os+(r_ds*t)).copy().reshape((-1,1,3))
      line_seg = np.stack([r_from,r_to],axis=1).astype(np.float32)
      return line_seg # shape=(N,2,3)
   
   @staticmethod
   def draw_ray_vtk(r_os:np.ndarray, r_ds:np.ndarray, t:float=30):
      r_from = r_os
      r_to = (r_os + (r_ds*t)).astype(np.float32)
      vtk_points_data = numpy_support.numpy_to_vtk(
         np.concatenate([r_from,r_to],axis=0,dtype=np.float32),
         deep=1,
         array_type=vtk.VTK_FLOAT
      )
      del r_from
      del r_to
      vtk_points = vtk.vtkPoints()
      vtk_points.SetData(vtk_points_data)
      n_lines = r_os.shape[0]
      vtk_cells_data = numpy_support.numpy_to_vtkIdTypeArray(
         np.stack([np.repeat(2,repeats=n_lines).astype(np.int64),np.arange(start=0,stop=n_lines,dtype=np.int64),np.arange(start=n_lines,stop=(2*n_lines),dtype=np.int64)],axis=-1).ravel(),
         deep=1
      )
      vtk_cells = vtk.vtkCellArray()
      vtk_cells.SetCells(n_lines,vtk_cells_data)
      vtk_lines = vtk.vtkPolyData()
      vtk_lines.SetPoints(vtk_points)
      vtk_lines.SetLines(vtk_cells)
      mapper = vtk.vtkPolyDataMapper()
      mapper.SetInputData(vtk_lines)
      actor = vtk.vtkActor()
      actor.SetMapper(mapper)
      GIZMO_COLOR = [0.0,1.0,0.0]
      actor.GetProperty().SetColor(0.0,1.0,0.0)
      actor.GetProperty().SetLineWidth(2.0)
      return actor
   
   @staticmethod
   def draw_aabb():
      pass

   @staticmethod
   def draw_aabb3_vtk(aabbs:np.ndarray,max_count=200): # expected aabbs shape=(N,2,3) 
      boxes=[]
      GIZMO_COLOR = [0.0,1.0,0.0]
      for idx in range(min(aabbs.shape[0],max_count)):
         pmin = aabbs[idx,0]
         pmax = aabbs[idx,1]
         cube = vtk.vtkCubeSource()
         cube.SetBounds(pmin[0],pmax[0],
                        pmin[1],pmax[1],
                        pmin[2],pmax[2])
         cubeMapper = vtk.vtkPolyDataMapper()
         cubeMapper.SetInputConnection(cube.GetOutputPort())
         cubeActor = vtk.vtkActor()
         cubeActor.SetMapper(cubeMapper)
         cubeActor.GetProperty().SetRepresentationToWireframe()
         cubeActor.GetProperty().SetColor(*GIZMO_COLOR)
         boxes.append(cubeActor)
      return boxes
