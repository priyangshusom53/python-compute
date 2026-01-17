import vtk

def minimal_vtk_scene():
   # 1. Create a cube (data)
   cube = vtk.vtkCubeSource()
   cube.SetXLength(1)
   cube.SetYLength(1)
   cube.SetZLength(1)
   cube.Update()

   # 2. Mapper (data → GPU)
   mapper = vtk.vtkPolyDataMapper()
   mapper.SetInputConnection(cube.GetOutputPort())

   # 3. Actor (transform + appearance)
   actor = vtk.vtkActor()
   actor.SetMapper(mapper)

   # 4. Renderer
   renderer = vtk.vtkRenderer()
   renderer.AddActor(actor)
   renderer.SetBackground(0.1, 0.1, 0.2)

   # 5. Render window
   window = vtk.vtkRenderWindow()
   window.AddRenderer(renderer)
   window.SetSize(1920, 1080)

   # 6. Interactor (mouse controls)
   interactor = vtk.vtkRenderWindowInteractor()
   interactor.SetRenderWindow(window)

   # IMPORTANT: interaction style
   style = vtk.vtkInteractorStyleTrackballCamera()
   interactor.SetInteractorStyle(style)

   # 7. Reset camera so cube is visible
   renderer.ResetCamera()

   # 8. Start
   window.Render()
   interactor.Initialize()
   interactor.Start()

if __name__ == "__main__":
    minimal_vtk_scene()
