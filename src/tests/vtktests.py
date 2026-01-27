import numpy as np
import vtk
from vtkmodules.util import numpy_support

from python_raytracer.gltf_loader import GLTFLoader
from python_raytracer.transformation import Transform

lh_to_rh_t = Transform.scale(1,1,-1)

def visualize():
    loader = GLTFLoader()
    meshes, materials = loader.load(r"D:\3D Models\sponza_gltf\scene.gltf")

    all_wrld_verts = []
    all_indices = []

    vertex_offset = 0

    for mesh in meshes:
        # --- world-space vertices ---
        verts_w = (mesh.positions.array @ mesh.transform.matrix.T)[:, :3]
        all_wrld_verts.append(verts_w)

        # --- offset indices ---
        all_indices.append(mesh.vertex_indices + vertex_offset)

        vertex_offset += verts_w.shape[0]

    all_wrld_verts = np.concatenate(all_wrld_verts, axis=0)
    indices = np.concatenate(all_indices, axis=0)

    # --- VTK points ---
    vtk_points_data = numpy_support.numpy_to_vtk(
        all_wrld_verts.astype(np.float32),
        deep=True,
        array_type=vtk.VTK_FLOAT
    )
    positions = vtk.vtkPoints()
    positions.SetData(vtk_points_data)

    # --- VTK triangles ---
    num_tris = indices.shape[0]
    vtk_cells = np.hstack([
        np.full((num_tris, 1), 3, dtype=np.int64),
        indices.astype(np.int64)
    ]).ravel()

    vtk_tri_cells = numpy_support.numpy_to_vtkIdTypeArray(
        vtk_cells,
        deep=True,
    )

    triangles = vtk.vtkCellArray()
    triangles.SetCells(num_tris, vtk_tri_cells)

    # --- PolyData ---
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(positions)
    polydata.SetPolys(triangles)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.2, 0.4)

    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(800, 600)
    render_window.AddRenderer(renderer)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    renderer.ResetCamera()

    render_window.Render()
    interactor.Initialize()
    interactor.Start()

from python_raytracer.plots.vtkvisualizer import Visualizer
if __name__ == "__main__":
    loader = GLTFLoader()
    meshes, materials = loader.load(r"D:\3D Models\sponza_gltf\scene.gltf")
    vs = Visualizer(meshes)
    print(vs.get_camera_transform().matrix)
