
import open3d as o3d
import numpy as np

from python_raytracer.c_functions import _aabb

def plot_mesh_data(
      vertices:np.ndarray,
      indices:np.ndarray,
      normals:np.ndarray=None,
      uvs:np.ndarray=None,
      world_bounds=None,
   ):
   try:

      mesh = o3d.geometry.TriangleMesh()
      mesh.vertices = o3d.utility.Vector3dVector(vertices[:, :3])
      mesh.triangles = o3d.utility.Vector3iVector(indices.reshape(-1, 3))

      # add normals
      mesh.compute_vertex_normals()
      # add uvs

      line_set = o3d.geometry.LineSet()
      # display bounds
      if world_bounds is not None:
         points, lines, colors = _aabb.build_aabb_wireframe_numpy(world_bounds)
         line_set.points = o3d.utility.Vector3dVector(points)
         line_set.lines  = o3d.utility.Vector2iVector(lines)
         line_set.colors = o3d.utility.Vector3dVector(colors)

      o3d.visualization.draw_geometries([mesh,line_set])
   except Exception as e:
      print("Error in plot_mesh_data:", e)