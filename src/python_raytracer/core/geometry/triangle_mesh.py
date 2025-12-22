import numpy as np
from dataclasses import dataclass

class TriangleMesh:

    def __init__(
        self,
        object_to_world: np.ndarray,
        n_triangles: int,
        vertex_indices: np.ndarray, # A pointer/array of vertex indices (e.g., dtype=int)
        n_vertices: int,
        positions: np.ndarray,             # An array of vertex positions (e.g., dtype=float)
        tangents: np.ndarray = None, # Optional array of tangent vectors
        normals: np.ndarray = None, # Optional array of normal vectors
        uv: np.ndarray = None, # Optional array of parametric (u, v) values
        alpha_mask:np.ndarray = None # Optional alpha mask texture (e.g., image/texture format)
    ):
        """
        Constructor for a TriangleMesh with all required arguments typed.
        Optional arguments default to None.
        """
        self.object_to_world = object_to_world
        self.n_triangles = n_triangles
        self.vertex_indices = vertex_indices
        self.n_vertices = n_vertices
        if (positions.shape[1] == 3):
            ones = np.ones((positions.shape[0], 1), dtype=positions.dtype)
            self.positions = np.column_stack((positions, ones))
        elif (positions.shape[1] == 4):
            self.positions = positions
        self.tangents = tangents
        self.normals = normals
        self.uv = uv
        self.alpha_mask = alpha_mask

        print("TriangleMesh initialized with provided data.")

    def to_np_arrays(self):
        """Convert the TriangleMesh data to numpy arrays for further processing."""
        data = {
            "object_to_world": self.object_to_world,
            "n_triangles": self.n_triangles,
            "vertex_indices": self.vertex_indices,
            "n_vertices": self.n_vertices,
            "positions": self.positions,
            "tangents": self.tangents,
            "normals": self.normals,
            "uv": self.uv,
            "alpha_mask": self.alpha_mask
        }
        return data
    
    def add_tangents(self,tangents:np.ndarray):
        self.tangents = tangents
        print("Tangents added to TriangleMesh.")
    
    def add_uv(self,uv:np.ndarray):
        self.uv = uv
        print("UV coordinates added to TriangleMesh.")

    def add_alpha_mask(self,alpha_mask:np.ndarray):
        self.alpha_mask = alpha_mask
        print("Alpha mask added to TriangleMesh.")




