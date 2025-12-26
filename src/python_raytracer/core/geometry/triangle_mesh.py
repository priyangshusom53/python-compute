import numpy as np

from .vector import (Vec4Buffer,Vec2Buffer,Vec3Buffer)

class TriangleMesh:

    def __init__(
        self,
        n_triangles: int,
        vertex_indices: np.ndarray,
        n_vertices: int,
        positions: Vec4Buffer,
        tangents: Vec4Buffer = None, 
        normals: Vec4Buffer = None,
        uv: Vec2Buffer = None, 
    ):
        """
        Constructor for a TriangleMesh with all required arguments typed.
        Optional arguments default to None.
        """
        
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

        print("TriangleMesh initialized with provided data.")

    def to_np_arrays(self):
        """Convert the TriangleMesh data to numpy arrays for further processing."""
        data = {
            "n_triangles": self.n_triangles,
            "vertex_indices": self.vertex_indices,
            "n_vertices": self.n_vertices,
            "positions": self.positions,
            "tangents": self.tangents,
            "normals": self.normals,
            "uv": self.uv,
        }
        return data
    
    def add_tangents(self,tangents:np.ndarray):
        self.tangents = tangents
        print("Tangents added to TriangleMesh.")
    
    def add_uv(self,uv:np.ndarray):
        self.uv = uv
        print("UV coordinates added to TriangleMesh.")





