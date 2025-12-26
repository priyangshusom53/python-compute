import numpy as np

from .vector import (Vec4Buffer,Vec2Buffer,Vec3Buffer)
from .transformation import Transform

class TriangleMesh:

    def __init__(
        self,
        transform:np.ndarray,
        n_triangles: int,
        vertex_indices: np.ndarray,
        n_vertices: int, # n_vertices = 3 * n_triangles
        positions: np.ndarray,
        bounds: np.ndarray = None, # shape(n_triangles, 2, 3)
        tangents: np.ndarray = None, 
        normals: np.ndarray = None,
        uv: np.ndarray = None, 
    ):
        """
        Constructor for a TriangleMesh with all required arguments typed.
        Array elements are type=np.float32
        Optional arguments default to None.
        """
        
        self.transform = Transform(transform)
        self.n_triangles = n_triangles
        self.vertex_indices = vertex_indices
        self.n_vertices = n_vertices
        if (positions.shape[1] == 3):

            ones = np.ones((positions.shape[0], 1), dtype=positions.dtype)
            positions = np.column_stack((positions, ones))
            self.positions = Vec4Buffer(size=positions.shape[0], data=positions)

        elif (positions.shape[1] == 4):

            self.positions = Vec4Buffer(size=positions.shape[0], data=positions)

        if tangents is not None:
            if (tangents.shape[1] == 3):

                ones = np.zeros((tangents.shape[0], 1), dtype=tangents.dtype)
                tangents = np.column_stack((tangents, ones))
                self.tangents = Vec4Buffer(size=tangents.shape[0], data=tangents)
            
            elif (tangents.shape[1] == 4):
                self.tangents = Vec4Buffer(size=tangents.shape[0], data=tangents)

        if normals is not None:
            if (normals.shape[1] == 3):

                ones = np.zeros((normals.shape[0], 1), dtype=normals.dtype)
                normals = np.column_stack((normals, ones))
                self.normals = Vec4Buffer(size=normals.shape[0], data=normals)
            
            elif (normals.shape[1] == 4):
                self.normals = Vec4Buffer(size=normals.shape[0], data=normals)

        if uv is not None:
            self.uv = Vec2Buffer(size=uv.shape[0], data=uv)


        print("TriangleMesh initialized with provided data.")

    def set_positions(self, positions:np.ndarray):
        if (positions.shape[1] == 3):

            ones = np.ones((positions.shape[0], 1), dtype=positions.dtype)
            positions = np.column_stack((positions, ones))
            self.positions = Vec4Buffer(size=positions.shape[0], data=positions)

        elif (positions.shape[1] == 4):

            self.positions = Vec4Buffer(size=positions.shape[0], data=positions)
    
    def set_normals(self, normals:np.ndarray):
        if (normals.shape[1] == 3):

            ones = np.zeros((normals.shape[0], 1), dtype=normals.dtype)
            normals = np.column_stack((normals, ones))
            self.normals = Vec4Buffer(size=normals.shape[0], data=normals)
        
        elif (normals.shape[1] == 4):
            self.normals = Vec4Buffer(size=normals.shape[0], data=normals)

    def set_tangents(self, tangents:np.ndarray):
        if (tangents.shape[1] == 3):

            ones = np.zeros((tangents.shape[0], 1), dtype=tangents.dtype)
            tangents = np.column_stack((tangents, ones))
            self.tangents = Vec4Buffer(size=tangents.shape[0], data=tangents)
        
        elif (tangents.shape[1] == 4):
            self.tangents = Vec4Buffer(size=tangents.shape[0], data=tangents)

    def set_uvs(self, uv:np.ndarray):
        self.uv = Vec2Buffer(size=uv.shape[0], data=uv)

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





