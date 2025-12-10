import numpy as np

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
        self.positions = positions
        self.tangents = tangents
        self.normals = normals
        self.uv = uv
        self.alpha_mask = alpha_mask

        print("TriangleMesh initialized with provided data.")

