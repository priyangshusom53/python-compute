import numpy as np

LEFT_HANDED:int = 0
RIGHT_HANDED:int = 1

class TriangleMesh:

    __slots__ = [
        'nTriangles',
        'indices',
        'nVertices',
        'positions',
        'ObjectToWorldMatrix',
        'hasNormals',
        'normals',
        'generateNormals',
        'hasTexCoords',
        'uvs',
        'triangleBounds',
        'materialIndex',
        'handedness'
    ]

    def __init__(
        self,
        nTriangles:int,
        indices:np.ndarray,               # np.array dtype=np.int32 shape=(nTriangles,3)
        nVertices:int,
        positions:np.ndarray,             # np.array dtype=np.float32 shape=(nVertices,3) 
        ObjectToWorldMatrix:np.ndarray,   # np.array dtype=np.float32 shape=(4,4)
        normals:np.ndarray = None,        # np.array dtype=np.float32 shape=(nVertices,3)
        generateNormals:bool = False,
        uvs:np.ndarray = None,            # np.array dtype=np.float32 shape=(nVertices,2)
        triangleBounds:np.ndarray = None, # np.array dtype=np.float32 shape=(nTriangles,2,3)
        materialIndex:int = 0,            #
        handedness:int = 0                #
    ):
        self.nTriangles = nTriangles
        self.indices = np.ascontiguousarray(indices,dtype=np.int32)
        self.nVertices = nVertices
        self.positions = np.ascontiguousarray(positions, dtype=np.float32)
        self.ObjectToWorldMatrix = np.ascontiguousarray(ObjectToWorldMatrix, dtype=np.float32)
        if normals is not None:
            self.normals = np.ascontiguousarray(normals, dtype=np.float32)
            self.hasNormals = True
        else:
            self.hasNormals = False
        self.generateNormals = generateNormals
        if uvs is not None:
            self.uvs = np.ascontiguousarray(uvs, dtype=np.float32)
            self.hasTexCoords = True
        else:
            self.hasTexCoords = False
        if triangleBounds is not None:
            self.triangleBounds = np.ascontiguousarray(triangleBounds, dtype=np.float32)
        else:
            self.triangleBounds = None
        self.materialIndex = materialIndex
        self.handedness = handedness

    def SetPositions(self, positions:np.ndarray):
        self.positions = np.ascontiguousarray(positions, dtype=np.float32)

    def SetNormals(self, normals:np.ndarray):
        self.normals = np.ascontiguousarray(normals, dtype=np.float32)

    def SetUVs(self, uvs:np.ndarray):
        self.uvs = np.ascontiguousarray(uvs, dtype=np.float32)

class TriangleMesh4:
    """
    **TriangleMesh4** is for easy transformation modification,
    DEBUG only
    Contains:
        ObjectToWorld: 4x4 matrix,\n
        nTriangles: int,\n
        indices: int32 array (nTriangles, 3),\n
        nVertices: int,\n
        positions: float32 array (nVertices, 4),\n
        normals: float32 array (nVertices, 4),\n
    """
    __slots__ = [
        'ObjectToWorld',
        'nTriangles',
        'indices',
        'nVertices',
        'positions',
        'normals'
    ]

    def __init__(self,mesh:TriangleMesh):
        self.ObjectToWorld = mesh.ObjectToWorldMatrix
        self.nTriangles = mesh.nTriangles
        self.indices = mesh.indices
        self.nVertices = mesh.nVertices
        self.positions = np.ascontiguousarray(np.concatenate([mesh.positions,np.ones((self.nVertices,1),dtype=np.float32)],axis=1),dtype=np.float32)
        self.normals = np.ascontiguousarray(np.concatenate([mesh.normals,np.zeros((self.nVertices,1),dtype=np.float32)],axis=1),dtype=np.float32)



