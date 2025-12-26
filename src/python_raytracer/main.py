
from .loader.gltf_loader import GLTFLoader

from .core.geometry.triangle_mesh import TriangleMesh

def main():
   loader = GLTFLoader()
   path = r"D:\3D Models\sponza_gltf\scene.gltf"
   meshes, materials = loader.load(path)
