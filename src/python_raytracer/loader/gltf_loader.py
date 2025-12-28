
import json
def json_loader(path:str):
    try:
        with open(path, 'r') as file:
            data = json.load(file)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None
    return data



# GLTFLoader implementation
from python_raytracer.loader.scene_loader import FLoader
from python_raytracer.core.geometry.triangle_mesh import TriangleMesh
from python_raytracer.core.material.pbr_material import PBRMaterial

# trimesh types
import trimesh 
from trimesh.visual import TextureVisuals
from trimesh.visual.material import (Material,SimpleMaterial,PBRMaterial as TrimeshPBRMaterial)

import numpy as np

import sys
import os
import traceback
from pathlib import Path
from typing import Dict, Any

class GLTFLoader(FLoader):
    def load(self, path: str):
        '''
        convert to TriangleMesh arrays
        '''

        if not os.path.exists(path):
            raise FileNotFoundError(f"GLTF file not found: {path}")

        gltf_json = json_loader(path)
        if gltf_json is None:
            raise ValueError(f"Failed to load GLTF JSON from: {path}")
        
        try:
            # Ask trimesh to try to load a scene
            scene = trimesh.load(str(path), force='scene')
        except Exception:
            # Print traceback to help debugging
            print("trimesh.load() raised an exception:")
            traceback.print_exc()
            raise

        # Scene handling
        world_meshes = []
        meshes:list[TriangleMesh] = []
        pbr_materials:list[PBRMaterial] = []
    
        # Handle single mesh case
        if isinstance(scene, trimesh.Trimesh):
            print("Loaded a single Trimesh object.")
            world_meshes = [scene.copy()]
            meshes = [scene.copy()]
            meshes[0] = TriangleMesh(
                np.eye(4),
                n_triangles=len(scene.faces),
                vertex_indices=scene.faces,
                n_vertices=len(scene.vertices),
                positions=scene.vertices,
                tangents=None,
                normals=scene.vertex_normals,
                uv= None,
                alpha_mask=None
            )                                    
            
            uvs,pbr_material = self.load_material_and_uv_of_mesh(scene)
            meshes[0].uv = uvs
            
            pbr_materials = [pbr_material]

            return meshes, pbr_materials
        elif isinstance(scene, trimesh.Scene):
            for node_name in scene.graph.nodes:
                # transform: local_to_world np.array(4,4)
                transform, geometry_name = scene.graph[node_name]
        
                # Skip nodes without geometry
                if geometry_name is None:
                    continue
            
                geometry = scene.geometry[geometry_name]
        
                # Process only meshes (skip point clouds, paths, etc.)
                if not isinstance(geometry, trimesh.Trimesh):
                    continue
            
                # Create a copy and apply world transform
                mesh_world = geometry.copy()
                mesh_world.apply_transform(transform)
                world_meshes.append(mesh_world)


                mesh = geometry.copy()
                triangles_vertices:np.ndarray = mesh.triangles
                # get triangles bounds
                min_coords = triangles_vertices.min(axis=1)
                max_coords = triangles_vertices.max(axis=1)
                all_triangle_bounds = np.stack([min_coords, max_coords], axis=1)
                print(f"bounds array shape: {all_triangle_bounds.shape}")


                ones = np.ones((*triangles_vertices.shape[:-1], 1), dtype=triangles_vertices.dtype)
                tri_h = np.concatenate([triangles_vertices, ones], axis=-1)

                # convert to world space tranform with 
                # transform.T as transform matrix and 
                # tri_h is row-major
                tri_world_h = tri_h @ transform.T
                triangles_world = tri_world_h[..., :3]
                # get triangles world bounds
                min_coords = triangles_world.min(axis=1)
                max_coords = triangles_world.max(axis=1)
                all_triangle_world_bounds = np.stack([min_coords, max_coords], axis=1)


                mesh = TriangleMesh(
                    transform,
                    n_triangles=len(mesh.faces),
                    vertex_indices=np.asarray(mesh.faces,dtype=np.int32),
                    n_vertices=len(mesh.vertices),
                    positions=mesh.vertices.astype(np.float32),
                    bounds=all_triangle_bounds.astype(np.float32),
                    world_bounds=all_triangle_world_bounds.astype(np.float32),
                    tangents=None,
                    normals=mesh.vertex_normals.astype(np.float32),
                    uv= None,
                )                                    

                # Get uvs, materials of this mesh
                uvs,pbr_material = self.load_material_and_uv_of_mesh(geometry)
                # add uv to the mesh
                mesh.set_uvs(uvs)
                if not any(material.name == pbr_material.name for material in pbr_materials):
                    pbr_materials.append(pbr_material)

                meshes.append(mesh)

    
            return meshes, pbr_materials

    def _extract_material_no_use(self,mesh: trimesh.Trimesh) -> Dict[str, Any]:
        """Extract material properties from a trimesh mesh"""
        material_info = {
            'name': None,
            'base_color': None,           # [R, G, B, A] (0-1)
            'metallic': 0.0,
            'roughness': 1.0,
            'emissive': [0.0, 0.0, 0.0],
            'normal_texture': None,       # PIL Image or None
            'base_color_texture': None,   # PIL Image or None
            'vertex_colors': None,        # numpy array or None
            'type': 'unknown'
        }
    
        # Handle vertex colors first (highest priority)
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            colors = mesh.visual.vertex_colors
            if colors.shape[1] == 4:  # RGBA
                material_info['vertex_colors'] = colors.astype(np.float32) / 255.0
            elif colors.shape[1] == 3:  # RGB
                material_info['vertex_colors'] = np.column_stack([
                    colors.astype(np.float32) / 255.0,
                    np.ones((colors.shape[0], 1), dtype=np.float32)  # Add alpha=1
                ])
            return material_info
        
        # Handle PBR materials (GLTF standard)
        if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
            mat = mesh.visual.material
            material_info['name'] = getattr(mat, 'name', None)
            
            # Base color (diffuse)
            if hasattr(mat, 'baseColorFactor'):
                material_info['base_color'] = np.array(mat.baseColorFactor, dtype=np.float32)
            elif hasattr(mat, 'diffuse'):
                material_info['base_color'] = np.array(mat.diffuse, dtype=np.float32)
            
            # PBR metallic/roughness
            if hasattr(mat, 'metallicFactor'):
                material_info['metallic'] = float(mat.metallicFactor)
            if hasattr(mat, 'roughnessFactor'):
                material_info['roughness'] = float(mat.roughnessFactor)
            
            # Emissive
            if hasattr(mat, 'emissiveFactor'):
                material_info['emissive'] = np.array(mat.emissiveFactor, dtype=np.float32)
            
            # Textures
            if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
                material_info['base_color_texture'] = mat.baseColorTexture
            elif hasattr(mat, 'diffuseTexture') and mat.diffuseTexture is not None:
                material_info['base_color_texture'] = mat.diffuseTexture
                
            if hasattr(mat, 'normalTexture') and mat.normalTexture is not None:
                material_info['normal_texture'] = mat.normalTexture
                
            material_info['type'] = mat.__class__.__name__
        
        # Fallback: simple color material
        elif hasattr(mesh.visual, 'main_color'):
            color = mesh.visual.main_color
            if len(color) == 4:
                material_info['base_color'] = color.astype(np.float32) / 255.0
            else:
                material_info['base_color'] = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        
        return material_info

    @staticmethod
    def trimesh_to_open3d(tmesh: 'trimesh.Trimesh'):
        """Convert a trimesh.Trimesh to an open3d.geometry.TriangleMesh (if open3d is installed)."""
        if o3d is None:
            raise RuntimeError("open3d not installed.")
        verts = np.asarray(tmesh.vertices)
        faces = np.asarray(tmesh.faces, dtype=np.int32)
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
        o3d_mesh.compute_vertex_normals()
        return o3d_mesh

    def check_vertex_attributes(self, gltf_json: any, mesh_name:str):

        has_normals = False
        has_tangents = False
        has_texcoords = False

    def load_material_and_uv_of_mesh(self, mesh: trimesh.Trimesh):
        material = None
        if mesh.visual is not None and isinstance(mesh.visual, TextureVisuals):
            visual: TextureVisuals = mesh.visual
            try:
                uv:np.ndarray = visual.uv
                material = visual.material
                pbr_material:PBRMaterial = None
                if isinstance(material,SimpleMaterial) or isinstance(material,Material):
                    pbr_material = PBRMaterial(name = material.name)
                    return uv, pbr_material
                elif isinstance(material,TrimeshPBRMaterial):
                    pbr_material = PBRMaterial(
                        name = material.name,
                        base_color_factor = material.baseColorFactor if hasattr(material,'baseColorFactor') else [1.0,1.0,1.0,1.0],
                        metallic_factor = material.metallicFactor if hasattr(material,'metallicFactor') else 0.0,
                        roughness_factor = material.roughnessFactor if hasattr(material,'roughnessFactor') else 1.0,
                        base_color_texture= material.baseColorTexture if hasattr(material,'baseColorTexture') else None,
                        normal_texture= material.normalTexture if hasattr(material,'normalTexture') else None,
                        metallic_roughness_texture= material.metallicRoughnessTexture if hasattr(material,'metallicRoughnessTexture') else None,
                    )
                    return uv, pbr_material
            except Exception as e:
                print(f"Error extracting material/uv: {e}")
                raise
        else:
            return None, None


def create_open3d_mesh_from_trimesh(mesh:TriangleMesh):
    """Convert a trimesh.Trimesh to an open3d.geometry.TriangleMesh (if open3d is installed)."""
    if o3d is None:
        raise RuntimeError("open3d not installed.")
    try:
        verts = np.asarray(mesh.positions.array)
        faces = np.asarray(mesh.vertex_indices, dtype=np.int32)
        o3d_mesh = o3d.geometry.TriangleMesh()
        verts3 = verts[:,:3]
        o3d_mesh.vertices = o3d.utility.Vector3dVector(verts3)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
        o3d_mesh.compute_vertex_normals()
    except Exception as e:
        print(f"Error converting TriangleMesh to Open3D mesh: {e}")
        raise
    return o3d_mesh

import open3d as o3d
import copy

if __name__ == "__main__":
    loader = GLTFLoader()
    # update the path below
    example_path = r"D:\3D Models\sponza_gltf\scene.gltf"
    
    meshes,materials = loader.load(example_path)
    #print(meshes[0].positions.shape)
    wrld_meshes:list[TriangleMesh] = []
    
    for mesh in meshes:
        wrld_vertices = []
        vertices = mesh.positions.array

        wrld_mesh = copy.deepcopy(mesh)
        wrld_vertices = vertices @ mesh.transform.matrix.T
        wrld_mesh.positions = np.array(wrld_vertices)
        wrld_mesh.set_positions(wrld_vertices)
        wrld_meshes.append(wrld_mesh)

    open3d_meshes = []    
    for wrld_mesh in wrld_meshes:
        open3d_meshes.append(create_open3d_mesh_from_trimesh(wrld_mesh))
    
    
    print(f"Loaded {len(meshes)} meshes and {len(materials)} materials.")
    o3d.visualization.draw_geometries(open3d_meshes)



