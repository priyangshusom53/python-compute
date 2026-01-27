from typing import List

import numpy as np
import trimesh

from python_raytracer.triangle_mesh import (TriangleMesh, LEFT_HANDED, RIGHT_HANDED)

def load_gltf_as_triangle_meshes(
    filepath: str,
    target_handedness: int = LEFT_HANDED,
    generate_missing_normals: bool = True
) -> List[TriangleMesh]:
    """
    Load a GLTF (.gltf or .glb) file and convert all meshes to TriangleMesh instances.

    Parameters:
        filepath: path to .gltf or .glb file
        target_handedness: LEFT_HANDED (0) or RIGHT_HANDED (1)
        generate_missing_normals: if True, compute normals when missing

    Returns:
        List of TriangleMesh objects
    """
    # Load scene (preserves hierarchy and transforms)
    scene = trimesh.load(filepath, force='scene')

    meshes: List[TriangleMesh] = []

    # Iterate over geometry (meshes) in the scene
    for mesh_name, mesh in scene.geometry.items():
        if not isinstance(mesh, trimesh.Trimesh):
            continue  # Skip points, lines, etc.

        # Get world transform for this mesh instance(s)
        # In GLTF, a mesh can be instanced multiple times with different transforms
        # We'll find all scene graph nodes that reference this mesh
        mesh_nodes = [node for node in scene.graph.nodes if scene.graph[node][1] == mesh_name]

        if not mesh_nodes:
            # Fallback: use identity if no node found
            transforms = [np.eye(4)]
        else:
            # Get world transforms for each instance
            transforms = []
            for node in mesh_nodes:
                transform = scene.graph.get(node)[0]  # world transform
                transforms.append(transform)

        # For simplicity, we create one TriangleMesh per instance
        # (If you want to merge instances, you'd need to transform vertices and combine)
        for instance_idx, transform in enumerate(transforms):
            # Ensure float32
            vertices = np.array(mesh.vertices, dtype=np.float32)
            faces = np.array(mesh.faces, dtype=np.int32)

            # Apply transform to vertices (if needed later); but store transform separately
            # We store the transform in ObjectToWorldMatrix, and keep positions in object space
            # (This matches your class design)

            # Handle normals
            normals = None
            if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
                normals = np.array(mesh.vertex_normals, dtype=np.float32)
            elif generate_missing_normals:
                # Compute normals from faces
                normals = mesh.vertex_normals  # trimesh computes on access
                normals = np.array(normals, dtype=np.float32)

            # Handle UVs (texture coordinates)
            uvs = None
            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv'):
                uv_data = mesh.visual.uv
                if uv_data is not None and len(uv_data) > 0:
                    # GLTF UVs are (u, v), sometimes with padding
                    uvs = np.array(uv_data[:, :2], dtype=np.float32)  # take first 2 channels

            # Material index: use hash or index; here we use a simple hash of material
            material_index = 0
            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'material'):
                # Simple deterministic hash
                mat_str = str(mesh.visual.material)
                material_index = hash(mat_str) & 0x7FFFFFFF  # positive int

            # Handedness conversion
            current_handedness = RIGHT_HANDED  # GLTF is right-handed
            final_handedness = target_handedness

            # If converting from right to left, we may need to flip triangle winding
            if current_handedness != final_handedness:
                # Flip face winding: reverse order of indices
                faces = np.flip(faces, axis=1).copy()
                vertices[:, 2] *= -1
                # Also flip normals if they exist
                if normals is not None:
                    normals[:, 2] *= -1  # assuming Z is flipped? 
                    # Better: recompute after vertex transform, but for now, note:
                    # Full handedness change often requires coordinate system flip (e.g., Z -> -Z)
                    # But since we're storing transform separately, we handle it via matrix

            # Create identity or provided transform
            obj_to_world = np.array(transform, dtype=np.float32)
            if current_handedness != final_handedness:
                diag = np.diag([1,1,-1,1])
                obj_to_world = diag @ obj_to_world @ diag

            # Create TriangleMesh
            tri_mesh = TriangleMesh(
                nTriangles=len(faces),
                indices=faces,
                nVertices=len(vertices),
                positions=vertices,
                ObjectToWorldMatrix=obj_to_world,
                normals=normals,
                generateNormals=False,
                uvs=uvs,
                triangleBounds=None,  # You can compute if needed
                materialIndex=material_index,
                handedness=final_handedness
            )

            meshes.append(tri_mesh)

    return meshes