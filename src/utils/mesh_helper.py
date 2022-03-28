import numpy as np
import trimesh


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh


def merge_meshes(mesh_list):
    vertices = []
    faces = []
    face_offset = 0
    for mesh in mesh_list:
        num_vertices = len(mesh.vertices)
        vertices.append(mesh.vertices)
        faces.append(mesh.faces + face_offset)
        face_offset += num_vertices
    vertices = np.concatenate(vertices, axis=0)
    faces = np.concatenate(faces, axis=0)
    merged_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return merged_mesh


def scale_to_unit_sphere(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    vertices /= np.max(distances)

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

