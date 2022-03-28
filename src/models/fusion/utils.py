import numpy as np
from skimage.measure import marching_cubes_lewiner
from tqdm import tqdm
import trimesh
import torch



def decode_feature_grid(
    nerf,
    volume,
    weight_mask,
    num_hits,
    sdf_delta,
    min_coords,
    max_coords,
    volume_resolution,
    voxel_size,
    step_size=0.25,
    batch_size=500,
    level=0.,
    path=None
):
    device = volume.device
    occupied_voxels = torch.nonzero(num_hits[0][0]).cpu().numpy()
    assert step_size <= 1
    all_vertices = []
    all_faces = []
    last_face_id = 0
    min_sdf = []
    max_sdf = []
    for i in tqdm(range(0, len(occupied_voxels), batch_size)):
        origin = occupied_voxels[i:i+batch_size]
        n_batches = len(origin)
        range_ = np.arange(0, 1+step_size, step_size) - 0.5
        spacing = [range_[1] - range_[0]] * 3
        voxel_coords = np.stack(
            np.meshgrid(range_, range_, range_, indexing="ij"),
            axis=-1
        )
        voxel_coords = np.tile(voxel_coords, (n_batches, 1, 1, 1, 1))
        voxel_coords += origin[:, None, None, None, :]
        voxel_coords = torch.from_numpy(
            voxel_coords).float().to(device)
        voxel_pts = voxel_coords * voxel_size + min_coords
        H, W, D = voxel_pts.shape[1:4]
        voxel_pts = voxel_pts.reshape(1, n_batches, -1, 3)
        dirs = torch.zeros_like(voxel_pts)
        pts_and_dirs = torch.cat([voxel_pts, dirs], dim=-1)
        out, _ = nerf(
            pts_and_dirs,
            volume,
            weight_mask,
            sdf_delta,
            voxel_size,
            volume_resolution,
            min_coords,
            max_coords,
            active_voxels=None,
        )
        sdf = out[0, :, :, -1].reshape(n_batches, H, W, D)
        sdf = sdf.detach().cpu().numpy()
        min_sdf.append(np.min(sdf))
        max_sdf.append(np.max(sdf))
        for j in range(n_batches):
            if np.max(sdf[j]) > level and np.min(sdf[j]) < level:
                verts, faces, normals, values = \
                    marching_cubes_lewiner(
                        sdf[j],
                        level=level,
                        spacing=spacing
                    )
                verts += origin[j] - 0.5
                all_vertices.append(verts)
                all_faces.append(faces + last_face_id)
                last_face_id += np.max(faces) + 1
    print(np.min(min_sdf))
    print(np.max(max_sdf))
    
    if len(all_vertices) == 0:
        return None
    final_vertices = np.concatenate(all_vertices, axis=0)
    final_faces = np.concatenate(all_faces, axis=0)
    final_vertices = final_vertices * voxel_size + min_coords.cpu().numpy()
    # all_normals = np.concatenate(all_normals, axis=0)
    mesh = trimesh.Trimesh(
        vertices=final_vertices,
        faces=final_faces,
        # vertex_normals=all_normals,
        process=False
    )
    if path is None:
        return mesh
    else:
        mesh.export(path)


def get_neighbors(points):
    """
    args: voxel_coordinates: [b, n_steps, n_samples, 3]
    """
    return torch.stack([
        torch.stack(
            [
                torch.floor(points[:, :, :, 0]),
                torch.floor(points[:, :, :, 1]),
                torch.floor(points[:, :, :, 2])
            ],
            dim=-1
        ),
        torch.stack(
            [
                torch.ceil(points[:, :, :, 0]),
                torch.floor(points[:, :, :, 1]),
                torch.floor(points[:, :, :, 2])
            ],
            dim=-1
        ),
        torch.stack(
            [
                torch.floor(points[:, :, :, 0]),
                torch.ceil(points[:, :, :, 1]),
                torch.floor(points[:, :, :, 2])
            ],
            dim=-1
        ),
        torch.stack(
            [
                torch.floor(points[:, :, :, 0]),
                torch.floor(points[:, :, :, 1]),
                torch.ceil(points[:, :, :, 2])
            ],
            dim=-1
        ),
        torch.stack(
            [
                torch.ceil(points[:, :, :, 0]),
                torch.ceil(points[:, :, :, 1]),
                torch.floor(points[:, :, :, 2])
            ],
            dim=-1
        ),
        torch.stack(
            [
                torch.ceil(points[:, :, :, 0]),
                torch.floor(points[:, :, :, 1]),
                torch.ceil(points[:, :, :, 2])
            ],
            dim=-1
        ),
        torch.stack(
            [
                torch.floor(points[:, :, :, 0]),
                torch.ceil(points[:, :, :, 1]),
                torch.ceil(points[:, :, :, 2])
            ],
            dim=-1
        ),
        torch.stack(
            [
                torch.ceil(points[:, :, :, 0]),
                torch.ceil(points[:, :, :, 1]),
                torch.ceil(points[:, :, :, 2])
            ],
            dim=-1
        ),
    ], dim=1)
