import cv2
import numpy as np
import pickle
import torch
import torch.nn.functional as F

import src.utils.geometry as geometry


# threshold for numerical stability
THRES = 1e-8


def get_next_int(x, dir_):
    """
    threshold to decide whether a ray is parralel to an axis
    """

    x_bar = np.copy(x)
    positive_mask = dir_ >= 0
    negative_mask = dir_ < 0
    assert np.sum(positive_mask + negative_mask) == len(x)
    x_bar[positive_mask] = np.floor(x[positive_mask]+1)
    x_bar[negative_mask] = np.ceil(x[negative_mask]-1)

    return x_bar


def position_to_coords(
    pts,
    min_coords,
    max_coords,
    volume_resolution
):
    return (pts - min_coords) /\
         (max_coords - min_coords) * (volume_resolution - 1)

def position_to_coords_new(
    pts,
    min_coords,
    voxel_size
):
    return (pts - min_coords) / voxel_size


def coords_to_positions_new(
    coords, min_coords, voxel_size
):
    return coords * voxel_size + min_coords


def coords_to_positions(
    coords,
    min_coords,
    max_coords,
    volume_resolution
):
    return coords / (volume_resolution-1) *\
        (max_coords-min_coords) + min_coords


def flatten(voxels, volume_resolution):
    out = voxels[..., 0] * volume_resolution[1] * volume_resolution[2] + \
        voxels[..., 1] * volume_resolution[2] + voxels[..., 2]
    return out


def unflatten(flat_id, volume_resolution):
    x = torch.div(flat_id, (volume_resolution[1] * volume_resolution[2]), rounding_mode="floor")
    # x = flat_id // (volume_resolution[1] * volume_resolution[2])
    rest = flat_id % (volume_resolution[1] * volume_resolution[2])
    y = torch.div(rest, volume_resolution[2], rounding_mode="floor")
    # y = rest // volume_resolution[2]

    rest = rest % volume_resolution[2]
    z = flat_id - x * volume_resolution[1] * volume_resolution[2] - y * volume_resolution[2]
    if isinstance(z, torch.Tensor):
        return torch.stack([x, y, z], axis=-1)
    else:
        return np.stack([x, y, z], axis=-1)


def get_world_range(dimensions, voxel_size):
    min_ = -dimensions / 2 - voxel_size
    max_ = dimensions / 2 + voxel_size
    n_xyz = np.ceil((max_ - min_) / voxel_size).astype(int).tolist()
    max_ = min_ + voxel_size * np.asarray(n_xyz)
    return min_, max_, n_xyz


def get_world_range_minmax(min_, max_, voxel_size):
    volume_resolution = (max_ - min_) / voxel_size
    volume_resolution = np.ceil(volume_resolution)
    return volume_resolution


def get_frustrum_range(intr_mat, img_h, img_w, max_depth, voxel_size):
    depth = np.zeros((img_h, img_w)) + max_depth
    pts = geometry.depth2xyz(depth, intr_mat).reshape(-1, 3)
    min_ = np.min(pts, axis=0)
    max_ = np.max(pts, axis=0)
    min_[2] = 0
    max_[2] = max_depth
    volume_resolution = (max_ - min_) / voxel_size
    volume_resolution = np.ceil(volume_resolution)
    return min_, max_, volume_resolution


def voxel_traversal(x0, directions, end_coords, volume_resolution):
    # add the origin voxel to traversed_list
    traversed_list = []
    traversed_dist = [np.zeros_like(x0[:, 0])]
    unmask_traversed_dist = [np.zeros_like(x0[:, 0])]

    curr_voxel_id = np.floor(x0)
    traversed_list.append(np.copy(curr_voxel_id))
    curr_coords = x0
    end_voxel_id = np.floor(end_coords)
    _safe_directions = np.where(
        np.abs(directions) < THRES,
        directions + THRES,
        directions
    )
    minimum_dists = ((end_coords - x0) / _safe_directions)[:,0]
    end_voxel_id = np.floor(end_coords)

    pts_inds = np.arange(len(x0))
    # START ITERATING:
    curr_dist = np.sqrt(np.sum(
        (curr_coords - end_coords) ** 2,
        axis=-1
    ))
    n_steps = 0
    while True:
        # 0. check if a point has arrived the end voxel
        # update_mask = np.sum(np.abs(end_voxel_id - curr_voxel_id), axis=-1) > THRES
        update_mask = curr_dist > 1e-2
        
        # 1. compute the next border on x, y, and z direction
        x_borders = get_next_int(curr_coords[:, 0], directions[:, 0])
        y_borders = get_next_int(curr_coords[:, 1], directions[:, 1])
        z_borders = get_next_int(curr_coords[:, 2], directions[:, 2])
        borders = np.stack([x_borders, y_borders, z_borders], axis=-1)

        # 2. compute the distance to the borders
        dist_to_border = (borders - curr_coords) / _safe_directions
        dist_to_end_coords = np.abs(curr_coords - end_coords)
        mask = (dist_to_end_coords[:, 0] <= 1) *\
            (dist_to_end_coords[:, 1] <= 1) *\
            (dist_to_end_coords[:, 2] <= 1)
        dist = np.where(
            mask[:, None],
            dist_to_end_coords,
            dist_to_border
        )
        assert np.min(dist) >= 0

        # 3. move along the direction with minimum distance and register the voxel
        # to traversed list
        mv_dir = np.where(
            mask,
            np.argmax(dist, axis=1),
            np.argmin(dist, axis=1)
        )
        mv_dist = dist[pts_inds, mv_dir]

        # 4. move the points using the distance
        unmask_traversed_dist.append(mv_dist * 1.)
        mv_dist[~update_mask] = 0
        new_coords = curr_coords + mv_dist[:, None] * _safe_directions
        mask = np.isclose(
            new_coords, np.round(new_coords),
            rtol=1e-8, atol=1e-10
        )
        new_voxel_id = np.where(
            mask,
            np.round(new_coords),
            np.floor(new_coords)
        )
        new_voxel_id = np.where(
            np.logical_and(x0 > end_coords, mask),
            new_voxel_id - 1,
            new_voxel_id
        )
        traversed_list.append(np.copy(new_voxel_id))
        traversed_dist.append(mv_dist)
        # some sanity checks for debugging
        new_dist = np.sqrt(np.sum(
            (new_coords - end_coords) ** 2,
            axis=-1
        ))
        # new_dist = np.sum(np.abs(end_voxel_id - new_voxel_id), axis=-1)
        n_steps += 1
        if n_steps > 500:
            import pdb
            pdb.set_trace()
        if np.any(new_dist > curr_dist):
            import pdb
            pdb.set_trace()

        curr_dist = new_dist
        curr_coords = new_coords
        curr_voxel_id = new_voxel_id

        # 6. return the traversed_list when all points are finished
        if np.sum(update_mask) < THRES:
            traversed_dist = np.asarray(traversed_dist)
            unmask_traversed_dist = np.asarray(unmask_traversed_dist)
            accumulated_dist = np.sum(traversed_dist, axis=0)
            end_pts = x0 + accumulated_dist[:, None] * directions
            if not np.all(
                np.sqrt(np.sum((end_pts - end_coords) ** 2, axis=-1)) < np.sqrt(2)
            ):
                import pdb
                pdb.set_trace()
            if np.max(curr_dist) > 1e-2:
                import pdb
                pdb.set_trace()
            if np.any(accumulated_dist + 2 < minimum_dists):
                import pdb
                pdb.set_trace()
            return np.asarray(traversed_list), \
                traversed_dist, unmask_traversed_dist


def get_active_voxels(
    traversed_voxels, traversed_dists, unmask_dists,
    flat_active_coords, volume_resolution
):
    accumulated_dists = np.cumsum(traversed_dists, axis=0)
    unmask_accumulated_dists = np.cumsum(unmask_dists, axis=0)
    last_ids = np.argmax(accumulated_dists, axis=0)
    traversed_flatten_coords = flatten(traversed_voxels, volume_resolution)

    active_start_dists = []
    active_end_dists = []
    active_voxels = []
    n_pts = traversed_voxels.shape[1]
    for i in range(n_pts):
        last_id_per_pt = last_ids[i]
        flatten_coords_per_pt = traversed_flatten_coords[:last_id_per_pt, i]
        traversed_active_coords, ind1, _ = np.intersect1d(
            flatten_coords_per_pt,
            flat_active_coords,
            assume_unique=False,
            return_indices=True
        )
        active_start_dists.append(accumulated_dists[ind1, i])
        active_end_dists.append(unmask_accumulated_dists[ind1+1, i])
        active_voxels.append(traversed_voxels[ind1, i, :])
    return active_voxels, active_start_dists, active_end_dists


def get_active_voxels_from_coords(coords):
    voxel_id = np.floor(coords)
    x = voxel_id[:, 0] * 1.
    y = voxel_id[:, 1] * 1.
    z = voxel_id[:, 2] * 1.
    minus_x = x - 1
    minus_y = y - 1
    minus_z = z - 1
    plus_x = x + 1
    plus_y = y + 1
    plus_z = z + 1
    neighbor_ids = np.concatenate(
        [
            np.stack([minus_x, minus_y, minus_z], axis=-1),
            np.stack([minus_x, y, minus_z], axis=-1),
            np.stack([minus_x, plus_y, minus_z], axis=-1),
            np.stack([x, minus_y, minus_z], axis=-1),
            np.stack([x, y, minus_z], axis=-1),
            np.stack([x, plus_y, minus_z], axis=-1),
            np.stack([plus_x, minus_y, minus_z], axis=-1),
            np.stack([plus_x, y, minus_z], axis=-1),
            np.stack([plus_x, plus_y, minus_z], axis=-1),

            np.stack([minus_x, minus_y, z], axis=-1),
            np.stack([minus_x, y, z], axis=-1),
            np.stack([minus_x, plus_y, z], axis=-1),
            np.stack([x, minus_y, z], axis=-1),
            np.stack([x, y, z], axis=-1),
            np.stack([x, plus_y, z], axis=-1),
            np.stack([plus_x, minus_y, z], axis=-1),
            np.stack([plus_x, y, z], axis=-1),
            np.stack([plus_x, plus_y, z], axis=-1),

            np.stack([minus_x, minus_y, plus_z], axis=-1),
            np.stack([minus_x, y, plus_z], axis=-1),
            np.stack([minus_x, plus_y, plus_z], axis=-1),
            np.stack([x, minus_y, plus_z], axis=-1),
            np.stack([x, y, plus_z], axis=-1),
            np.stack([x, plus_y, plus_z], axis=-1),
            np.stack([plus_x, minus_y, plus_z], axis=-1),
            np.stack([plus_x, y, plus_z], axis=-1),
            np.stack([plus_x, plus_y, plus_z], axis=-1),
        ],
        axis=0
    )
    return neighbor_ids


def grid_transform(
    source_grid,
    T_src_tgt,
    src_volume_resolution,
    tgt_volume_resolution,
    tgt_min_coords,
    tgt_max_coords,
    src_min_coords,
    src_max_coords,
):

    """
    source_grid: [B, C, H_src, W_src, D_src]
    T_src_tgt: [B, 4, 4]
    source_volume_resolution: [H_src, W_src, D_src]
    target_volume_resolution: [H_tgt, W_tgt, D_tgt]
    target_min_coords: [3]
    target_max_coords: [3]
    """

    src_volume_resolution = src_volume_resolution.unsqueeze(0).unsqueeze(0)
    tgt_volume_resolution = tgt_volume_resolution.unsqueeze(0).unsqueeze(0)
    tgt_min_coords = tgt_min_coords.unsqueeze(0).unsqueeze(0)
    tgt_max_coords = tgt_max_coords.unsqueeze(0).unsqueeze(0)
    src_min_coords = src_min_coords.unsqueeze(0).unsqueeze(0)
    src_max_coords = src_max_coords.unsqueeze(0).unsqueeze(0)

    batch_size = T_src_tgt.shape[0]
    n_channels = source_grid.shape[1]
    device = source_grid.device

    # [X, Y, Z, 3]
    x = torch.arange(tgt_volume_resolution[0, 0, 0])
    y = torch.arange(tgt_volume_resolution[0, 0, 1])
    z = torch.arange(tgt_volume_resolution[0, 0, 2])
    # [N, 3]
    coords_target = torch.stack(
        torch.meshgrid(x, y, z),
        dim=-1
    ).reshape(1, -1, 3).to(device)
    coords_target = coords_target.repeat(batch_size, 1, 1)

    # coords to points
    positions_target = coords_to_positions(
        coords_target,
        tgt_min_coords,
        tgt_max_coords,
        tgt_volume_resolution
    )

    # transform to source coordinate
    positions_target = geometry.get_homogeneous(positions_target)
    positions_target = positions_target.permute(0, 2, 1).float()
    positions_src = torch.bmm(T_src_tgt, positions_target)
    positions_src = positions_src.permute(0, 2, 1)[:, :, :3]

    # points to coords
    coords_src = position_to_coords(
        positions_src,
        src_min_coords,
        src_max_coords,
        src_volume_resolution
    )

    # normalize to [-1, 1] to grid_sample
    coords_src_normalized = coords_src / src_volume_resolution
    coords_src_normalized = coords_src_normalized * 2 - 1

    # switch order for grid_sample
    coords_src_normalized = coords_src_normalized[:, :, [2, 1, 0]]
    coords_src_normalized = coords_src_normalized.unsqueeze(1).unsqueeze(1)
    # get features
    sampled_feats = F.grid_sample(
        source_grid,
        coords_src_normalized,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True
    )[:, :, 0, 0]  # [B, C, N]
    tgt_volume_resolution = tgt_volume_resolution.int()
    sampled_feats = sampled_feats.permute(0, 2, 1)
    sampled_feats = sampled_feats.reshape(
        batch_size, tgt_volume_resolution[0, 0, 0].item(),
        tgt_volume_resolution[0, 0, 1].item(), tgt_volume_resolution[0, 0, 2].item(),
        n_channels
    )
    return sampled_feats


def depth_to_tsdf_tensor(
    pts,
    depth,
    T_wc,
    intr_mat,
    min_coords,
    max_coords,
    volume_resolution,
    voxel_size,
    device="cpu"
):
    """ a sparse version

    Args:
        depth (torch.Tensor):  [h, w]
        T_wc (torch.Tensor): [4, 4]
        intr_mat (torch.Tensor): [3, 3]
        min_coords (torch.Tensor): [3]
        max_coords (torch.Tensor): [3]
        volume_resolution (torch.Tensor): [3]
        voxel_size (float): 
        device (str, optional): Defaults to "cpu".

    Returns:
        _type_: _description_
    """
    img_h, img_w = depth.shape

    truncated_region = 5
    range_ = torch.arange(0, truncated_region, device=device) - int(truncated_region/2)
    voxel_coords = torch.stack(
        torch.meshgrid(
            range_, range_, range_,
            indexing="ij",
        ),
        dim=-1
    ).float()  # [H, W, D, 3]
    coords = (pts - min_coords) / voxel_size
    voxel_coords = voxel_coords.unsqueeze(0).tile(len(coords), 1, 1, 1, 1)
    voxel_coords += coords[:, None, None, None, :]
    voxel_coords = torch.round(voxel_coords.reshape(-1, 3))
    voxel_coords[:, 0] = torch.clip(voxel_coords[:, 0], min=0, max=volume_resolution[0])
    voxel_coords[:, 1] = torch.clip(voxel_coords[:, 1], min=0, max=volume_resolution[1])
    voxel_coords[:, 2] = torch.clip(voxel_coords[:, 2], min=0, max=volume_resolution[2])

    # get rid of repeated voxel coordinates
    flat_coords = voxel_coords[:, 0] * volume_resolution[1] * volume_resolution[2] + voxel_coords[:, 1] * volume_resolution[2] + voxel_coords[:,2]
    flat_coords = torch.unique(flat_coords)
    voxel_coords = unflatten(flat_coords, volume_resolution)

    # convert coords to point positions in the camera coordinate system
    voxel_pts = coords_to_positions_new(
        voxel_coords, min_coords, voxel_size)
    cam_voxel_pts = (torch.inverse(T_wc) @ geometry.get_homogeneous(voxel_pts.reshape(-1, 3)).T)[:3, :].T

    # project points to image plane to get the observed depth values
    pixels = geometry.projection(cam_voxel_pts, intr_mat, keep_z=True)

    # out-of-frustum points have zero depth
    pixels_grid_sample = pixels[:, :2] * 1.
    pixels_grid_sample[:, 0] = pixels_grid_sample[:, 0] / img_w * 2 - 1
    pixels_grid_sample[:, 1] = pixels_grid_sample[:, 1] / img_h * 2 - 1

    gt_depths = F.grid_sample(
        depth.unsqueeze(0).unsqueeze(0),
        pixels_grid_sample.unsqueeze(0).unsqueeze(0),
        mode="nearest", padding_mode="zeros", align_corners=True
    )  # [B, C, H, W]
    gt_depths = gt_depths[0, :, 0].T  # [N, 1]

    sdf = gt_depths - pixels[:, 2:]

    # one of the conditions for a voxel to be valid is that
    # it has valid depth values or within frustrum
    valid_ids = torch.abs(gt_depths) > 1e-5

    valid_ids *= pixels[:, 2:] > 0
    # the other condition is that it is not behind the depth measurement 
    # (i.e. being visible)
    valid_ids *= sdf > -2*voxel_size

    # invalid points to have large positive SDF.
    sdf[~valid_ids] = voxel_size * 100

    return voxel_coords, sdf.squeeze(1)


def depth_to_tsdf(
    depth,
    T_wc,
    intr_mat,
    min_coords,
    max_coords,
    volume_resolution,
    voxel_size,
    device="cpu"
):
    """_summary_

    Args:
        depth (nump.array):  [h,w] 
        T_wc (torch.Tensor): _description_
        intr_mat (torch.Tensor): _description_
        min_coords (_type_): _description_
        max_coords (_type_): _description_
        volume_resolution (_type_): _description_
        voxel_size (_type_): _description_
        device (str, optional): _description_. Defaults to "cpu".

    Returns:
        _type_: _description_
    """
    img_h, img_w = depth.shape

    # get voxel coordinates in the volume
    voxel_coords = np.stack(
        np.meshgrid(
            np.arange(volume_resolution[0]),
            np.arange(volume_resolution[1]),
            np.arange(volume_resolution[2]),
            indexing="ij"
        ),
        axis=-1
    )  # [H, W, D, 3]
    H, W, D = voxel_coords.shape[:3]

    # convert coords to point positions in the camera coordinate system
    voxel_pts = coords_to_positions_new(
        voxel_coords, min_coords, voxel_size)
    cam_voxel_pts = (np.linalg.inv(T_wc) @ geometry.get_homogeneous(voxel_pts.reshape(-1, 3)).T)[:3, :].T

    # project points to image plane to get the observed depth values
    pixels = geometry.projection(cam_voxel_pts, intr_mat, keep_z=True)

    # out-of-frustum points have zero depth
    pixels_grid_sample = pixels[:, :2] * 1.
    pixels_grid_sample[:, 0] = pixels_grid_sample[:, 0] / img_w * 2 - 1
    pixels_grid_sample[:, 1] = pixels_grid_sample[:, 1] / img_h * 2 - 1

    gt_depths = F.grid_sample(
        torch.from_numpy(depth).to(device).unsqueeze(0).unsqueeze(0).float(),
        torch.from_numpy(pixels_grid_sample).to(device).unsqueeze(0).unsqueeze(0).float(),
        mode="nearest", padding_mode="zeros", align_corners=True
    )  # [B, C, H, W]
    gt_depths = gt_depths[0, :, 0].T.cpu().numpy()  # [N, 1]

    sdf = gt_depths - pixels[:, 2:]
    sdf = np.clip(sdf, a_min=-5 * voxel_size, a_max=5 * voxel_size)
    weights = np.zeros_like(sdf)

    # one of the conditions for a voxel to be valid is that
    # it has valid depth values or within frustrum
    valid_ids = np.abs(gt_depths) > 1e-5

    valid_ids *= pixels[:, 2:] > 0
    # the other condition is that it is not behind the depth measurement 
    # (i.e. being visible)
    valid_ids *= sdf > -2*voxel_size

    weights[valid_ids] = 1.
    sdf[~valid_ids] = 0.
    sdf = sdf.reshape(H, W, D)
    weights = weights.reshape(H, W, D)

    return sdf, weights


def is_active(coords, active_voxels, volume_resolution):
    """ compute whether coordinates belong to active voxels by two criterias:
    1) a coordinate should be within the volume bound
    2) it belongs to the input active voxels
    
    Args:
        coords: [b, n_pts, n_steps, 3]
        active_voxels: [H, W, D]
    Returns:
        active_masks: [b, n_pts, n_steps]
    """

    batch, n_pts, n_steps = coords.shape[:3]
    within_grid = (coords[..., 0] >= 0) * (coords[..., 0] < volume_resolution[0]) *\
        (coords[..., 1] >= 0) * (coords[..., 1] < volume_resolution[1]) *\
        (coords[..., 2] >= 0) * (coords[..., 2] < volume_resolution[2])

    # 1 unit of voxel size for buffer
    active_x = (coords[..., 0] > -1) * (coords[..., 0] < volume_resolution[0] + 1)
    active_y = (coords[..., 1] > -1) * (coords[..., 1] < volume_resolution[1] + 1)
    active_z = (coords[..., 2] > -1) * (coords[..., 2] < volume_resolution[2] + 1)

    capped_x = torch.clip(coords[..., 0] * 1., 0, volume_resolution[0]-1).long().reshape(-1)
    capped_y = torch.clip(coords[..., 1] * 1., 0, volume_resolution[1]-1).long().reshape(-1)
    capped_z = torch.clip(coords[..., 2] * 1., 0, volume_resolution[2]-1).long().reshape(-1)

    active_masks = active_voxels[capped_x, capped_y, capped_z].reshape(batch, n_pts, n_steps)
    active_masks = torch.where(within_grid, active_masks, torch.ones_like(active_masks))
    active_masks = active_masks * active_x * active_y * active_z
    return active_masks
