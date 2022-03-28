import numpy as np
import torch


def chamfer_distance_naive(points1, points2):
    ''' Naive implementation of the Chamfer distance.
    Args:
        points1 (numpy array): first point set
        points2 (numpy array): second point set
    '''
    assert(points1.size() == points2.size())
    batch_size, T, _ = points1.size()

    points1 = points1.view(batch_size, T, 1, 3)
    points2 = points2.view(batch_size, 1, T, 3)

    distances = (points1 - points2).pow(2).sum(-1)

    chamfer1 = distances.min(dim=1)[0].mean(dim=1)
    chamfer2 = distances.min(dim=2)[0].mean(dim=1)

    chamfer = chamfer1 + chamfer2
    return chamfer


def compute_iou(occ1, occ2):
    ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.

    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    '''
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = (area_intersect / area_union)

    return iou


def coordinate2index(x, reso, coord_type='2d'):
    ''' Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        x (tensor): coordinate
        reso (int): defined resolution
        coord_type (str): coordinate type
    '''
    x = (x * reso).long()
    if coord_type == '2d':  # plane
        index = x[:, :, 0] + reso * x[:, :, 1]
    elif coord_type == '3d':  # grid
        index = x[:, :, 0] + reso * (x[:, :, 1] + reso * x[:, :, 2])
    index = index[:, None, :]
    return index


def coordinate2index_rectangle(x, res, coord_type='2d'):
    ''' Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        x (tensor): coordinate
        reso (int): defined resolution
        coord_type (str): coordinate type
    '''
    res_x, res_y, res_z = res
    index = x[:, :, 0] * res_y * res_z + x[:, :, 1] * res_z + x[:, :, 2]
    index = index[:, None, :]
    return index


def get_neighbors(points):
    """
    Get the aabb of points
    """
    return torch.stack([
        torch.stack(
            [
                torch.floor(points[:, :, 0]),
                torch.floor(points[:, :, 1]),
                torch.floor(points[:, :, 2])
            ],
            dim=-1
        ),
        torch.stack(
            [
                torch.ceil(points[:, :, 0]),
                torch.floor(points[:, :, 1]),
                torch.floor(points[:, :, 2])
            ],
            dim=-1
        ),
        torch.stack(
            [
                torch.floor(points[:, :, 0]),
                torch.ceil(points[:, :, 1]),
                torch.floor(points[:, :, 2])
            ],
            dim=-1
        ),
        torch.stack(
            [
                torch.floor(points[:, :, 0]),
                torch.floor(points[:, :, 1]),
                torch.ceil(points[:, :, 2])
            ],
            dim=-1
        ),
        torch.stack(
            [
                torch.floor(points[:, :, 0]),
                torch.ceil(points[:, :, 1]),
                torch.ceil(points[:, :, 2])
            ],
            dim=-1
        ),
        torch.stack(
            [
                torch.ceil(points[:, :, 0]),
                torch.floor(points[:, :, 1]),
                torch.ceil(points[:, :, 2])
            ],
            dim=-1
        ),
        torch.stack(
            [
                torch.ceil(points[:, :, 0]),
                torch.ceil(points[:, :, 1]),
                torch.floor(points[:, :, 2])
            ],
            dim=-1
        ),
        torch.stack(
            [
                torch.ceil(points[:, :, 0]),
                torch.ceil(points[:, :, 1]),
                torch.ceil(points[:, :, 2])
            ],
            dim=-1
        ),
    ], dim=0)


def recenter(points, grid_resolution):
    """
    Args:
        points: [B, N, 3] point position in 3D in [-1, 1]
        grid_resolution: the resolution of the feature volume

    Returns:
        local_coordinates [B, 8, N, 3] coordinate wrt neighboring points
        indices: [B, 8, N, 3]: feature grid indices of neighboring points
    """

    # convert points from [-1, 1] to the voxel grid coordinate
    points = (points + 1) / 2 * grid_resolution

    # get neighbouring points
    indices = get_neighbors(points)

    # calculate relative coordinate
    local_coordinates = points.unsqueeze(0).repeat((8, 1, 1, 1)) \
        - indices

    # rescale the coordinates to [-1, 1]
    local_coordinates = local_coordinates  # / grid_resolution * 2

    # [8, B, N, 3] -> [B, 8, N, 3]
    local_coordinates = local_coordinates.permute(1, 0, 2, 3)
    indices = indices.permute(1, 0, 2, 3)

    return local_coordinates, indices.int()


def get_neighbors_new(points, resolution):
    """
    Get the aabb of points
    """
    return torch.stack([
        torch.stack(
            [
                torch.floor(points[:, :, 0] / resolution),
                torch.floor(points[:, :, 1] / resolution),
                torch.floor(points[:, :, 2] / resolution)
            ],
            dim=-1
        ),
        torch.stack(
            [
                torch.ceil(points[:, :, 0] / resolution),
                torch.floor(points[:, :, 1] / resolution),
                torch.floor(points[:, :, 2] / resolution)
            ],
            dim=-1
        ),
        torch.stack(
            [
                torch.floor(points[:, :, 0] / resolution),
                torch.ceil(points[:, :, 1] / resolution),
                torch.floor(points[:, :, 2] / resolution)
            ],
            dim=-1
        ),
        torch.stack(
            [
                torch.floor(points[:, :, 0] / resolution),
                torch.floor(points[:, :, 1] / resolution),
                torch.ceil(points[:, :, 2] / resolution)
            ],
            dim=-1
        ),
        torch.stack(
            [
                torch.floor(points[:, :, 0] / resolution),
                torch.ceil(points[:, :, 1] / resolution),
                torch.ceil(points[:, :, 2] / resolution)
            ],
            dim=-1
        ),
        torch.stack(
            [
                torch.ceil(points[:, :, 0] / resolution),
                torch.floor(points[:, :, 1] / resolution),
                torch.ceil(points[:, :, 2] / resolution)
            ],
            dim=-1
        ),
        torch.stack(
            [
                torch.ceil(points[:, :, 0] / resolution),
                torch.ceil(points[:, :, 1] / resolution),
                torch.floor(points[:, :, 2] / resolution)
            ],
            dim=-1
        ),
        torch.stack(
            [
                torch.ceil(points[:, :, 0] / resolution),
                torch.ceil(points[:, :, 1] / resolution),
                torch.ceil(points[:, :, 2] / resolution)
            ],
            dim=-1
        ),
    ], dim=0)


def recenter_new(points, grid_step_size, resolution):
    """
    Args:
        points: [B, N, 3] point position in 3D in [-1, 1]
        grid_resolution: the resolution of the feature volume

    Returns:
        local_coordinates [B, 8, N, 3] coordinate wrt neighboring points
        indices: [B, 8, N, 3]: feature grid indices of neighboring points
    """

    # convert points from [-1, 1] to the voxel grid coordinate
    points = (points + 1) / 2 * (resolution - grid_step_size)

    # get neighbouring points
    indices = get_neighbors_new(points, grid_step_size)

    neighbor_centers = indices * grid_step_size

    # calculate relative coordinate
    local_coordinates = points.unsqueeze(0).repeat((8, 1, 1, 1)) \
        - neighbor_centers

    # rescale the coordinates to [-1, 1]
    local_coordinates = local_coordinates / grid_step_size

    # [8, B, N, 3] -> [B, 8, N, 3]
    local_coordinates = local_coordinates.permute(1, 0, 2, 3)
    indices = indices.permute(1, 0, 2, 3)

    return local_coordinates, indices.int()


def index_w_border(grid, batch_ind, x, y, z, D, H, W):
    batch_size, feat_dim = grid.shape[:2]
    n_pts = x.shape[1]
    x = torch.clamp(x, 0, D-1)  # [B, N]
    y = torch.clamp(y, 0, H-1)  # [B, N]
    z = torch.clamp(z, 0, W-1)

    out = grid[batch_ind, :, x.flatten(), y.flatten(), z.flatten()].reshape(
        batch_size, n_pts, feat_dim)
    return out


def bilinear_interpolate_torch(im, xy):
    """ grid_sample in pytorch can't have second order derivative

    Argument:
        im: (B, C, H, W)
        xy: (B, N, 2) in [-1, 1]

    Return:
        result: (B, C, N)
    """

    batch_size, feat_dim, img_d, img_h, img_w = im.shape
    n_pts = xy.shape[1]
    assert batch_size == xy.shape[0]

    xy[:, :, 0] = (xy[:, :, 0] + 1) / 2 * (img_w-1)
    xy[:, :, 1] = (xy[:, :, 1] + 1) / 2 * (img_h-1)
    xy[:, :, 2] = (xy[:, :, 2] + 1) / 2 * (img_d-1)

    x = xy[:, :, 2]
    y = xy[:, :, 1]
    z = xy[:, :, 0]
    dtype = x.type()

    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1
    z0 = torch.floor(z).long()
    z1 = z0 + 1

    batch_ind = torch.arange(0, batch_size).unsqueeze(-1).to(im.device)
    batch_ind = batch_ind.repeat(1, n_pts).flatten()

    I000 = index_w_border(im, batch_ind, x0, y0, z0, img_d, img_h, img_w)
    I010 = index_w_border(im, batch_ind, x0, y1, z0, img_d, img_h, img_w)
    I100 = index_w_border(im, batch_ind, x1, y0, z0, img_d, img_h, img_w)
    I110 = index_w_border(im, batch_ind, x1, y1, z0, img_d, img_h, img_w)
    I001 = index_w_border(im, batch_ind, x0, y0, z1, img_d, img_h, img_w)
    I011 = index_w_border(im, batch_ind, x0, y1, z1, img_d, img_h, img_w)
    I101 = index_w_border(im, batch_ind, x1, y0, z1, img_d, img_h, img_w)
    I111 = index_w_border(im, batch_ind, x1, y1, z1, img_d, img_h, img_w)

    x1_weight = (x1.type(dtype)-x)
    y1_weight = (y1.type(dtype)-y)
    z1_weight = (z1.type(dtype)-z)
    x0_weight = (x-x0.type(dtype))
    y0_weight = (y-y0.type(dtype))
    z0_weight = (z-z0.type(dtype))

    w000 = x1_weight * y1_weight * z1_weight
    w000 = w000.unsqueeze(-1)
    w010 = x1_weight * y0_weight * z1_weight
    w010 = w010.unsqueeze(-1)
    w100 = x0_weight * y1_weight * z1_weight
    w100 = w100.unsqueeze(-1)
    w110 = x0_weight * y0_weight * z1_weight
    w110 = w110.unsqueeze(-1)
    w001 = x1_weight * y1_weight * z0_weight
    w001 = w001.unsqueeze(-1)
    w011 = x1_weight * y0_weight * z0_weight
    w011 = w011.unsqueeze(-1)
    w101 = x0_weight * y1_weight * z0_weight
    w101 = w101.unsqueeze(-1)
    w111 = x0_weight * y0_weight * z0_weight
    w111 = w111.unsqueeze(-1)

    return I000*w000 + I010*w010 + I100*w100 + I110*w110 + I001*w001 + I011*w011 + I101*w101 + I111*w111


if __name__ == "__main__":
    # points = np.array([
    #     [
    #         [-0.9999, -0.9999, -0.9999],
    #         [-0.975, -0.99, -0.99],
    #         [-0.8125, -0.8125, -0.8125],
    #         [0.99, 0.99, 0.99]
    #     ]
    # ])
    points = np.stack(np.meshgrid(np.arange(16), np.arange(
        16), np.arange(16)), axis=-1).reshape(-1, 3)
    points = points / 128 * 2 - 1
    recenter_new(torch.from_numpy(points).unsqueeze(0),
                 grid_step_size=8, resolution=128)
