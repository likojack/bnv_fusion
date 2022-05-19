import torch
import torch.nn.functional as F
import time

from src.models.fusion.utils import get_neighbors


class Sampler():
    def __init__(self, n_coarse_samples, n_fine_samples, device):
        self.n_coarse_samples = n_coarse_samples
        self.n_fine_samples = n_fine_samples
        self.coarse_interval_dists = torch.linspace(
            0, 1, steps=n_coarse_samples, device=device)
        self.fine_interval_dists = torch.linspace(
            0, 1, steps=n_fine_samples, device=device)

    def stratified_sampling(self, interval_dists, n_pts, n_samples, distances):
        """
        distances: [(b), N, 1]
        """
        if len(distances.shape) < 3:
            distances = distances.unsqueeze(0)
        batch_size, n_pts = distances.shape[:2]
        interval_dists = interval_dists.unsqueeze(0).repeat(batch_size, n_pts, 1) *\
            distances
        mids = .5 * (interval_dists[..., 1:] + interval_dists[..., :-1])
        upper = torch.cat([mids, interval_dists[..., -1:]], dim=-1)
        lower = torch.cat([interval_dists[..., :1], mids], dim=-1)
        # stratified samples in those intervals
        t0 = time.time()
        # t_rand = torch.rand(batch_size, n_pts, n_samples).to(distances.device)
        t_rand = torch.rand(batch_size, n_pts, n_samples, device=interval_dists.device)

        t1 = time.time()
        print("get random: ", t1 - t0)
        interval_dists = lower + (upper - lower) * t_rand
        interval_dists = interval_dists.unsqueeze(-1)
        return interval_dists

    def hierarchical_sampling(
        self, depths, surface, ray_directions,
        cam_loc, offset_distance=0.5, max_depth=5.
    ):
        """
        surface: [B, N, 3]
        depths:  [B, N]
        """
        batch_size, n_pts = ray_directions.shape[:2]
        half_distance = torch.zeros_like(depths) + offset_distance / 10 * 9  # [B, N]
        negative_offset = torch.where(depths - offset_distance/10 * 9 < 0, depths, half_distance)
        start_pts = surface - negative_offset.unsqueeze(-1) * ray_directions
        start_depths = torch.sqrt(torch.sum(
            (start_pts - cam_loc.unsqueeze(1)) ** 2,
            dim=-1
        ))  # [B, N]
        distances = torch.zeros_like(ray_directions[:, :, :1]) + offset_distance
        # distances = torch.zeros((batch_size, n_pts, 1)).float().cuda() + offset_distance
        t0 = time.time()
        interval_dists_fine = self.stratified_sampling(
            self.fine_interval_dists, n_pts, self.n_fine_samples, distances
        )
        t1 = time.time()
        print("stratified_sampe: ", t1 - t0)
        interval_dists_fine += start_depths.unsqueeze(-1).unsqueeze(-1)

        depths = depths.unsqueeze(-1)
        interval_dists_coarse = self.stratified_sampling(
            self.coarse_interval_dists, n_pts, self.n_coarse_samples, depths
        )
        dists, _ = torch.sort(
            torch.cat([interval_dists_fine, interval_dists_coarse], -2), -2)
        pts = cam_loc.unsqueeze(1).unsqueeze(1) + dists \
            * ray_directions.unsqueeze(2)  # [b, N, S, 3]
        return pts, dists


def stratified_sampling(n_pts, n_samples, distances):
    """
    distances: [(b), N, 1]
    """
    if len(distances.shape) < 3:
        distances = distances.unsqueeze(0)
    batch_size, n_pts = distances.shape[:2]
    interval_dists = torch.linspace(0, 1, steps=n_samples, device=distances.device)
    interval_dists = interval_dists.unsqueeze(0).repeat(batch_size, n_pts, 1) *\
        distances
    mids = .5 * (interval_dists[..., 1:] + interval_dists[..., :-1])
    upper = torch.cat([mids, interval_dists[..., -1:]], dim=-1)
    lower = torch.cat([interval_dists[..., :1], mids], dim=-1)
    # stratified samples in those intervals
    t_rand = torch.rand(batch_size, n_pts, n_samples, device=distances.device)
    interval_dists = lower + (upper - lower) * t_rand
    interval_dists = interval_dists.unsqueeze(-1)
    return interval_dists


def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous().to(cdf.device)
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])
    return samples


def dist2pts(dists, cam_loc, ray_directions):
    pts = cam_loc.unsqueeze(1) + dists * ray_directions.unsqueeze(1)
    return pts


def ray_sampling(n_samples, distances, ray_directions, cam_loc):
    """
    distances: [b, n]
    ray_directions: [b, n ,3]
    cam_loc: [b, 3]
    """
    distances = distances.unsqueeze(-1)
    if len(ray_directions.shape) == 3:
        n_pts = ray_directions.shape[1]
    elif len(ray_directions.shape) == 2:
        n_pts = ray_directions.shape[0]
    else:
        raise NotImplementedError()
    interval_dists = stratified_sampling(n_pts, n_samples, distances)

    pts = cam_loc.unsqueeze(1).unsqueeze(1) + interval_dists \
        * ray_directions.unsqueeze(2)  # [b, N, S, 3]
    return pts, interval_dists


def ray_sampling_around_surface(
    n_samples, depths, surface, ray_directions, distance=0.5
):
    """
        n_samples: int
        surface: [B, N, 3]
        cam_loc: [B, 1, 3]
        ray_directions: [B, N, 3]
    """
    # depths = torch.sqrt(torch.sum(
    #     (surface - cam_loc) ** 2,
    #     dim=-1
    # ))  # [B, N]
    half_distance = torch.zeros_like(depths) + distance / 10 * 9  # [B, N]
    negative_offset = torch.where(depths - distance/10 * 9 < 0, depths, half_distance)
    batch_size, n_pts = ray_directions.shape[:2]
    distances = torch.zeros((batch_size, n_pts, 1)).float().cuda() + distance
    interval_dists = stratified_sampling(n_pts, n_samples, distances)
    interval_dists = interval_dists - negative_offset.unsqueeze(-1).unsqueeze(-1)

    pts = surface.unsqueeze(2) + interval_dists * ray_directions.unsqueeze(2)  # [B, N, S, 3]
    return pts, interval_dists


def hierarchical_sampling(
    n_fine_samples, n_coarse_samples, depths, surface, ray_directions,
    cam_loc, offset_distance=0.5, max_depth=5.
):
    """hierarchical sample points on rays

    Args:
        n_fine_samples (_type_): _description_
        n_coarse_samples (_type_): _description_
        depths (_type_): _description_
        surface (_type_): _description_
        ray_directions (_type_): _description_
        cam_loc (_type_): _description_
        offset_distance (float, optional): _description_. Defaults to 0.5.
        max_depth (_type_, optional): _description_. Defaults to 5..

    Returns:
        _type_: _description_
    """
    batch_size, n_pts = ray_directions.shape[:2]
    half_distance = torch.zeros_like(depths) + offset_distance # [B, N]
    negative_offset = torch.where(depths - offset_distance < 0, depths, half_distance)
    start_pts = surface - negative_offset.unsqueeze(-1) * ray_directions
    start_depths = torch.sqrt(torch.sum(
        (start_pts - cam_loc.unsqueeze(1)) ** 2,
        dim=-1
    ))  # [B, N]
    distances = torch.zeros_like(ray_directions[:, :, :1]) + offset_distance * 2
    # distances = torch.zeros((batch_size, n_pts, 1)).float().cuda() + offset_distance

    interval_dists_fine = stratified_sampling(n_pts, n_fine_samples, distances)
    interval_dists_fine += start_depths.unsqueeze(-1).unsqueeze(-1)

    depths = depths.unsqueeze(-1)

    interval_dists_coarse = stratified_sampling(
        n_pts, n_coarse_samples, depths)

    dists, _ = torch.sort(
        torch.cat([interval_dists_fine, interval_dists_coarse], -2), -2)
    pts = cam_loc.unsqueeze(1).unsqueeze(1) + dists \
        * ray_directions.unsqueeze(2)  # [b, N, S, 3]
    return pts, dists


def render_pts_old(pts, occupied_prob):
    if len(pts.shape) == 4:
        n_samples = pts.shape[2]
    else:
        n_samples = pts.shape[1]
    passthrough_prob = torch.cumprod(1 - occupied_prob, dim=-1)
    passthrough_prob = torch.cat(
        [torch.ones_like(passthrough_prob[..., :1]), passthrough_prob],
        dim=-1
    )
    background_prob = passthrough_prob[..., -1]
    depth_prob = passthrough_prob[..., :n_samples] * occupied_prob
    surface_pts = torch.sum(depth_prob.unsqueeze(-1) * pts, dim=-2)

    return surface_pts, depth_prob, background_prob


def render_pts(pts, occupied_prob, ray_dirs, cam_loc, dists):
    if len(pts.shape) == 4:
        n_samples = pts.shape[2]
    else:
        n_samples = pts.shape[1]
    passthrough_prob = torch.cumprod(1 - occupied_prob, dim=-1)
    passthrough_prob = torch.cat(
        [torch.ones_like(passthrough_prob[..., :1]), passthrough_prob],
        dim=-1
    )
    background_prob = passthrough_prob[..., -1]
    depth_prob = passthrough_prob[..., :n_samples] * occupied_prob
    surface_pts = torch.sum(depth_prob.unsqueeze(-1) * pts, dim=-2)

    expected_dist = torch.sum(depth_prob.unsqueeze(-1) * dists, dim=-2)

    pts = cam_loc.unsqueeze(1) + expected_dist * ray_dirs  # [b, N, S, 3]
    return pts, depth_prob, background_prob


def render_pts_sdf(pts, tsdf, ray_dirs, cam_loc):
    n_batches, n_pts, n_samples = pts.shape[:3]
    # [n_batchs, n_pts, 2]
    pad_indices = torch.stack(
        torch.meshgrid(
            torch.arange(n_batches, device=pts.device),
            torch.arange(n_pts, device=pts.device)
        ),
        dim=-1
    )

    tsdf_inds = torch.argmin(torch.abs(tsdf), dim=-1)
    indices = torch.cat([pad_indices.reshape(-1, 2), tsdf_inds.reshape(-1).unsqueeze(-1)], dim=-1)

    tsdf_offset = tsdf[indices[:, 0], indices[:, 1], indices[:, 2]].reshape(
        n_batches, n_pts
    )
    closest_pts = pts[indices[:, 0], indices[:, 1], indices[:, 2]].reshape(
        n_batches, n_pts, 3
    )
    pts = closest_pts + tsdf_offset.unsqueeze(-1) * ray_dirs
    return pts


def ray_sphere_intersection(ray_0, ray_direction):
    """ calculate intersection between rays and the unit sphere
    """

    t = torch.sum(-ray_0 * ray_direction, dim=-1)
    p = ray_0 + ray_direction * t.unsqueeze(-1)
    len_p = torch.norm(p, dim=-1)
    overlap_masks = len_p <= 1
    pos_x = torch.sqrt(1 - len_p ** 2)
    t0 = t - pos_x
    t1 = t + pos_x
    return t0, t1, overlap_masks


def image_points_to_world(image_points, camera_mat, world_mat, scale_mat,
                          invert=True):
    ''' Transforms points on image plane to world coordinates.
    In contrast to transform_to_world, no depth value is needed as points on
    the image plane have a fixed depth of 1.
    Args:
        image_points (tensor): image points tensor of size B x N x 2
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert matrices (default: true)
    '''
    batch_size, n_pts, dim = image_points.shape
    assert(dim == 2)
    device = image_points.device

    d_image = torch.ones(batch_size, n_pts, 1).to(device)
    return transform_to_world(image_points, d_image, camera_mat, world_mat,
                              scale_mat, invert=invert)


def transform_to_world(pixels, depth, K, T_cw, invert=True, is_numpy=False):
    ''' Transforms pixel positions p with given depth value d to world coordinates.
    Args:
        pixels (tensor): pixel tensor of size B x N x 2
        depth (tensor): depth tensor of size B x N x 1
        K (tensor): camera matrix
        T_cw (tensor): world matrix T_cw
    ''' 
    assert(pixels.shape[-1] == 2)

    if invert:
        cam_mat = torch.inverse(K)
        world_mat = torch.inverse(T_cw)

    # Transform pixels to homogen coordinates
    pixels = pixels.permute(0, 2, 1)
    pixels = torch.cat([pixels, torch.ones_like(pixels)], dim=1)

    # Project pixels into camera space
    pixels[:, :3] = pixels[:, :3] * depth.permute(0, 2, 1)

    # Transform pixels to world space
    p_world = world_mat @ cam_mat @ pixels

    # Transform p_world back to 3D coordinates
    p_world = p_world[:, :3].permute(0, 2, 1)

    if is_numpy:
        p_world = p_world.numpy()
    return p_world


def sample_dists(start_dists, end_dists, n_samples=50):
    """
        start_dists: [n_pts, n_voxels]
        end_dists: [n_pts, n_voxels]
    """

    if torch.any(start_dists[:, 0] == 1.25):
        import pdb
        pdb.set_trace()
    n_pixels = len(start_dists)
    pixel_id_pad = torch.arange(
        0, n_pixels
    ).unsqueeze(-1).repeat(1, n_samples).reshape(-1).to(start_dists.device)
    dist_range = end_dists - start_dists
    assert torch.min(dist_range) >= 0
    weights = dist_range / torch.sum(dist_range, dim=1, keepdim=True)
    indices = torch.multinomial(weights, num_samples=n_samples, replacement=True)
    full_indices = torch.stack([pixel_id_pad, indices.reshape(-1)], dim=-1)

    assert torch.min(dist_range[full_indices[:, 0], full_indices[:, 1]]) > 0
    proportion = torch.rand(size=(n_pixels * n_samples, )).to(start_dists.device)
    sampled_dists = start_dists[full_indices[:, 0], full_indices[:, 1]] +\
        dist_range[full_indices[:, 0], full_indices[:, 1]] * proportion
    sampled_dists = sampled_dists.reshape(n_pixels, n_samples)
    return sampled_dists


def quat_to_rot(q):
    batch_size, _ = q.shape
    q = F.normalize(q, dim=1)
    R = torch.ones((batch_size, 3,3)).cuda()
    qr=q[:,0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]
    R[:, 0, 0]=1-2 * (qj**2 + qk**2)
    R[:, 0, 1] = 2 * (qj *qi -qk*qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1-2 * (qi**2 + qk**2)
    R[:, 1, 2] = 2*(qj*qk - qi*qr)
    R[:, 2, 0] = 2 * (qk * qi-qj * qr)
    R[:, 2, 1] = 2 * (qj*qk + qi*qr)
    R[:, 2, 2] = 1-2 * (qi**2 + qj**2)
    return R


def lift(x, y, z, intrinsics):
    # parse intrinsics
    device = x.device
    intrinsics = intrinsics.to(device)
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    sk = intrinsics[:, 0, 1]

    x_lift = (x - cx.unsqueeze(-1) + cy.unsqueeze(-1)*sk.unsqueeze(-1)/fy.unsqueeze(-1) - sk.unsqueeze(-1)*y/fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z
    y_lift = (y - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z

    # homogeneous
    return torch.stack(
        (x_lift, y_lift, z, torch.ones_like(z).to(device)),
        dim=-1
    )


def get_camera_params(uv, pose, intrinsics):
    if pose.shape[1] == 7: #In case of quaternion vector representation
        cam_loc = pose[:, 4:]
        R = quat_to_rot(pose[:,:4])
        p = torch.eye(4).repeat(pose.shape[0],1,1).cuda().float()
        p[:, :3, :3] = R
        p[:, :3, 3] = cam_loc
    else: # In case of pose matrix representation
        cam_loc = pose[:, :3, 3]
        p = pose

    batch_size, num_samples, _ = uv.shape

    # depth = torch.ones((batch_size, num_samples)).to(uv.device)
    x_cam = uv[:, :, 0].view(batch_size, -1)
    y_cam = uv[:, :, 1].view(batch_size, -1)
    # z_cam = depth.view(batch_size, -1)
    z_cam = uv[:, :, 0] * 0. + 1.
    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics)

    # permute for batch matrix product
    pixel_points_cam = pixel_points_cam.permute(0, 2, 1)

    world_coords = torch.bmm(p, pixel_points_cam).permute(0, 2, 1)[:, :, :3]
    ray_dirs = world_coords - cam_loc[:, None, :]
    ray_dirs = F.normalize(ray_dirs, dim=2)

    return ray_dirs, cam_loc


def render_with_rays(
    volume,
    rays,
    nerf,
    sdf_delta,
    truncated_units,
    truncated_dist,
    ray_max_dist
):
    """
    rays:
        uv [b, n, 2]:
        gt_pts [b, n, 3]: gt world positions of rays
        T_wc [b, 4, 4]:
        intr_mat [b, 3, 3]:
    """

    uv = rays['uv']
    T_wcs = rays['T_wc']
    intr_mats = rays['intr_mat']
    ray_dirs, cam_loc = \
        get_camera_params(uv, T_wcs, intr_mats)
    gt_depths = torch.sqrt(torch.sum(
        (rays['gt_pts'] - cam_loc.unsqueeze(1)) ** 2,
        dim=-1
    ))  # [v, n_pts]
    pts, dists = hierarchical_sampling(
        truncated_units*2,
        int(ray_max_dist*5), gt_depths, rays['gt_pts'],
        ray_dirs, cam_loc, offset_distance=truncated_dist,
        max_depth=ray_max_dist
    )
    coords = (pts - volume.min_coords) / volume.voxel_size
    coords = get_neighbors(coords)
    volume.count_optim(coords)
    pred_sdf = volume.decode_pts(
        pts, nerf, sdf_delta=sdf_delta)
    pred_sdf = pred_sdf[..., 0]
    out = {
        "cam_loc": cam_loc,
        "ray_dirs": ray_dirs,
        "sdf_on_rays": pred_sdf,
        "pts_on_rays": pts,
    }
    return out


def compute_sdf_loss(
    rays,
    pred_sdf,
    pred_pts,
    cam_loc,
    num_valid_pixels,
    truncated_dist
):
    gt_depths = torch.sqrt(torch.sum(
        (rays['gt_pts'] - cam_loc.unsqueeze(1)) ** 2,
        dim=-1
    )).unsqueeze(-1)  # [1, n_pts, 1]

    depths = torch.sqrt(torch.sum(
        (pred_pts - cam_loc.unsqueeze(1).unsqueeze(1)) ** 2,
        dim=-1
    ))  # [1, n_pts, n_steps]
    gt_sdf = gt_depths - depths
    gt_sdf = torch.clip(
        gt_sdf, min=-truncated_dist, max=truncated_dist)
    valid_map = gt_sdf > max(-truncated_dist*0.5, -0.05)
    dists = torch.sqrt(
        torch.sum(
            (rays['neighbor_pts'].unsqueeze(2) - pred_pts.unsqueeze(3)) ** 2,
            dim=-1)
    )  # [1, n_pts, n_steps, n_neighbors]
    n_samples = pred_pts.shape[2]
    neighbor_mask = rays['neighbor_masks'].unsqueeze(2).repeat(1, 1, n_samples, 1) # [1, n_pts, n_steps, n_neighbors]
    dists = torch.where(neighbor_mask.bool(), dists, torch.ones_like(dists) * 10000)
    # get the corrected SDF using the minimum of a neighborhood
    gt_nearest_dists = torch.min(dists, dim=-1)[0]
    sign = torch.where(gt_sdf > 0, torch.ones_like(gt_sdf), torch.ones_like(gt_sdf) * -1)
    gt_nearest_signed_dists = gt_nearest_dists * sign
    gt_nearest_signed_dists = torch.clip(
        gt_nearest_signed_dists, min=-truncated_dist, max=truncated_dist)
    depth_bce = F.l1_loss(
        pred_sdf,
        gt_nearest_signed_dists,
        reduction='none'
    ) * valid_map
    depth_bce = (depth_bce * rays['mask'].unsqueeze(-1)).sum() / num_valid_pixels
    return depth_bce

def calculate_loss(
    volume,
    rays,
    nerf,
    truncated_units,
    truncated_dist,
    ray_max_dist,
    sdf_delta=None,
):
    """ calculate RGB and occupancy loss given rays
    geometry loss:
        zero_leve loss: points on the surface should be occupied
        ray bce loss: points before the gt_pts should be unoccupied
    rgb loss: the rendered images given surface pts and view directions

    rays:
        uv [v, n, 2]:
        gt_pts [v, n, 3]: gt world positions of rays
        T_wc [v, 4, 4]:
        intr [v, 4, 4]:
    """

    object_mask = rays['mask']
    num_valid_pixels = torch.sum(object_mask) + 1e-4 
    render_out = render_with_rays(
        volume,
        rays,
        nerf,
        sdf_delta,
        truncated_units,
        truncated_dist,
        ray_max_dist
    )
    sdf_loss = compute_sdf_loss(
        rays,
        render_out['sdf_on_rays'],
        render_out['pts_on_rays'],
        render_out['cam_loc'],
        num_valid_pixels,
        truncated_dist
    )
    loss_output = {}
    loss_output['depth_bce_loss'] = sdf_loss
    return loss_output


if __name__ == "__main__":
    import torch
    surface_pts = torch.zeros((2, 9, 3)).float().to("cuda")
    surface_pts[0, :, 2] = 2
    surface_pts[1, :, 2] = 1
    
    cam_loc = torch.zeros((2, 1, 3)).float().to("cuda")
    ray_dirs = torch.zeros((2, 9, 3)).float()
    ray_dirs[:, :, 2] = 1
    ray_dirs = ray_dirs.to("cuda")
    pts, _ = ray_sampling_around_surface(100, surface_pts, cam_loc, ray_dirs)