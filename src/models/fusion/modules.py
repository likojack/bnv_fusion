import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.fusion.embedder import *
from src.models.models import register


class Implicit(nn.Module):
    def __init__(
        self,
        feat_vector,
        d_in,
        d_out,
        dims,
        bias,
        skip_in,
        weight_norm,
        multires,
    ):
        super().__init__()
        dims = [d for d in dims]
        dims = [feat_vector] + dims + [d_out + feat_vector]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] += input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)
            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

    def forward(self, volume, coords, scale):
        coords = coords / scale  # normalize x to [-1, 1] for grid sampling
        b, n_pts = coords.shape[:2]
        coords = coords.reshape(1, b * n_pts, 3)
        in_feats = F.grid_sample(
            volume,
            coords.unsqueeze(1).unsqueeze(1),
            mode="bilinear",
            align_corners=True
        )
        in_feats = in_feats.squeeze(2).squeeze(2).permute(0, 2, 1)
        in_feats = in_feats.reshape(b, n_pts, -1)
        coords = coords.reshape(b, n_pts, 3)
        if self.embed_fn is not None:
            coords = self.embed_fn(coords)
        x0 = torch.cat([in_feats, coords], dim=-1)
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, x0], -1)
            if l == 0:
                x = lin(x0)
            else:
                x = lin(x)

            if l < self.num_layers - 2:
                x = F.relu(x)
        return x


def positional_encoding(
    tensor, num_encoding_functions=6, include_input=True, log_sampling=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)


def get_embedding_function(
    num_encoding_functions=6, include_input=True, log_sampling=True
):
    r"""Returns a lambda function that internally calls positional_encoding.
    """
    return lambda x: positional_encoding(
        x, num_encoding_functions, include_input, log_sampling
    )


class ReplicateNeRFModel(torch.nn.Module):
    r"""NeRF model that follows the figure (from the supp. material of NeRF) to
    every last detail. (ofc, with some flexibility)
    """

    def __init__(
        self,
        feat_dims,
        hidden_size=256,
        num_layers=4,
        num_encoding_fn_xyz=8,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        xyz_agnostic=False,
        interpolate_decode=True,
        global_coords=False
    ):
        super(ReplicateNeRFModel, self).__init__()
        self.dim_xyz = (3 if include_input_xyz else 0) + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = (3 if include_input_dir else 0) + 2 * 3 * num_encoding_fn_dir
        self.hidden_size = hidden_size
        self.xyz_agnostic = xyz_agnostic
        self.interpolate_decode = interpolate_decode
        self.global_coords = global_coords

        dims = [self.dim_xyz + feat_dims] + [hidden_size] * num_layers
        for i, dim in enumerate(dims[:-1]):
            layer = torch.nn.Linear(dim, dims[i+1])
            setattr(self, f"geo_layer{i}", layer)

        self.fc_alpha = torch.nn.Linear(hidden_size, 1)

        dims = [hidden_size + self.dim_dir] + [hidden_size // 2] * int(num_layers/2)
        for i, dim in enumerate(dims[:-1]):
            layer = torch.nn.Linear(dim, dims[i+1])
            setattr(self, f"color_layer{i}", layer)
        self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
        self.relu = torch.nn.functional.relu
        self.tanh = nn.Tanh()
        self.xyz_encoding = get_embedding_function(
            num_encoding_functions=num_encoding_fn_xyz,
            include_input=True,
            log_sampling=True
        )
        self.view_encoding = get_embedding_function(
            num_encoding_functions=num_encoding_fn_dir,
            include_input=True,
            log_sampling=True
        )
        self.num_layers = num_layers

    def get_neighbors(self, points):
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
        ], dim=1).int()

    def geo_forward(self, xyz):
        for i in range(self.num_layers):
            lin = getattr(self, f"geo_layer{i}")
            xyz = self.relu(lin(xyz))
        alpha = self.fc_alpha(xyz)
        return xyz, alpha

    def forward(
        self,
        x,
        volume,
        weight_mask,
        sdf_delta,
        voxel_size,
        volume_resolution,
        min_coords,
        max_coords,
        active_voxels,
        geo_only=False
    ):
        if self.global_coords:
            return self.forward_global(
                x, volume, weight_mask, sdf_delta, voxel_size, volume_resolution, min_coords, max_coords,
                active_voxels, geo_only
            )
        else:
            assert self.xyz_agnostic != True, "local mode does not support xyz_agnostic"
            return self.forward_local(
                x, volume, weight_mask, sdf_delta, voxel_size, volume_resolution, min_coords,
                max_coords, active_voxels, geo_only
            )

    def forward_global(
        self, x, volume, weight_mask, sdf_delta, voxel_size, volume_resolution, min_coords,
        max_coords, active_voxels, geo_only=False
    ):
        xyz = x[..., :3]  # [b, n_pts, n_steps, 3]
        voxel_coordinates = (xyz - min_coords) / voxel_size
        neighbor_coords_grid_sample = voxel_coordinates / (volume_resolution-1)
        neighbor_coords_grid_sample = neighbor_coords_grid_sample * 2 - 1
        neighbor_coords_grid_sample = neighbor_coords_grid_sample[..., [2, 1, 0]]

        b, n_pts, n_samples = xyz.shape[:3]
        in_feats = F.grid_sample(
            volume,
            neighbor_coords_grid_sample.unsqueeze(0),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True
        )
        weight_mask = F.grid_sample(
            weight_mask,
            neighbor_coords_grid_sample.unsqueeze(0),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True
        )
        sdf_delta = F.grid_sample(
            sdf_delta,
            neighbor_coords_grid_sample.unsqueeze(0),  # [1, 8, n_pts, n_steps, 3]
            mode="nearest",
            padding_mode="zeros",
            align_corners=True
        )
        # for scale_factor in [2, 4]:
        #     tmp = self.sample_features(
        #         xyz_grid_sample.unsqueeze(0),
        #         volume,
        #         1./scale_factor
        #     )
        #     in_feats += tmp * 1. / scale_factor
        in_feats = in_feats[0].permute(1, 2, 3, 0)  # [1, n, s, f]
        weight_mask = weight_mask[0].permute(1, 2, 3, 0)  # [1, n, s, 1]
        sdf_delta = sdf_delta[0].permute(1, 2, 3, 0)
        xyz = self.xyz_encoding(neighbor_coords_grid_sample)
        if self.xyz_agnostic:
            xyz = xyz * 0
        total_pts = b * n_pts * n_samples
        geo_in = torch.cat([xyz, in_feats], dim=-1)

        if active_voxels is not None:
            geo_in = geo_in.reshape(total_pts, -1)
            out_feats = torch.zeros(
                (total_pts, self.hidden_size), device=geo_in.device).float()
            out_alpha = torch.zeros(
                (total_pts, 1), device=geo_in.device
            ).float() + 1
            feats, alpha = self.geo_forward(geo_in[masks.reshape(-1)])
            out_alpha[masks.reshape(-1)] = alpha
            out_feats[masks.reshape(-1)] = feats
            alpha = out_alpha.reshape(b, n_pts, n_samples, 1)
            feats = out_feats.reshape(b, n_pts, n_samples, self.hidden_size)
        else:
            # feats, alpha = self.geo_forward(geo_in)
            alpha = self.forward_with_mask(geo_in, weight_mask.bool())
            alpha = alpha + sdf_delta
        if geo_only:
            return alpha
        else:
            # direction = x[..., 3:]
            # direction = self.view_encoding(direction)
            # y_ = torch.cat((feats, direction), dim=-1)
            # for i in range(int(self.num_layers/2)):
            #     lin = getattr(self, f"color_layer{i}")
            #     y_ = self.relu(lin(y_))
            # rgb = self.tanh(self.fc_rgb(y_))
            rgb = torch.zeros_like(x[..., 3:])
            return torch.cat((rgb, alpha), dim=-1), in_feats

    def sample_features(self, coords, volume, scale_factor):
        """
        coords: [B, H, W, D, 3] in the range of [-1, 1]
        volume: [B, C, H, W, D]
        scale_factor
        """
        volume = F.interpolate(
            volume, scale_factor=scale_factor, mode="trilinear",
            align_corners=True
        )
        in_feats = F.grid_sample(
            volume,
            coords,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True
        )
        return in_feats

    def compute_num_hits(self, num_hits, coordinates, min_coords, max_coords):
        min_coords = min_coords.flatten()
        max_coords = max_coords.flatten()
        coordinates = coordinates.reshape(-1, 3)
        mask = (coordinates[:, 0] >= min_coords[0]) * (coordinates[:, 0] < max_coords[0]) * \
            (coordinates[:, 1] >= min_coords[1]) * (coordinates[:, 1] < max_coords[1]) * \
            (coordinates[:, 2] >= min_coords[2]) * (coordinates[:, 2] < max_coords[2])
        coordinates = coordinates[mask].long()
        num_hits[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] += 1
        return num_hits

    def forward_with_mask(self, input_feats, mask):
        shapes = [l for l in input_feats.shape]
        out_shapes = shapes[:-1] + [1]
        mask = mask.reshape(-1)
        input_feats = input_feats.reshape(-1, shapes[-1])
        out = torch.zeros_like(input_feats[:, :1])
        _, alpha = self.geo_forward(input_feats[mask])
        out[mask] = alpha
        out = out.reshape(out_shapes)
        return out

    def forward_local(
        self,
        x,
        volume,
        weight_mask,
        sdf_delta,
        voxel_size,
        volume_resolution,
        min_coords,
        max_coords,
        active_voxels=None,
        geo_only=False
    ):
        xyz = x[..., :3]  # [1, n_pts, n_steps, 3]
        optim_num_hits = torch.zeros(volume_resolution.flatten().int().cpu().numpy().tolist(), device=x.device)
        # [0, resolution-1]
        voxel_coordinates = (xyz - min_coords) / voxel_size
        if self.interpolate_decode:
            min_coords = min_coords.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            max_coords = max_coords.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            volume_resolution = volume_resolution.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            # [1, 8, n_pts, n_steps, 3]
            neighbor_coordinates = self.get_neighbors(voxel_coordinates)
            optim_num_hits = self.compute_num_hits(
                optim_num_hits, neighbor_coordinates,
                torch.zeros_like(volume_resolution), volume_resolution
            )
            local_coordinates = \
                (voxel_coordinates.unsqueeze(1) - neighbor_coordinates)
            neighbor_coords_grid_sample = neighbor_coordinates / (volume_resolution-1)
            neighbor_coords_grid_sample = neighbor_coords_grid_sample * 2 - 1
            neighbor_coords_grid_sample = neighbor_coords_grid_sample[..., [2, 1, 0]]
        else:
            neighbor_coordinates = torch.round(voxel_coordinates)
            local_coordinates = voxel_coordinates - neighbor_coordinates
            neighbor_coords_grid_sample = neighbor_coordinates / (volume_resolution-1)
            neighbor_coords_grid_sample = neighbor_coords_grid_sample * 2 - 1
            neighbor_coords_grid_sample = neighbor_coords_grid_sample[..., [2, 1, 0]]
            neighbor_coords_grid_sample = neighbor_coords_grid_sample.unsqueeze(0)
        assert torch.min(local_coordinates) >= -1
        assert torch.max(local_coordinates) <= 1
        # [1, feat_dims, 8, n_pts, n_samples]
        in_feats = F.grid_sample(
            volume,
            neighbor_coords_grid_sample,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True
        )
        sdf_delta = F.grid_sample(
            sdf_delta,
            neighbor_coords_grid_sample,  # [1, 8, n_pts, n_steps, 3]
            mode="nearest",
            padding_mode="zeros",
            align_corners=True
        )
        weight_mask = F.grid_sample(
            weight_mask,
            neighbor_coords_grid_sample,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True
        )
        if self.interpolate_decode:
            in_feats = in_feats.permute(0, 2, 3, 4, 1)  # [B, 8, N, S, F]
            sdf_delta = sdf_delta.permute(0, 2, 3, 4, 1)  # [B, 8, N, S, 1]
            weight_mask = weight_mask.permute(0, 2, 3, 4, 1)  # [B, 8, N, S, 1]
            weights_unmasked = torch.prod(
                1 - torch.abs(local_coordinates),
                dim=-1,
                keepdim=True
            )
            weights = torch.where(
                weight_mask.bool(),
                weights_unmasked,
                torch.zeros_like(weights_unmasked)
            )
            # FIXME: the sum of weights could be larger than 1 if the points
            # lie exactly on the border. normalize to 1 as follows:
            normalizer = torch.sum(weights, dim=1, keepdim=True)
            invalid_mask = normalizer == 0
            weights = weights / normalizer
            weights = torch.where(
                invalid_mask.repeat(1, 8, 1, 1, 1),
                torch.zeros_like(weights),
                weights
                )
            # weights should be either 0 for invalid points or 1 for valid points
            assert not torch.any(torch.isnan(weights))
            assert torch.all(
                torch.logical_or(
                    torch.abs(torch.sum(weights, dim=1) - 1) < 1e-5,
                    torch.sum(weights, dim=1) == 0
                )
            )
        else:
            in_feats = in_feats.permute(0, 2, 3, 4, 1).squeeze(1)  # [B, N, S, F]

        local_coords_encoded = self.xyz_encoding(local_coordinates)
        geo_in = torch.cat([local_coords_encoded, in_feats], dim=-1)
        # alpha: [1, 8, n_pts, n_samples, 1]
        # alpha = self.forward_with_mask(geo_in, weights.bool())
        feats, alpha = self.geo_forward(geo_in)
        # NOTE: un-normalization for sdf prediction when using pointnet pretrained network.
        # the output from the network is normalized to [-1, 1]
        alpha = alpha * voxel_size

        if self.interpolate_decode:
            # alpha = torch.sum(alpha * weights, dim=1)
            # alpha = torch.where(
            #     invalid_mask.squeeze(1),
            #     torch.zeros_like(alpha),
            #     alpha
            # )
            normalizer = torch.sum(weights_unmasked, dim=1, keepdim=True)
            weights_unmasked = weights_unmasked / normalizer
            assert torch.all(torch.abs(weights_unmasked.sum(1) - 1) < 1e-5)
            alpha = torch.sum(alpha * weights_unmasked, dim=1)
            sdf_delta = torch.sum(sdf_delta * weights_unmasked, dim=1)
            alpha = sdf_delta + alpha
        else:
            alpha = alpha
        if geo_only:
            return alpha, in_feats
        else:
            # direction = torch.zeros_like(local_coordinates)
            # direction = self.view_encoding(direction)
            # # [1, 8, n_pts, n_samples, feat_dims + view_encode_dims]
            # y_ = torch.cat((feats, direction), dim=-1)
            # for i in range(int(self.num_layers/2)):
            #     lin = getattr(self, f"color_layer{i}")
            #     y_ = self.relu(lin(y_))
            # rgb = self.tanh(self.fc_rgb(y_))
            # rgb = torch.sum(rgb * weights, dim=1)
            rgb = torch.zeros_like(alpha).repeat(1, 1, 1, 3)
            return torch.cat((rgb, sdf_delta, alpha), dim=-1), optim_num_hits


class LocalNeRFModel(ReplicateNeRFModel):
    def __init__(
        self,
        feat_dims,
        hidden_size=256,
        num_layers=4,
        num_encoding_fn_xyz=8,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        interpolate_decode=True,
        global_coords=False,
        xyz_agnostic=False,
    ):
        super(LocalNeRFModel, self).__init__(
            feat_dims, hidden_size, num_layers, num_encoding_fn_xyz,
            num_encoding_fn_dir, include_input_xyz, include_input_dir, interpolate_decode)

    def forward(
        self, x, feats, mask=None, test=False
    ):
        """
        if no test:
            x: [B, N, in_channels]
            feats: [B, F]
        else: feats and x is one-to-one matching
            x: [..., 3]
            feats: [..., F]
        """
        xyz = x[..., :3]  # [B, N, 3]
        local_coords_encoded = self.xyz_encoding(xyz)
        if test:
            geo_in = torch.cat([local_coords_encoded, feats], dim=-1)
            if mask is not None:
                alpha = self.forward_with_mask(geo_in, mask)
            else:
                feats, alpha = self.geo_forward(geo_in)
            return alpha
        else:
            assert len(xyz.shape) == 3  # [B, N, 3]
            assert len(feats.shape) == 2  # [B, F]
            N = xyz.shape[1]
            feats = feats.unsqueeze(1).repeat(1, N, 1)
        geo_in = torch.cat([local_coords_encoded, feats], dim=-1)
        if mask is not None:
            alpha = self.forward_with_mask(geo_in, mask)
        else:
            feats, alpha = self.geo_forward(geo_in)
        return alpha


if __name__ == "__main__":
    net = UpdateNet(
        feat_vector=32, n_input_channels=4,
        contraction=32, depth=3, double_factor=1.5
    )
    data = torch.zeros((1, 4, 240, 430)).float()
    print(net(data).shape)