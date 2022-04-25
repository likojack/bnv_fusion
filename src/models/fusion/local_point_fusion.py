import numpy as np
from typing import Optional, List
import os
import open3d.core as o3c
import trimesh
import mcubes
from skimage.measure import marching_cubes_lewiner
from torch_scatter import scatter_mean

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from src.models.model_utils import set_optimizer_and_lr
from src.models.models import register
from src.models.fusion.modules import LocalNeRFModel
import src.utils.voxel_utils as voxel_utils
import src.utils.pointnet_utils as pointnet_utils


@register('lit_fusion_pointnet')
class LitFusionPointNet(pl.LightningModule):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.to(cfg.device_type)
        self.dense_volume = cfg.trainer.dense_volume
        self.feat_dims = cfg.model.feature_vector_size
        self.interpolate_decode = cfg.model.nerf.interpolate_decode
        self.pointnet_backbone = pointnet_utils.PointNetEncoder(
            self.feat_dims, **cfg.model.point_net)
        self.nerf = LocalNeRFModel(
            self.feat_dims, **cfg.model.nerf)
        self.voxel_size = cfg.model.voxel_size
        self.n_xyz = np.ceil((np.asarray(cfg.model.bound_max) - np.asarray(cfg.model.bound_min)) / self.voxel_size).astype(int).tolist()
        self.bound_min = torch.tensor(cfg.model.bound_min, device=self.device).float()
        self.bound_max = self.bound_min + self.voxel_size * torch.tensor(self.n_xyz, device=self.device)
        self.training_global = cfg.model.training_global
        self.loss_weight = cfg.model.loss
        self.min_pts_in_grid = cfg.model.min_pts_in_grid
        self.automatic_optimization = False
        if cfg.dataset.out_root is not None:
            self.plots_dir = os.path.join(cfg.dataset.out_root, cfg.dataset.scan_id)
        else:
            self.plots_dir = os.path.join(os.getcwd(), "plots")
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)

    def forward(self, input_feats, normalize, voxel_size=None, global_feats=True):
        """
        extract features using PointNet

        Args:
            input_feats: [B, N, F]. The first three dimensions are xyz coordinate
        """
        if normalize:
            input_feats[:, :, :3] = input_feats[:, :, :3] / voxel_size
            assert torch.min(input_feats[:, :, :3]) >= -1
            assert torch.max(input_feats[:, :, :3]) <= 1
        input_feats = input_feats.permute(0, 2, 1)
        point_feats = self.pointnet_backbone(
            input_feats, global_feats)  # [B, N, F]
        return point_feats

    def prune_pts(self, pts, min_coords, max_coords, volume_resolution):
        xyz = pts[:, :, :3]
        assert xyz.shape[0] == 1
        coords = voxel_utils.position_to_coords(
            xyz, min_coords, max_coords, volume_resolution)
        coords = torch.floor(coords + 0.5)
        flat_ids = voxel_utils.flatten(
            coords, volume_resolution).long()  # [1, N]
        _, unique_inv, unique_count = torch.unique(
            flat_ids[0], return_counts=True, return_inverse=True)
        valid_mask = (unique_count > self.min_pts_in_grid)[unique_inv]
        pts = pts[:, valid_mask, :]
        return pts

    def encode_pointcloud(
        self,
        input_pts,
        n_xyz,
        bound_min,
        bound_max,
        voxel_size,
        return_dense=True
    ):
        res_x, res_y, res_z = [int(v) for v in n_xyz]
        in_xyz = input_pts[:, :, :3] * 1.
        in_normal = input_pts[:, :, 3:]

        bound_mask = \
              (in_xyz[0, :, 0] < bound_max[0] - voxel_size) \
            * (in_xyz[0, :, 1] < bound_max[1] - voxel_size) \
            * (in_xyz[0, :, 2] < bound_max[2] - voxel_size) \
            * (in_xyz[0, :, 0] > bound_min[0] + voxel_size) \
            * (in_xyz[0, :, 1] > bound_min[1] + voxel_size) \
            * (in_xyz[0, :, 2] > bound_min[2] + voxel_size)

        in_xyz = in_xyz[:, bound_mask, :]
        in_normal = in_normal[:, bound_mask, :]

        relative_xyz, grid_id = self.get_relative_xyz(in_xyz, bound_min, voxel_size)
        grid_id = grid_id.reshape(1, -1, 3)  # [1, N, 3]
        pointnet_input = torch.cat(
            [relative_xyz, in_normal.unsqueeze(1).repeat(1, 8, 1, 1)],
            dim=-1
        )  # [1, N, 6]
        pointnet_input = pointnet_input.reshape(1, -1, 6)  # [1, N, 6]
        point_feats = self(pointnet_input, normalize=True, voxel_size=voxel_size, global_feats=False)
        # point_feats = self.pointnet_backbone(
        #     pointnet_input, False)  # [1, F, N]
        flat_ids = voxel_utils.flatten(
            grid_id, n_xyz).long()  # [1, N]
        unique_flat_ids, pinds, pcounts = torch.unique(
            flat_ids[0], return_inverse=True, return_counts=True)
        unique_grid_ids = voxel_utils.unflatten(unique_flat_ids, n_xyz).long()
        assert torch.max(unique_grid_ids[:, 0]) < res_x
        assert torch.max(unique_grid_ids[:, 1]) < res_y
        assert torch.max(unique_grid_ids[:, 2]) < res_z
        assert torch.min(unique_grid_ids) >= 0
        point_feats_mean = scatter_mean(point_feats, pinds.unsqueeze(0).unsqueeze(0))
        point_feats_mean[:, :, pcounts<self.min_pts_in_grid] = 0
        if return_dense:
            feat_grids = torch.zeros(
                (1, self.feat_dims, res_x, res_y, res_z),
                device=self.device,
                dtype=torch.float
            )
            mask = torch.zeros(
                (1, 1, res_x, res_y, res_z),
                device=self.device,
                dtype=torch.float
            )
            mask[0, 0, unique_grid_ids[:, 0], unique_grid_ids[:, 1], unique_grid_ids[:, 2]] = pcounts.float()
            # pcounts[pcounts < self.min_pts_in_grid] = 0
            feat_grids[0, :, unique_grid_ids[:, 0], unique_grid_ids[:, 1], unique_grid_ids[:, 2]] = point_feats_mean
            return feat_grids, mask, unique_flat_ids, flat_ids
        else:
            n_avg_pts = torch.mean(pcounts.float())
            valid_mask = pcounts >= self.min_pts_in_grid
            point_feats_mean = point_feats_mean[:, :, valid_mask]
            pcounts = pcounts[valid_mask]
            unique_flat_ids = unique_flat_ids[valid_mask]
            unique_grid_ids = voxel_utils.unflatten(unique_flat_ids, n_xyz).long()
            pcounts = pcounts.unsqueeze(-1)
            point_feats_mean = point_feats_mean[0].permute(1, 0)
            return point_feats_mean, pcounts, unique_flat_ids, unique_grid_ids, n_avg_pts

    def get_relative_xyz(
        self,
        xyz,  # [B, N, 3]
        bound_min,
        voxel_size
    ):
        xyz_zeroed = xyz - bound_min
        xyz_normalized = xyz_zeroed / voxel_size
        grid_id = self.nerf.get_neighbors(
            xyz_normalized.unsqueeze(1)).squeeze(2)  # [B, 8, N, 3]
        relative_xyz_normalized = xyz_normalized.unsqueeze(1) - grid_id
        relative_xyz = relative_xyz_normalized * voxel_size
        return relative_xyz, grid_id

    def decode_feature_grid(
        self,
        feat_grid,
        mask,
        n_xyz,
        voxel_size,
        bound_min,
        step_size=0.25,
        split_voxel_size=50
    ):
        # surface_pts = self.prune_pts(surface_pts, min_coords, max_coords, volume_resolution)
        # usee the finest level
        n_xyz = n_xyz
        voxel_size = voxel_size
        bound_min = bound_min

        res_x, res_y, res_z = [int(v) for v in n_xyz]
        volume_resolution = torch.tensor([res_x, res_y, res_z], device=self.device).float()
        x = np.arange(0, res_x-1, step_size)
        y = np.arange(0, res_y-1, step_size)
        z = np.arange(0, res_z-1, step_size)
        n_x, n_y, n_z = len(x), len(y), len(z)
        all_verts = []
        all_faces = []
        last_face_id = 0
        for i in range(0, n_x, split_voxel_size):
            i = max(i-1, 0)
            for j in range(0, n_y, split_voxel_size):
                j = max(j-1, 0)
                for k in range(0, n_z, split_voxel_size):
                    i_end = min(i + split_voxel_size, n_x)
                    j_end = min(j + split_voxel_size, n_y)
                    k_end = min(k + split_voxel_size, n_z)
                    k = max(k-1, 0)
                    sub_x = x[i: i_end]
                    sub_y = y[j: j_end]
                    sub_z = z[k: k_end]
                    h, w, d = len(sub_x), len(sub_y), len(sub_z)
                    spacing = [
                        sub_x[1] - sub_x[0],
                        sub_y[1] - sub_y[0],
                        sub_z[1] - sub_z[0]]
                    origin = np.array([sub_x[0], sub_y[0], sub_z[0]])
                    voxel_coords = np.stack(
                        np.meshgrid(sub_x, sub_y, sub_z, indexing="ij"),
                        axis=-1
                    ).reshape(1, -1, 3)  # [1, HWD, 3]
                    voxel_coords = torch.from_numpy(
                        voxel_coords).float().to(self.device)
                    sdf, _ = self.decode_feature_grid_w_pts(
                        voxel_coords, feat_grid, mask,
                        voxel_size, bound_min,
                        global_coords=self.cfg.model.global_coords
                    )
                    sdf = sdf.reshape(h, w, d)
                    sdf = sdf.detach().cpu().numpy()
                    if np.min(sdf) > 0 or np.max(sdf) < 0:
                        continue
                    verts, faces, _, _ = marching_cubes_lewiner(sdf, level=0.)
                    if len(verts) == 0:
                        continue
                    verts = verts * np.asarray(spacing)[None, :]
                    verts += origin
                    all_verts.append(verts)
                    all_faces.append(faces + last_face_id)
                    last_face_id += np.max(faces) + 1
        if len(all_verts) == 0:
            return None
        all_verts = np.concatenate(all_verts, axis=0)
        all_verts = all_verts * voxel_size
        all_verts = all_verts + bound_min.cpu().numpy()
        all_faces = np.concatenate(all_faces, axis=0)
        mesh = trimesh.Trimesh(
            vertices=all_verts,
            faces=all_faces,
            # vertex_normals=all_normals,
            process=False
        )
        return mesh

    def gradient(self, voxel_coords, feat_grid, mask, normalize=True):
        voxel_coords.requires_grad_(True)
        y = self.decode_feature_grid_w_pts(voxel_coords, feat_grid, mask)
        d_output = torch.ones_like(
            y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=voxel_coords,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )
        gradients = gradients[0].detach()
        if normalize:
            gradients = gradients / (torch.norm(gradients, dim=-1, keepdim=True) + 1e-5)
        return gradients

    def decode_feature_grid_w_pts(
        self, voxel_coords, feat_grid, pts_weight,
        voxel_size, bound_min,
        gradient=False, global_coords=True
    ):
        """ get sdf values ath the coords in the feature_grid.

        Args:
            coords: [1, N, 3]
            feat_grid: 
        Return:
            sdf: [1, N]
            gradient: [1, N, 3]
        """
        h, w, d = feat_grid.shape[-3:]
        volume_resolution = torch.tensor([h, w, d]).to(voxel_coords.device)
        if not global_coords:
            if self.interpolate_decode:
                neighbor_coords = self.nerf.get_neighbors(
                    voxel_coords.unsqueeze(1)
                ).squeeze(2)  # [1, 8, HWD, 3]
                coords_grid_sample = neighbor_coords / (volume_resolution-1)
                coords_grid_sample = coords_grid_sample * 2 - 1
                coords_grid_sample = coords_grid_sample[..., [2, 1, 0]]
                coords_grid_sample = coords_grid_sample.unsqueeze(0)
            else:
                neighbor_coords = torch.round(voxel_coords)
                coords_grid_sample = neighbor_coords / (volume_resolution-1)
                coords_grid_sample = coords_grid_sample * 2 - 1
                coords_grid_sample = coords_grid_sample[..., [2, 1, 0]]
                coords_grid_sample = coords_grid_sample.unsqueeze(0).unsqueeze(0)
            neighbor_feats = F.grid_sample(
                feat_grid,
                coords_grid_sample,  # [1, 1, 8, HWD, 3]
                mode="nearest",
                padding_mode="zeros",
                align_corners=True
            )
            pts_weight_mask = F.grid_sample(
                pts_weight,
                coords_grid_sample,  # [1, 1, 8, HWD, 3]
                mode="nearest",
                padding_mode="zeros",
                align_corners=True
            )
            pts_weight_mask = pts_weight_mask * (pts_weight_mask >= self.min_pts_in_grid)
            if self.interpolate_decode:
                neighbor_feats = neighbor_feats.squeeze(2).permute(0, 2, 3, 1)  # [1, 8, HWD, F]
                pts_weight_mask = pts_weight_mask.squeeze(2).permute(0, 2, 3, 1)[..., 0]  # [1, 8, HWD]
                relative_coords = voxel_coords.unsqueeze(1) - neighbor_coords
                bilinear_weights = torch.prod(
                    1 - torch.abs(relative_coords),
                    dim=-1,
                )  # [1, 8, HWD]
                normalizer = torch.sum(bilinear_weights, dim=1, keepdim=True)
                bilinear_weights = bilinear_weights / normalizer
                relative_xyz = relative_coords * voxel_size
                sdf = self.decode_implicit(
                    neighbor_feats, relative_xyz,
                    mask=pts_weight_mask >= self.min_pts_in_grid,
                    normalize=True, voxel_size=voxel_size, test=True
                )[..., 0]  # [1, 8, HWD]
                sdf = torch.sum(sdf * bilinear_weights, dim=1)
                valid_mask = torch.sum(pts_weight_mask, dim=1) > 0
                sdf = torch.where(valid_mask, sdf, torch.ones_like(sdf) * voxel_size)
            else:
                neighbor_feats = neighbor_feats.squeeze(2).squeeze(2).permute(0, 2, 1)  # [1, HWD, F]
                pts_weight_mask = pts_weight_mask.squeeze(2).squeeze(2).permute(0, 2, 1)[..., 0]  # [1, HWD]
                relative_coords = voxel_coords - neighbor_coords
                relative_xyz = relative_coords * voxel_size
                sdf = self.decode_implicit(
                    neighbor_feats, relative_xyz,
                    mask=pts_weight_mask >= self.min_pts_in_grid,
                    normalize=True, voxel_size=voxel_size, test=True
                )[..., 0]  # [1, HWD]
                valid_mask = pts_weight_mask > 0
                sdf = torch.where(valid_mask, sdf, torch.ones_like(sdf) * voxel_size)
        else:
            coords_grid_sample = voxel_coords / (volume_resolution-1)
            coords_grid_sample = coords_grid_sample * 2 - 1
            coords_grid_sample = coords_grid_sample[..., [2, 1, 0]]
            neighbor_feats = F.grid_sample(
                feat_grid,
                coords_grid_sample.unsqueeze(0).unsqueeze(0),  # [1, 1, 1, HWD, 3]
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True
            ).squeeze(2).squeeze(2).permute(0, 2, 1)  # [1, HWD, F]
            pts_weight_mask = F.grid_sample(
                pts_weight,
                coords_grid_sample.unsqueeze(0).unsqueeze(0),  # [1, 1, 1, HWD, 3]
                mode="nearest",
                padding_mode="zeros",
                align_corners=True
            ).squeeze(2).squeeze(2).permute(0, 2, 1)[..., 0]  # [1, HWD]
            pts_normalize = voxel_coords / (volume_resolution-1)
            sdf = self.decode_implicit(
                neighbor_feats, pts_normalize,
                normalize=False, test=True
            )[..., 0]  # [1, HWD]
            valid_mask = pts_weight_mask >= self.min_pts_in_grid
            sdf = torch.where(valid_mask, sdf, torch.ones_like(sdf) * voxel_size)
        if gradient:
            gradient = self.gradient(voxel_coords, feat_grid, mask)
            return sdf, gradient
        return sdf, neighbor_feats

    def decode_implicit(self, feat_grid, points, normalize, voxel_size=None, mask=None, test=False):
        if normalize:
            points = points / voxel_size
        predictions = self.nerf(points, feat_grid, mask, test)
        if normalize:
            # max_unsigned_dist = np.sqrt(3 * (2 * voxel_size) ** 2)
            predictions = predictions * voxel_size
        return predictions

    def compute_loss(self, data, point_feats):
        loss_output = {}
        predictions = self.decode_implicit(
            point_feats,
            data['training_pts'],
            normalize=False
        )
        gt = data['gt'].unsqueeze(-1)
        bce_loss = F.l1_loss(
            predictions, gt
        )
        loss_output['bce_loss'] = bce_loss
        reg_loss = torch.norm(point_feats, dim=1)
        loss_output['reg_loss'] = torch.mean(reg_loss)
        return loss_output

    def training_step(self, data, batch_idx):
        """
        data:
            frame: new frame information
            volume: world feature volume
            rays: rays for supervision
        """

        opt = self.optimizers()
        opt.zero_grad()

        for k in data:
            if isinstance(data[k], torch.Tensor):
                data[k] = data[k].float()
        batch_loss = {}
        if not self.training_global:
            ind = torch.randint(
                low=int(self.cfg.dataset.n_local_samples/2),
                high=self.cfg.dataset.n_local_samples,
                size=(1,)
            ).item()
            input_ = data['input_pts'][:, :ind, :]
            point_feats = self(input_, normalize=False)
            loss_out = self.compute_loss(data, point_feats)
        else:
            n_xyz = data['n_xyz'][0].cpu().numpy().astype(np.int32).tolist()
            feat_grids, pts_weight, _, _ = self.encode_pointcloud(
                data['input_pts'],
                n_xyz,
                data['bound_min'][0],
                data['bound_max'][0],
                self.voxel_size
            )
            voxel_coords = (data['training_pts'] - data['bound_min']) / self.voxel_size
            pred_sdf, _ = self.decode_feature_grid_w_pts(
                voxel_coords,
                feat_grids,
                pts_weight,
                self.voxel_size,
                data['bound_min'],
                gradient=False,
                global_coords=self.cfg.model.global_coords
            )
            pred_sdf = pred_sdf[-1]
            loss_out = {}
            loss_out['bce_loss'] = F.l1_loss(pred_sdf, data['gt'])

        loss_for_backward = 0
        for k in loss_out:
            if k[0] != "_":
                try:
                    weight = getattr(self.loss_weight, k)
                except AttributeError:
                    log.info("[warning]: can't find loss weight")
                loss_for_backward += loss_out[k] * weight
                if k not in batch_loss:
                    batch_loss[k] = loss_out[k]
                else:
                    batch_loss[k] += loss_out[k]
        self.manual_backward(loss_for_backward)
        for k in batch_loss:
            self.log(f"train/{k}", batch_loss[k])
        opt.step()
        return loss_for_backward

    def validation_step(self, data, batch_idx):
        """
        data:
            frame: new frame information
            volume: world feature volume
            rays: rays for supervision
        """
        for k in data:
            if isinstance(data[k], torch.Tensor):
                data[k] = data[k][0].float()
        if not self.training_global:
            batch_loss = {}
            n_pts = data['input_pts'].shape[1]
            point_feats = self(data['input_pts'], normalize=False)
            loss_out = self.compute_loss(data, point_feats)
            for k in loss_out:
                if k[0] != "_":
                    try:
                        weight = getattr(self.loss_weight, k)
                    except AttributeError:
                        log.info("[warning]: can't find loss weight")

                    if k not in batch_loss:
                        batch_loss[k] = loss_out[k]
                    else:
                        batch_loss[k] += loss_out[k]
            for k in batch_loss:
                self.log(f"val/{k}", batch_loss[k])
            self.log("val_loss", batch_loss['bce_loss'])

            all_verts = np.empty((0, 3))
            all_faces = np.empty((0, 3))
            gt_pts = np.empty((0, 3))
            normals = np.empty((0, 3))
            last_face_id = 0
            out_path = os.path.join(
                self.plots_dir,
                f"{data['scene_id'][0]}_{self.current_epoch}.ply"
            )
            out_gt_path = os.path.join(
                self.plots_dir,
                f"{data['scene_id'][0]}_{self.current_epoch}_gt.ply"
            )
            sample_centers = data['sample_center'].cpu().numpy()
            input_pts = data['input_pts'].cpu().numpy()
            input_xyz = input_pts[:, :, :3] * self.voxel_size
            input_normal = input_pts[:, :, 3:]
            n_samples = point_feats.shape[0]

            for i in range(n_samples):
                gt_pts = np.concatenate(
                    [gt_pts, input_xyz[i] + sample_centers[i:i+1]],
                    axis=0
                )
                normals = np.concatenate(
                    [normals, input_normal[i]],
                    axis=0
                )
                mesh = self.meshing_local_patch(point_feats[i:i+1])
                if mesh is not None:
                    all_verts = np.concatenate(
                        [all_verts, mesh.vertices + sample_centers[i:i+1]],
                        axis=0
                    )
                    all_faces = np.concatenate(
                        [all_faces, mesh.faces + last_face_id],
                        axis=0
                    )
                    last_face_id += np.max(mesh.faces) + 1
            mesh = trimesh.Trimesh(
                vertices=gt_pts,
                faces=np.zeros_like(gt_pts),
                vertex_colors=((normals * 0.5 + 0.5) * 255).astype(np.int32),
                process=False
            )
            mesh.export(out_gt_path)
            if len(all_verts) > 0:
                mesh = trimesh.Trimesh(
                    vertices=all_verts,
                    faces=all_faces.astype(np.int32),
                    process=False
                )
                mesh.export(out_path)

            all_xyz = input_xyz + sample_centers[:, None, :]
            all_pts = np.concatenate([all_xyz, input_normal], axis=-1)
            all_pts = all_pts.reshape(1, -1, 6)
            mesh = self.generate_global_shape(all_pts)
            out_path = out_path[:-4] + "_global.ply"
            if mesh is not None:
                mesh.export(out_path)
        else:
            if batch_idx % 10 != 0:
                return None
            n_xyz = data['n_xyz'].cpu().numpy().astype(np.int32).tolist()
            feat_grids, pts_weight, _, _ = self.encode_pointcloud(
                data['input_pts'].unsqueeze(0),
                n_xyz,
                data['bound_min'],
                data['bound_max'],
                self.voxel_size
            )
            voxel_coords = (data['training_pts'] - data['bound_min']) / self.voxel_size
            pred_sdf, _ = self.decode_feature_grid_w_pts(
                voxel_coords.unsqueeze(0),
                feat_grids,
                pts_weight,
                self.voxel_size,
                data['bound_min'],
                gradient=False,
                global_coords=self.cfg.model.global_coords
            )
            pred_sdf = pred_sdf[-1]
            val_loss = F.l1_loss(pred_sdf, data['gt'].unsqueeze(0))
            self.log("val_loss", val_loss)
            mesh = self.decode_feature_grid(
                feat_grids, pts_weight,
                n_xyz, self.voxel_size, data['bound_min']
            )
            out_path = os.path.join(
                self.plots_dir,
                f"{data['scene_id'][0]}_{data['frame_id'][0]}_{self.current_epoch}.ply"
            )
            if mesh is not None:
                mesh.export(out_path)
            out_gt_path = os.path.join(
                self.plots_dir,
                f"{data['scene_id'][0]}_{data['frame_id'][0]}_{self.current_epoch}_gt.ply"
            )
            input_pts = data['input_pts'].cpu().numpy()
            mesh_gt = trimesh.Trimesh(
                vertices=input_pts[:, :3],
                faces=np.zeros_like(input_pts[:, :3]),
                vertex_colors=((input_pts[:, 3:] * 0.5 + 0.5) * 255).astype(np.int32),
                process=False
            )
            mesh_gt.export(out_gt_path)

    def generate_global_shape(self, pts):
        # volume_resolution = np.ceil((max_coords - min_coords) / self.voxel_size)
        res_x, res_y, res_z = [int(v) for v in self.n_xyz]

        pts = torch.from_numpy(pts).float().to(self.device)
        feat_grids, mask, unique_flat_ids, flat_ids = self.encode_pointcloud(
            pts, self.n_xyz, self.bound_min, self.bound_max, self.voxel_size)
        in_xyz = pts[:, :, :3]
        in_normal = pts[:, :, 3:]

        relative_xyz, grid_id = self.get_relative_xyz(in_xyz, self.bound_min, self.voxel_size)
        grid_id = grid_id.reshape(1, -1, 3)  # [1, 8N, 3]
        pointnet_input = torch.cat(
            [relative_xyz, in_normal.unsqueeze(1).repeat(1, 8, 1, 1)],
            dim=-1
        )  # [1, 8, N, 6]
        pointnet_input = pointnet_input.reshape(1, -1, 6)  # [1, 8N, 6]
        pointnet_input = pointnet_input.permute(0, 2, 1)

        mesh = self.decode_feature_grid(
            feat_grids, mask, self.n_xyz, self.voxel_size, self.bound_min)

        return mesh

    def on_test_epoch_start(self):
        self.test_dataloader.dataloader.dataset.init_volumes()
        volumes = self.test_dataloader.dataloader.dataset.volume_list
        scan_id = [l for l in volumes.keys()][0]
        # self.coarse_n_xyz = volumes[scan_id].coarse_n_xyz.long()
        # self.coarse_bound_min = volumes[scan_id].coarse_min_coords
        # self.coarse_bound_max = volumes[scan_id].coarse_max_coords
        # self.coarse_voxel_size = volumes[scan_id].coarse_voxel_size

        self.fine_n_xyz = volumes[scan_id].fine_n_xyz.long()
        self.fine_bound_min = volumes[scan_id].fine_min_coords
        self.fine_bound_max = volumes[scan_id].fine_max_coords
        self.fine_voxel_size = volumes[scan_id].fine_voxel_size

    def _apply_weight_mask(self, feats, weights, coords, thres):
        feats = feats[0].permute(1, 0)
        weight_masks = weights > thres
        feats = feats[weight_masks]
        weights = weights[weight_masks]
        coords = coords[weight_masks]
        weights = weights.unsqueeze(-1)
        return feats, weights, coords

    def _update(self, new_feats, new_weights, old_feats, old_weights):
        # update the feats, weights, and num_hits using the current frame
        updated_weights = old_weights + new_weights
        new_feats = (old_feats * old_weights + new_feats * new_weights) / updated_weights
        return new_feats, updated_weights

    def _integrate(
        self,
        volume_object,
        fine_coords,
        fine_feats,
        fine_weights
    ):
        fine_weights = torch.clip(fine_weights / 32, max=1)
        model_feats, model_weights, model_num_hits = volume_object.query(fine_coords)
        if len(fine_coords) > 0:
            new_fine_feats, new_fine_weights = self._update(
                fine_feats, fine_weights, model_feats, model_weights)
        else:
            new_fine_feats, new_fine_weights = None, None
        # fuse the updated info into the global feature volumes
        volume_object.insert(
            fine_coords,
            new_fine_feats,
            new_fine_weights,
            model_num_hits
        )
    
    def _overwrite_by_surface(
        self,
        n_xyz,
        empty_coords,
        empty_feats,
        empty_weights,
        surface_coords,
        surface_feats,
        surface_weights
    ):
        n_feats = surface_feats.shape[-1]
        flat_empty_keys = empty_coords[:, 0] * n_xyz[1] * n_xyz[2] + empty_coords[:, 1] * n_xyz[2] + empty_coords[:, 2]
        flat_empty_keys = flat_empty_keys.unsqueeze(-1)
        flat_surface_keys = surface_coords[:, 0] * n_xyz[1] * n_xyz[2] + surface_coords[:, 1] * n_xyz[2] + surface_coords[:, 2]
        flat_surface_keys = flat_surface_keys.unsqueeze(-1)
        hash_map = o3c.HashMap(
            len(flat_empty_keys),
            key_dtype=o3c.int64,
            key_element_shape=(1,),
            value_dtypes=(o3c.Dtype.Float32, o3c.Dtype.Float32),
            value_element_shapes=((n_feats,), (1,)),
            device=o3c.Device("cuda:0")
        )
        o3c_keys = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(flat_surface_keys)).to(o3c.int64)
        feats_o3c = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(surface_feats))
        surface_weights = surface_weights.float()
        weights_o3c = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(surface_weights))
        buf_indices, mask = hash_map.insert(o3c_keys, (feats_o3c, weights_o3c))
        assert mask.cpu().numpy().all()

        o3c_keys = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(flat_empty_keys)).to(o3c.int64)
        feats_o3c = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(empty_feats))
        empty_weights = empty_weights.float()
        weights_o3c = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(empty_weights))
        buf_indices, masks = hash_map.insert(o3c_keys, (feats_o3c, weights_o3c))

        active_buf_indices = hash_map.active_buf_indices().to(o3c.int64)
        active_keys = hash_map.key_tensor()[active_buf_indices].to(o3c.int64)
        active_keys = torch.utils.dlpack.from_dlpack(active_keys.to_dlpack())
        active_keys = voxel_utils.unflatten(active_keys.squeeze(1), n_xyz)
        features = torch.utils.dlpack.from_dlpack(
            hash_map.value_tensor(0)[active_buf_indices].to_dlpack())
        weights = torch.utils.dlpack.from_dlpack(
            hash_map.value_tensor(1)[active_buf_indices].to_dlpack())
        return active_keys, features, weights

    def _get_truncated_coords_and_feats(self, voxel_coords, empty_sdf, voxel_size, n_feats, min_weights):
        mask = (empty_sdf > -5 * voxel_size) * \
            (empty_sdf < 5 * voxel_size) 
        valid_sdf = empty_sdf[mask]
        valid_coords = voxel_coords[mask]
        feats = torch.zeros(
            (valid_coords.shape[0], n_feats),
            device=self.cfg.device_type
        )
        weights = torch.ones(
            (valid_coords.shape[0], 1),
            device=self.cfg.device_type
        ) * min_weights
        return valid_coords, feats, weights 

    def test_step(self, data, batch_idx):
        with torch.no_grad():
            frame, _ = data
            for k in frame.keys():
                frame[k] = frame[k][0]
                if isinstance(frame[k], torch.Tensor):
                    frame[k] = frame[k].float()
            scene_id = frame['scene_id']
            if len(frame['input_pts']) == 0:
                return None
            fine_feats, fine_weights, _, fine_coords, fine_n_pts = self.encode_pointcloud(
                frame['input_pts'].unsqueeze(0),  # [1, N, 6]
                self.fine_n_xyz,
                self.fine_bound_min,
                self.fine_bound_max,
                self.fine_voxel_size,
                return_dense=self.dense_volume
            )

            if self.dense_volume:
                weight = weight.detach()[0]
                feat_grids = feat_grids.detach()[0]
                num_hits = (weight[0] > 0).float()

                volume_object = self.test_dataloader.dataloader.dataset.volume_list[scene_id]

                new_num_hits = volume_object.num_hits + num_hits

                weight = torch.where(weight >= self.min_pts_in_grid, weight, torch.zeros_like(weight))

                new_weight = volume_object.weight + weight

                # volume_object.volume = (volume_object.volume * volume_object.num_hits + feat_grids * num_hits) / torch.clip(new_num_hits, min=1)
                volume_object.volume = (volume_object.volume * volume_object.weight + feat_grids * weight) / torch.clip(new_weight, min=1)
                volume_object.num_hits = new_num_hits
                volume_object.weight = new_weight
            else:
                volume_object = self.test_dataloader.dataloader.dataset.volume_list[scene_id]
                volume_object.fine_volume.track_n_pts(fine_n_pts)                
                # voxel_coords, sdf = voxel_utils.depth_to_tsdf_tensor(
                #     frame['input_pts'][:, :3] * 1.,
                #     frame['rgbd'][-1, :, :],
                #     frame['T_wc'],
                #     frame['intr_mat'],
                #     self.fine_bound_min,
                #     self.fine_bound_max,
                #     self.fine_n_xyz,
                #     self.fine_voxel_size,
                #     device=self.cfg.device_type,
                # )
                # empty_coords, empty_feats, empty_weights = \
                #     self._get_truncated_coords_and_feats(
                #         voxel_coords, sdf, self.fine_voxel_size, self.feat_dims, 0)
                # # overwrite truncated region by surface features at the coarse level
                # fine_coords, fine_feats, fine_weights = \
                #     self._overwrite_by_surface(
                #         self.fine_n_xyz,
                #         empty_coords, empty_feats, empty_weights,
                #         fine_coords, fine_feats, fine_weights)
                self._integrate(
                    volume_object,
                    fine_coords,
                    fine_feats,
                    fine_weights)

    def test_epoch_end(self, test_step_outs):
        for scene in self.test_dataloader.dataloader.dataset.volume_list:
            volume_object = \
                self.test_dataloader.dataloader.dataset.volume_list[scene]
            # print(f"coarse volume:")
            # volume_object.coarse_volume.print_statistic()
            print(f"fine volume:")
            volume_object.fine_volume.print_statistic()
            
            scene_name = scene.split("/")[1] if "/" in scene else scene
            if not self.dense_volume:
                volume_object.to_tensor()
                mesh_path = os.path.join(self.plots_dir, scene_name + ".ply")
                surface_pts, mesh = volume_object.meshlize(
                    self.nerf,
                    # self.voxel_size,
                    path=mesh_path
                )
                save_path = os.path.join(self.plots_dir, scene_name)
                mesh = trimesh.Trimesh(
                    vertices=surface_pts
                )
                mesh.export(os.path.join(self.plots_dir, scene_name + "_pts_only.ply"))
                volume_object.save(save_path)
                # pts_o3d = o3d_helper.np2pc(pts)
                # o3d.visualization.draw_geometries([pts_o3d])

            else:
                volume = volume_object.volume.to(self.device).unsqueeze(0)
                weight = volume_object.weight.to(self.device).unsqueeze(0)
                num_hits = volume_object.num_hits.to(self.device).unsqueeze(0)
                feat_grids = volume
                # weight = weight * (num_hits > 2)
                # feat_grids = volume / torch.clip(num_hits, min=1)

                # sdf = volume_object.sdf
                # sdf_weight = volume_object.sdf_weight
                # sdf = sdf / torch.clip(sdf_weight, min=1)
                # sdf = sdf.unsqueeze(0).unsqueeze(0).to(self.device)
                # import open3d as o3d
                # import src.utils.o3d_helper as o3d_helper
                # pts = torch.nonzero(sdf[0][0]<0).cpu().numpy()
                # o3d.visualization.draw_geometries([o3d_helper.np2pc(pts)])

                weight = torch.where(weight >= self.min_pts_in_grid, weight, torch.zeros_like(weight))
                weights = weight
                # weight = torch.ones_like(weight) * self.min_pts_in_grid
                mesh = self.decode_feature_grid(
                    feat_grids, weights, self.n_xyz, self.voxel_size, self.bound_min,
                    step_size=0.5,
                )
                mesh.export(os.path.join(
                    self.plots_dir, scene_name + ".ply"))
                volume_out_path = os.path.join(
                    self.plots_dir,
                    f"volume_{scene_name}_{self.current_epoch}.pt"
                )
                out = {
                    "feat_grids": feat_grids.detach().cpu(),
                    "weights": weights.detach().cpu(),
                    "num_hits": num_hits.detach().cpu(),
                }
                torch.save(out, volume_out_path)
                self.test_dataloader.dataloader.dataset.reset_volumes()

    def _generate_sub_mesh(
        self,
        pts,
        feat_grid
    ):
        H, W, D = pts.shape[:3]
        pts = pts.reshape(-1, 3)
        n_pts = pts.shape[0]
        start_ind = 0
        sub_pts_size = 100000
        occupancy = []
        while start_ind < n_pts:
            end_ind = min(start_ind + sub_pts_size, n_pts)
            sub_pts = pts[start_ind: end_ind, :]
            sub_pts = sub_pts.reshape(1, -1, 3)
            sub_pts = torch.from_numpy(sub_pts).float().to(self.device)
            out = self.decode_implicit(feat_grid, sub_pts, normalize=True, voxel_size=self.voxel_size, test=False)
            # out = self.nerf(sub_pts, feat_grid) * self.voxel_size
            occupancy.append(out[0, :, 0])
            start_ind = end_ind
        occupancy = torch.cat(occupancy, dim=0)
        occupancy = occupancy.reshape(H, W, D)
        return occupancy

    def meshing_local_patch(self, point_feats):
        steps = 10
        x = (np.arange(0, steps+1) / steps * 2 - 1) * self.voxel_size
        y = (np.arange(0, steps+1) / steps * 2 - 1) * self.voxel_size
        z = (np.arange(0, steps+1) / steps * 2 - 1) * self.voxel_size
        n_x, n_y, n_z = len(x), len(y), len(z)
        spacing = np.asarray([
            x[1] - x[0],
            y[1] - y[0],
            z[1] - z[0]
        ])
        pts = np.stack(
            np.meshgrid(x, y, z, indexing="ij"),
            axis=-1
        )
        occupancy_grid = self._generate_sub_mesh(
            pts,
            point_feats,
        )
        occupancy_grid = occupancy_grid.detach().cpu().numpy()
        if np.max(occupancy_grid) < 0.:
            return None
        if np.min(occupancy_grid) > 0.:
            return None
        visible = np.ones_like(occupancy_grid)
        verts, faces = mcubes.marching_cubes(
            -occupancy_grid, 0
        )
        verts = (verts - steps / 2) * spacing[None, :]
        if len(verts) == 0:
            return None
        mesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            process=False
        )
        return mesh

    def generate_mesh(self, data, point_feats, batch_idx, write=True):
        """ generate mesh of the first batch
        """
        point_feats = point_feats[:1]
        mesh = self.meshing_local_patch(point_feats)
        if mesh is None:
            return None
        if not write:
            return mesh
        out_path = os.path.join(
            self.plots_dir,
            f"{data['scene_id'][0]}_{self.current_epoch}_{batch_idx}.ply"
        )
        mesh.export(out_path)

        input_pts = data['input_pts']
        pos = input_pts[0, :, :3].detach().cpu().numpy()
        mesh_gt = trimesh.Trimesh(
            vertices=pos,
            faces=np.zeros_like(pos),
            process=False,
        )
        out_path = os.path.join(
            self.plots_dir,
            f"{data['scene_id'][0]}_{self.current_epoch}_{batch_idx}_gt.ply"
        )
        mesh_gt.export(out_path)

    def configure_optimizers(self):
        parameters = self.parameters()
        optimizer, lr_dict = set_optimizer_and_lr(
            self.cfg,
            parameters
        )
        return [optimizer], [lr_dict]