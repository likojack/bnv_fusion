import numpy as np
from typing import Optional, List
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from src.models.sparse_volume import VolumeList
from src.models.model_utils import set_optimizer_and_lr
from src.models.models import register
from src.models.fusion.modules import ReplicateNeRFModel
from src.utils.render_utils import get_camera_params
from src.utils.render_utils import hierarchical_sampling
import src.utils.voxel_utils as voxel_utils
import src.utils.hydra_utils as hydra_utils
from src.utils.common import override_weights

log = hydra_utils.get_logger(__name__)


@register("lit_fusion_refiner")
class LitFusionRefiner(pl.LightningModule):
    def __init__(self, cfg, **kwargs):
        super().__init__()

        self.cfg = cfg
        self.voxel_size = cfg.model.voxel_size
        self.ray_max_dist = cfg.model.ray_tracer.ray_max_dist
        self.truncated_units = cfg.model.ray_tracer.truncated_units
        self.truncated_dist = min(self.truncated_units * self.voxel_size * 0.5, 0.1)
        self.train_ray_splits = cfg.model.train_ray_splits
        self.sdf_delta_weight = cfg.model.sdf_delta_weight
        self.loss_weight = cfg.model.loss
        
        self.nerf = ReplicateNeRFModel(
            cfg.model.feature_vector_size, **cfg.model.nerf)
        
        if hasattr(cfg.dataset, "out_root"):
            self.plots_dir = os.path.join(cfg.dataset.out_root, cfg.dataset.scan_id)
        else:
            self.plots_dir = os.path.join(os.getcwd(), "plots")
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)

        pretrained_weights = torch.load(cfg.model.pretrained_model)
        pretrained_used_weights = {k: v for k, v in pretrained_weights['state_dict'].items() if "nerf." in k}
        keys = [k for k in self.nerf.state_dict().keys()]

        if "/" in cfg.dataset.scan_id:
            dataset_name, scan_id = cfg.dataset.scan_id.split("/")
        else:
            scan_id = cfg.dataset.scan_id
            dataset_name = "ScanNet"

        if cfg.model.freeze_pretrained_weights:
            override_weights(self, pretrained_used_weights, keys=keys)
        # load volume
        if "volume_path" in kwargs:
            volume_path = kwargs['volume_path']
        else:
            volume_path = os.path.join(
                cfg.model.volume_dir,
                f"{scan_id}_fine_sparse_volume.pth"
            )
        volume = VolumeList(
            cfg.model.feature_vector_size,
            self.voxel_size,
            kwargs['dimensions'],
            cfg.model.min_pts_in_grid
        )
        volume.load(volume_path)
        self.volume = volume
        # load tsdf volume
        if "tsdf_volume_dir" in kwargs:
            tsdf_volume_dir = kwargs["tsdf_volume_dir"]
        else:
            tsdf_volume_dir = os.path.join(
                "/home/kejie/repository/fast_sdf/render_out/tsdf_volume",
                dataset_name
            )
        if not os.path.exists(tsdf_volume_dir):
            pass
        else:
            tsdf_name = [f for f in os.listdir(tsdf_volume_dir) if (scan_id in f) and (f.endswith(".npy"))]
            if len(tsdf_name) == 0:
                tsdf_volume_dir = "/not_exist"
            else:
                tsdf_name = tsdf_name[0]
                tsdf_volume_dir = os.path.join(tsdf_volume_dir, tsdf_name)

        if not os.path.exists(tsdf_volume_dir):
            print("[warning]: tsdf volume does not exist")
            self.sdf_delta = None
        else:
            world_min_coords, world_max_coords, _ = \
                voxel_utils.get_world_range(
                    kwargs['dimensions'], cfg.model.voxel_size
                )
            world_volume_resolution = np.ceil((world_max_coords - world_min_coords) / cfg.model.voxel_size).astype(np.int32)
            new_sdf_delta = []
            tsdf_volume = np.load(tsdf_volume_dir)
            tsdf_volume = torch.from_numpy(tsdf_volume).float().unsqueeze(0).unsqueeze(0)
            resized_tsdf_volume = F.interpolate(
                tsdf_volume,
                size=(
                    world_volume_resolution[0],
                    world_volume_resolution[1],
                    world_volume_resolution[2]
                ),
                mode="trilinear",
                align_corners=True)
            resized_tsdf_volume = torch.clip(
                resized_tsdf_volume, min=-self.truncated_dist, max=self.truncated_dist)
            resized_tsdf_volume *= self.sdf_delta_weight
            new_sdf_delta.append(resized_tsdf_volume.to("cuda"))
            self.sdf_delta = new_sdf_delta

        self.volume.fine_volume.features = nn.Parameter(self.volume.fine_volume.features)

        self.automatic_optimization = False

    def render_with_rays(
        self,
        frame,
        volume,
        rays,
        weight_mask,
        sdf_delta,
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
            self.truncated_units*2,
            int(self.ray_max_dist*5), gt_depths, rays['gt_pts'],
            ray_dirs, cam_loc, offset_distance=self.truncated_dist,
            max_depth=self.ray_max_dist
        )
        assert weight_mask is None
        pred_sdf = volume.decode_pts(
            pts, self.nerf, sdf_delta=sdf_delta)
        pred_sdf = pred_sdf[..., 0]
        out = {
            "cam_loc": cam_loc,
            "ray_dirs": ray_dirs,
            "sdf_on_rays": pred_sdf,
            "pts_on_rays": pts,
        }
        return out

    def compute_sdf_loss(self, rays, pred_sdf, pred_pts, cam_loc, num_valid_pixels):
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
            gt_sdf, min=-self.truncated_dist, max=self.truncated_dist)
        valid_map = gt_sdf > max(-self.truncated_dist*0.5, -0.05)
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
            gt_nearest_signed_dists, min=-self.truncated_dist, max=self.truncated_dist)
        depth_bce = F.l1_loss(
            pred_sdf,
            gt_nearest_signed_dists,
            reduction='none'
        ) * valid_map
        depth_bce = (depth_bce * rays['mask'].unsqueeze(-1)).sum() / num_valid_pixels
        return depth_bce

    def calculate_loss(
        self,
        frame,
        volume,
        rays,
        weight_mask=None,
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
        loss_output = {}
        render_out = self.render_with_rays(
            frame,
            volume,
            rays,
            weight_mask,
            sdf_delta,
        )
        sdf_loss = self.compute_sdf_loss(
            rays,
            render_out['sdf_on_rays'],
            render_out['pts_on_rays'],
            render_out['cam_loc'],
            num_valid_pixels
        )
        loss_output['depth_bce_loss'] = sdf_loss
        return loss_output

    def decode_feature_grid_sparse(self, volume, sdf_delta, volume_resolution):
        surface_pts, mesh = volume.meshlize(self.nerf, sdf_delta, volume_resolution)
        return mesh

    def forward(self, frame, rays, volume, backward):
        batch_loss = {}
        n_rays = rays['uv'].shape[1]
        batch_loss = {}
        for i, indx in enumerate(torch.split(
            torch.arange(n_rays).cuda(), self.train_ray_splits, dim=0
        )):
            ray_splits = {
                "uv": torch.index_select(rays['uv'], 1, indx),
                "rgb": torch.index_select(rays['rgb'], 1, indx),
                "gt_pts": torch.index_select(rays['gt_pts'], 1, indx),
                "mask": torch.index_select(rays['mask'], 1, indx),
                "neighbor_pts": torch.index_select(rays['neighbor_pts'], 1, indx),
                "neighbor_masks": torch.index_select(rays['neighbor_masks'], 1, indx),
                "T_wc": rays['T_wc'],
                "intr_mat": rays['intr_mat'],
            }
            split_loss_out = \
                self.calculate_loss(
                    frame,
                    volume,
                    ray_splits,
                    sdf_delta=self.sdf_delta,
                )
            loss_for_backward = 0
            for k in split_loss_out:
                if k[0] != "_":
                    try:
                        weight = getattr(self.loss_weight, k)
                    except AttributeError:
                        log.info("[warning]: can't find loss weight")
                    loss_for_backward += split_loss_out[k] * weight
                    if k not in batch_loss:
                        batch_loss[k] = split_loss_out[k]
                    else:
                        batch_loss[k] += split_loss_out[k]
            if backward:
                self.manual_backward(loss_for_backward)
        return batch_loss

    def training_step(self, data, batch_idx):
        """
        data:
            frame: new frame information
            volume: world feature volume
            rays: rays for supervision
        """
        # setup manual optim.
        opt = self.optimizers()
        opt.zero_grad()

        frame, rays = data

        if torch.any(torch.isnan(frame['T_wc'])):
            return None
        for k in frame.keys():
            if isinstance(frame[k], torch.Tensor):
                frame[k] = frame[k].float()
        for k in rays.keys():
            if isinstance(rays[k], torch.Tensor):
                rays[k] = rays[k].float()

        batch_loss = self.forward(
            frame, rays, self.volume, backward=True)
        for k in batch_loss:
            self.log(f"train/{k}", batch_loss[k])
        opt.step()
        return frame

    def validation_step(self, data, batch_idx):
        frame, rays = data
        for k in frame.keys():
            if isinstance(frame[k], torch.Tensor):
                frame[k] = frame[k].float()
        for k in rays.keys():
            if isinstance(rays[k], torch.Tensor):
                rays[k] = rays[k].float()

        if batch_idx != 0:
            # return frame
            return None

        scene_id = frame['scene_id'][0]
        frame_id = frame['frame_id'][0]
        batch_loss = self.forward(
            frame, rays, self.volume, backward=False)

        val_loss = 0
        for k in batch_loss:
            val_loss += batch_loss[k]
            self.log(f"val/{k}", batch_loss[k])
        self.log("val_loss", val_loss)
        mesh = self.decode_feature_grid_sparse(
            self.volume,
            self.sdf_delta,
            frame['world_volume_resolution'].float()
        )
        scene_id = scene_id.split("/")[-1]
        mesh_out_path = os.path.join(self.plots_dir, f"{scene_id}_{self.current_epoch}.ply")

        if mesh is not None:
            mesh.export(mesh_out_path)

    def configure_optimizers(self):
        from src.utils.import_utils import import_from

        optimizers = []
        if not self.cfg.model.freeze_pretrained_weights:
            parameters = self.parameters()
            optimizer = import_from(
                module=self.cfg.optimizer._target_package_,
                name=self.cfg.optimizer._class_
            )(parameters, lr=self.cfg.optimizer.lr.initial)
            optimizers.append(optimizer)
        else:
            optimizer = import_from(
                module=self.cfg.optimizer._target_package_,
                name=self.cfg.optimizer._class_
            )(
                [
                    # {"params": [self.sdf_delta], "lr": self.cfg.optimizer.lr.initial * 0.01},
                    {"params": [self.volume.fine_volume.features]},
                ],
                lr=self.cfg.optimizer.lr.initial
            )
            optimizers.append(optimizer)
        return optimizers
