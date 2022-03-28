import cv2
import hydra
import numpy as np
import os
from omegaconf import DictConfig
import open3d as o3d
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor

from src.datasets import datasets
import src.utils.geometry as geometry
import src.utils.hydra_utils as hydra_utils
import src.utils.voxel_utils as voxel_utils
from src.models.fusion.local_point_fusion import LitFusionPointNet
from src.models.sparse_volume import SparseVolume
from src.utils.render_utils import calculate_loss
from src.utils.common import load_depth
import third_parties.fusion as fusion


log = hydra_utils.get_logger(__name__)


class NeuralMap:
    def __init__(
        self,
        dimensions,
        config,
        pointnet,
        working_dir,
    ):
        self.dataset_name, self.scan_id = config.dataset.scan_id.split("/")
        min_coords, max_coords, n_xyz = voxel_utils.get_world_range(
            dimensions, config.model.voxel_size)
        self.pointnet = pointnet
        self.working_dir = working_dir
        self.config = config
        self.volume = SparseVolume(
            config.model.feature_vector_size,
            config.model.voxel_size,
            dimensions,
            config.model.min_pts_in_grid)
        self.bound_min = torch.from_numpy(min_coords).to("cuda").float()
        self.bound_max = torch.from_numpy(max_coords).to("cuda").float()
        self.voxel_size = config.model.voxel_size
        self.n_xyz = n_xyz
        self.dimensions = dimensions
        self.sampling_size = config.dataset.num_pixels
        self.train_ray_splits = config.model.train_ray_splits
        self.ray_max_dist = config.model.ray_tracer.ray_max_dist
        self.truncated_units = config.model.ray_tracer.truncated_units
        self.truncated_dist = min(self.truncated_units * self.voxel_size * 0.5, 0.1)
        self.depth_scale = 1000.
        self.sdf_delta = None
        self.skip_images = config.dataset.skip_images
        self.tsdf_voxel_size = 0.025
        self.sdf_delta_weight = config.model.sdf_delta_weight
        min_coords, max_coords, n_xyz = voxel_utils.get_world_range(
            dimensions, self.tsdf_voxel_size)
        vol_bnds = np.zeros((3,2))
        vol_bnds[:, 0] = min_coords
        vol_bnds[:, 1] = max_coords
        self.tsdf_vol = fusion.TSDFVolume(
            vol_bnds,
            voxel_size=self.tsdf_voxel_size)

    def _get_neighbor_xyz(self, xyz_map, mask, uv, kernel_size=15):
        uv = uv.numpy().astype(np.int32)
        mask = mask.astype(np.float32)
        n_pts = len(uv)
        half_length = np.floor(kernel_size / 2).astype(np.int32)
        img_h, img_w = xyz_map.shape[:2]
        out = np.zeros((n_pts, int(kernel_size ** 2), 3)) - 1000
        out_mask = np.zeros((n_pts, int(kernel_size ** 2))) - 1
        range_ = np.arange(-half_length, half_length+1)
        indices = np.stack(np.meshgrid(range_, range_), axis=-1).reshape(-1, 2)
        indices = np.tile(indices, (n_pts, 1, 1))
        indices += uv[:, None, :]
        indices[:, :, 0] = np.clip(indices[:, :, 0], a_min=0, a_max=img_w-1)
        indices[:, :, 1] = np.clip(indices[:, :, 1], a_min=0, a_max=img_h-1)

        # xyz: [n_pts * kernel_size ** 2, 3]
        xyz = xyz_map[indices[:, :, 1].reshape(-1), indices[:, :, 0].reshape(-1), :3]
        out = xyz.reshape(n_pts, int(kernel_size ** 2), 3)
        # mask: [n_pts * kernel_size ** 2]
        mask = mask[indices[:, :, 1].reshape(-1), indices[:, :, 0].reshape(-1)]
        out_mask = mask.reshape(n_pts, int(kernel_size ** 2))

        assert (out_mask != -1).all()
        assert (out != -1000).all()
        out_mask = out_mask.astype(bool)
        return out, out_mask

    def _load_rgb(self, path):
        rgb = cv2.imread(path, -1)[:, :, ::-1].transpose(2, 0, 1)
        return rgb

    def _sample_key_frame(self, keyframes):
        n_frames = len(keyframes)
        img_id = torch.randint(0, n_frames, size=(1,))
        meta_frame = keyframes[img_id]
        rgb = self._load_rgb(meta_frame['img_path'])
        depth, _, frame_mask = load_depth(
            meta_frame['depth_path'],
            downsample_scale=0,
            max_depth=self.ray_max_dist)
        rgbd = np.concatenate([rgb, depth[None, ...]], axis=0)
        pts_c = geometry.depth2xyz(
            depth,
            meta_frame['intr_mat'].cpu().numpy()[0]
        ).reshape(-1, 3)
        pts_w_frame = (meta_frame['T_wc'][0].cpu().numpy() @ geometry.get_homogeneous(pts_c).T)[:3, :].T

        new_frame = {
            "scene_id": meta_frame['scan_id'],
            "frame_id": meta_frame['frame_id'],
            # "input_pts": frame['input_pts'],
            "T_wc": meta_frame['T_wc'],
            "intr_mat": meta_frame['intr_mat'],
            "rgbd": torch.from_numpy(rgbd).to("cuda").unsqueeze(0),
            "world_min_coords": self.bound_min,
            "world_max_coords": self.bound_max,
            "world_volume_resolution": torch.from_numpy(np.asarray(self.n_xyz)).to("cuda"),
            "mask": torch.from_numpy(frame_mask).to("cuda").unsqueeze(0),
            "gt_pts": torch.from_numpy(pts_w_frame).to("cuda").unsqueeze(0),
        }

        img_res = rgbd.shape[1:]
        img_h, img_w = img_res
        total_pixels = img_res[0] * img_res[1]
        sampling_idx = torch.randperm(total_pixels)[:self.sampling_size]
        uv = np.mgrid[0:img_res[0], 0:img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)
        uv = uv[sampling_idx, :]
        rgb_ray = rgb.reshape(3, -1).transpose(1, 0)[sampling_idx, :]
        gt_pts_ray = pts_w_frame[sampling_idx, :]
        mask_ray = frame_mask.reshape(-1)[sampling_idx]
        gt_xyz_map_w = pts_w_frame.reshape(img_h, img_w, 3)
        neighbor_pts, neighbor_mask = self._get_neighbor_xyz(
            gt_xyz_map_w, frame_mask, uv, 15)
        rays = {
            "uv": uv.to("cuda").unsqueeze(0),
            "rgb": torch.from_numpy(rgb_ray).to("cuda").unsqueeze(0),
            "gt_pts": torch.from_numpy(gt_pts_ray).to("cuda").unsqueeze(0).float(),
            "intr_mat": meta_frame['intr_mat'],
            "T_wc": meta_frame['T_wc'],
            "mask": torch.from_numpy(mask_ray).to("cuda").unsqueeze(0).float(),
            "neighbor_pts": torch.from_numpy(neighbor_pts).to("cuda").unsqueeze(0).float(),
            "neighbor_masks": torch.from_numpy(neighbor_mask).to("cuda").unsqueeze(0).float()
        }
        return new_frame, rays

    def integrate(self, frame):
        if len(frame['input_pts']) == 0:
            return None
        with torch.no_grad():
            # local-level fusion
            fine_feats, fine_weights, _, fine_coords, fine_n_pts = self.pointnet.encode_pointcloud(
                frame['input_pts'],  # [1, N, 6]
                self.volume.n_xyz,
                self.volume.min_coords,
                self.volume.max_coords,
                self.volume.voxel_size,
                return_dense=self.pointnet.dense_volume
            )
            self.volume.track_n_pts(fine_n_pts)
            self.pointnet._integrate(
                self.volume,
                fine_coords,
                fine_feats,
                fine_weights)
            # tsdf fusion
            rgbd = frame['rgbd'].cpu().numpy()
            rgb = (rgbd[0, :3, :, :].transpose(1, 2, 0) + 0.5) * 255.
            # depth_map = rgbd[0, 3, :, :]
            depth_map, _, mask = load_depth(frame['depth_path'], downsample_scale=1, max_depth=5)
            self.tsdf_vol.integrate(
                rgb,  # [h, w, 3], [0, 255]
                depth_map,  # [h, w], metric depth
                frame['intr_mat'].cpu().numpy()[0],
                frame["T_wc"].cpu().numpy()[0],
                obs_weight=1.)

    def optimize(self, keyframes, n_iters):
        self.volume.to_tensor()
        tsdf_delta = self.prepare_tsdf_volume()
        self.volume.features = torch.nn.Parameter(self.volume.features)
        optimizer = torch.optim.Adam([self.volume.features], lr=0.001)
        for it in tqdm(range(n_iters)):
            optimizer.zero_grad()
            frame, rays = self._sample_key_frame(keyframes)
            batch_loss = {}
            n_rays = rays['uv'].shape[1]
            n_splits = n_rays / self.train_ray_splits
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
                    "intr_mat": rays['intr_mat']}
                split_loss_out = calculate_loss(
                    frame,
                    self.volume,
                    ray_splits,
                    self.pointnet.nerf,
                    truncated_units=self.truncated_units,
                    truncated_dist=self.truncated_dist,
                    ray_max_dist=self.ray_max_dist,
                    sdf_delta=tsdf_delta)
                loss_for_backward = 0
                for k in split_loss_out:
                    if k[0] != "_":
                        loss_for_backward += split_loss_out[k]
                        if k not in batch_loss:
                            batch_loss[k] = split_loss_out[k]
                        else:
                            batch_loss[k] += split_loss_out[k]    
                loss_for_backward.backward()
            optimizer.step()
        mesh = self.extract_mesh()
        mesh.export(os.path.join(self.working_dir, "after_optimize.ply"))
        # store optimized features back to the sparse_volume
        self.volume.insert(
            self.volume.active_coordinates,
            self.volume.features,
            self.volume.weights,
            self.volume.num_hits)

    def extract_mesh(self):
        sdf_delta = self.prepare_tsdf_volume()
        surface_pts, mesh = self.volume.meshlize(self.pointnet.nerf, sdf_delta)
        return mesh

    def prepare_tsdf_volume(self):
        tsdf_volume, _ = self.tsdf_vol.get_volume()
        tsdf_volume = tsdf_volume * (self.tsdf_voxel_size * 5)
        tsdf_volume = torch.from_numpy(tsdf_volume).to(self.pointnet.device).float().unsqueeze(0).unsqueeze(0)
        resized_tsdf_volume = F.interpolate(
            tsdf_volume,
            size=(
                self.n_xyz[0],
                self.n_xyz[1],
                self.n_xyz[2]
            ),
            mode="trilinear",
            align_corners=True)
        resized_tsdf_volume = torch.clip(
            resized_tsdf_volume, min=-self.truncated_dist, max=self.truncated_dist)
        resized_tsdf_volume *= self.sdf_delta_weight
        return resized_tsdf_volume

    def save(self):
        # save tsdf volume
        tsdf_out_path = os.path.join(self.working_dir, self.scan_id + ".npy")
        tsdf_vol, _ = self.tsdf_vol.get_volume()
        tsdf_vol = tsdf_vol * (self.tsdf_voxel_size * 5)
        np.save(tsdf_out_path, tsdf_vol)
        self.volume.to_tensor()
        # save volume
        volume_path = os.path.join(self.working_dir, self.scan_id)
        self.volume.save(volume_path)

def track_memory():
    div_GB = 1024 * 1024 * 1024
    print("GPU status:")
    print(f"allocated: {torch.cuda.memory_allocated() / div_GB} GB")
    print(f"max allocated: {torch.cuda.max_memory_allocated() / div_GB} GB")
    print(f"reserved: {torch.cuda.memory_reserved() / div_GB} GB")
    print(f"max reserved: {torch.cuda.max_memory_reserved() / div_GB} GB")


@hydra.main(config_path="../configs/", config_name="config.yaml")
def main(config: DictConfig):

    if "seed" in config.trainer:
        seed_everything(config.trainer.seed)

    hydra_utils.extras(config)
    hydra_utils.print_config(config, resolve=True)

    # setup dataset
    log.info("initializing dataset")
    val_dataset = datasets.get_dataset(config, "val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.dataset.eval_batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        collate_fn=val_dataset.collate_fn if hasattr(val_dataset, "collate_fn") else None
    )

    # setup model
    plots_dir = os.getcwd()
    log.info("initializing model")
    pointnet_model = LitFusionPointNet(config)
    pretrained_weights = torch.load(config.trainer.checkpoint)
    pointnet_model.load_state_dict(pretrained_weights['state_dict'])
    pointnet_model.eval()
    pointnet_model.cuda()
    pointnet_model.freeze()
    neural_map = NeuralMap(
        val_dataset.dimensions,
        config,
        pointnet_model,
        working_dir=plots_dir)

    keyframes = []
    data_root_dir = os.path.join(
        config.dataset.data_dir,
        config.dataset.subdomain,
    )
    for idx, data in enumerate(tqdm(val_loader)):
        # LOCAL FUSION:
        # integrate information from the new frame to the feature volume
        frame, _ = data
        for k in frame.keys():
            if isinstance(frame[k], torch.Tensor):
                frame[k] = frame[k].cuda().float()
        img_path = os.path.join(
            data_root_dir, frame["scene_id"][0], "image", f"{frame['frame_id'][0]}.jpg")
        depth_path = os.path.join(
            data_root_dir, frame["scene_id"][0], "depth", f"{frame['frame_id'][0]}.png")
        frame['depth_path'] = depth_path
        neural_map.integrate(frame)
        meta_frame = {
            "frame_id": frame["frame_id"],
            "scan_id": frame["scene_id"],
            "T_wc": frame["T_wc"],
            "intr_mat": frame["intr_mat"],
            "img_path": img_path,
            "depth_path": depth_path,
        }
        keyframes.append(meta_frame)
        # clear memory for open3d hashmap
        if (idx+1) % 10 == 0:
            torch.cuda.empty_cache()

    neural_map.optimize(
        keyframes,
        n_iters=int(len(keyframes) * neural_map.skip_images * 2))


if __name__ == "__main__":
    main()