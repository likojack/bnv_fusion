from weakref import KeyedRef
import hydra
import numpy as np
import os
from omegaconf import DictConfig
import torch
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch_lightning import seed_everything

from src.datasets import datasets
from src.datasets.fusion_inference_dataset import IterableInferenceDataset
import src.utils.o3d_helper as o3d_helper
import src.utils.hydra_utils as hydra_utils
import src.utils.voxel_utils as voxel_utils
from src.models.fusion.local_point_fusion import LitFusionPointNet
from src.models.sparse_volume import SparseVolume
from src.utils.render_utils import calculate_loss
from src.utils.common import to_cuda, Timer
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
        if "/" in config.dataset.scan_id:
            self.dataset_name, self.scan_id = config.dataset.scan_id.split("/")
        else:
            self.scan_id = config.dataset.scan_id
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
        
        self.frames = []
        self.iterable_dataset = IterableInferenceDataset(
            self.frames, self.ray_max_dist, self.bound_min.cpu(),
            self.bound_max.cpu(), self.n_xyz, self.sampling_size)

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
            if fine_feats is None:
                return None
            self.volume.track_n_pts(fine_n_pts)
            self.pointnet._integrate(
                self.volume,
                fine_coords,
                fine_feats,
                fine_weights)
            # tsdf fusion
            rgbd = frame['rgbd'].cpu().numpy()
            depth_map = rgbd[0, -1, :, :]
            rgb = (rgbd[0, :3, :, :].transpose(1, 2, 0) + 0.5) * 255.
            # depth_map = rgbd[0, 3, :, :]
            self.tsdf_vol.integrate(
                rgb,  # [h, w, 3], [0, 255]
                depth_map,  # [h, w], metric depth
                frame['intr_mat'].cpu().numpy()[0],
                frame["T_wc"].cpu().numpy()[0],
                obs_weight=1.)

    def optimize(self, n_iters, last_frame):
        self.volume.to_tensor()
        tsdf_delta = self.prepare_tsdf_volume()
        self.volume.features = torch.nn.Parameter(self.volume.features)
        self.iterable_dataset.n_iters = n_iters
        self.iterable_dataset.last_frame = last_frame
        loader = torch.utils.data.DataLoader(self.iterable_dataset, batch_size=None, num_workers=4)
        optimizer = torch.optim.Adam([self.volume.features], lr=0.001)
        for rays in tqdm(loader):
            optimizer.zero_grad()
            if torch.isnan(rays['T_wc']).any():
                continue
            to_cuda(rays)
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
        resized_tsdf_volume = tsdf_volume
        # resized_tsdf_volume = F.interpolate(
        #     tsdf_volume,
        #     size=(
        #         self.n_xyz[0],
        #         self.n_xyz[1],
        #         self.n_xyz[2]
        #     ),
        #     mode="trilinear",
        #     align_corners=True)
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
        self.volume.save(os.path.join(self.working_dir, "final"))
        
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

    plots_dir = os.path.join(os.getcwd(), config.dataset.scan_id)
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # setup model
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
    timer = Timer(["local", "global"])
    for idx, data in enumerate(tqdm(val_loader)):
        # LOCAL FUSION:
        # integrate information from the new frame to the feature volume
        frame, _ = data
        for k in frame.keys():
            if isinstance(frame[k], torch.Tensor):
                frame[k] = frame[k].cuda().float()
        timer.start("local")
        neural_map.integrate(frame)
        timer.log("local")
        if torch.isnan(frame['T_wc']).any():
            continue
        meta_frame = {
            "frame_id": frame["frame_id"],
            "scan_id": frame["scene_id"],
            "T_wc": frame["T_wc"].clone().cpu(),
            "intr_mat": frame["intr_mat"].clone().cpu(),
            "img_path": frame['img_path'][0],
            "depth_path": frame['depth_path'][0],
        }
        del frame
        neural_map.frames.append(meta_frame)
        # clear memory for open3d hashmap
        if (idx+1) % 2 == 0:
            torch.cuda.empty_cache()
        if config.model.mode == "demo":
            if (idx) % config.model.optim_interval == 0:
                last_frame = max(0, len(neural_map.frames) - config.model.optim_interval)
                n_iters = min(len(neural_map.frames), config.model.optim_interval) * neural_map.skip_images
                timer.start("global")
                neural_map.optimize(n_iters=n_iters, last_frame=last_frame)
                timer.log("global")
                mesh = neural_map.extract_mesh()
                mesh = o3d_helper.post_process_mesh(mesh)
                mesh_out_path = os.path.join(neural_map.working_dir, f"{idx}.ply")
                mesh.export(mesh_out_path)
    neural_map.volume.to_tensor()
    mesh = neural_map.extract_mesh()
    mesh.export(os.path.join(neural_map.working_dir, "before_optim.ply"))
    global_steps = int(len(neural_map.frames) * neural_map.skip_images)
    global_steps = global_steps * 2 if config.model.mode != "demo" else global_steps
    timer.start("global")
    neural_map.optimize(n_iters=global_steps, last_frame=-1)
    timer.log("global")
    for n in ["local", "global"]:
        print(f"speed on {n} fusion: {global_steps / timer.times[n]} fps")
    
    mesh = neural_map.extract_mesh()
    mesh = o3d_helper.post_process_mesh(mesh, vertex_threshold=neural_map.voxel_size / 4)
    mesh_out_path = os.path.join(neural_map.working_dir, "final.ply")
    mesh.export(mesh_out_path)
    neural_map.save()


if __name__ == "__main__":
    main()