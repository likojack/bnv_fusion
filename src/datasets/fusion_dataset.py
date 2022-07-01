import cv2
import os
import torch
import trimesh
import numpy as np
import pickle
from scipy.ndimage.morphology import binary_dilation
from kornia.geometry.depth import depth_to_normals, depth_to_3d

from src.utils.common import load_rgb, load_depth
from src.datasets import register
from src.datasets.sampler import SampleManager
import src.utils.geometry as geometry
import src.utils.voxel_utils as voxel_utils
from src.models.sparse_volume import VolumeList
import src.utils.scannet_helper as scannet_helper


class FusionBaseDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, stage):
        self.cfg = cfg
        self.downsample_scale = cfg.dataset.downsample_scale
        img_res = cfg.dataset.img_res
        img_res = [int(f * self.downsample_scale) for f in img_res]
        self.img_res = img_res
        self.total_pixels = img_res[0] * img_res[1]
        self.num_pixels = cfg.dataset.num_pixels
        self.data_root_dir = os.path.join(
            cfg.dataset.data_dir,
            cfg.dataset.subdomain,
        )

        self.num_images = cfg.dataset.num_images
        self.skip = cfg.dataset.skip_images
        self.shift = cfg.dataset.sample_shift
        self.num_images = int(self.num_images / self.skip)
        self.feat_dim = cfg.model.feature_vector_size
        self.stage = stage

        assert os.path.exists(self.data_root_dir), "Data directory is empty"

    def read_pose(self, path):
        with open(path, "r") as f:
            line = f.read().splitlines()[0].split(" ")
            pose = np.asarray([float(t) for t in line])
            nrow = int(np.sqrt(len(pose)))
        return pose.reshape(nrow, nrow).astype(np.float32)

    def __len__(self):
        if self.stage == "val":
            return len(self.scene_images[self.scene_list[0]])
        else:
            return len(self.views)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

@register("fusion_dataset")
class FusionDataset(FusionBaseDataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self, cfg, stage):
        super().__init__(cfg, stage)

        if stage == "test":
            scan_id = cfg.dataset.scan_id
            img_ids = np.arange(0, cfg.dataset.num_images, self.skip)
            self.num_images = len(img_ids)
            self.max_depth = cfg.model.ray_tracer.ray_max_dist
            self.voxel_size = cfg.model.voxel_size
            self.views = []
            for img_id in img_ids:
                line = f"{scan_id} {img_id}"
                self.views.append(line)
        else:
            with open(os.path.join(self.data_root_dir, f"{stage}.txt"), "r") as f:
                self.views = [s for s in f.read().splitlines()]
            self.views = [self.views[v] for v in range(0, len(self.views), self.skip)]
        self.max_depth = cfg.model.ray_tracer.ray_max_dist
        self.voxel_size = cfg.model.voxel_size
        self.scene_list = np.unique([v.split(" ")[0] for v in self.views])
        self.stage = stage
        scene_images = {k: [] for k in self.scene_list}
        for v in self.views:
            scene, frame = v.split(" ")
            scene_images[scene].append(frame)
        self.scene_images = scene_images

        self.sampling_idx = torch.arange(0, self.num_pixels)
        if stage == "val":
            self.val_seq = self.scene_list[0]
        if stage == "test":
            self.init_volumes()

    def init_volumes(self):
        self.volume_list = {}
        for scene in self.scene_list:
            dimension_path = os.path.join(
                self.data_root_dir, scene, "pose", "dimensions.txt"
            )
            with open(dimension_path, "r") as f:
                line = f.read().splitlines()[0].split(" ")
                dimensions = np.asarray([float(l) for l in line])
            _, _, volume_resolution = voxel_utils.get_world_range(
                dimensions, self.voxel_size
            )
            volume_resolution = volume_resolution.astype(np.int32)
            self.volume_list[scene] = VolumeList(
                self.feat_dim,
                self.voxel_size,
                dimensions,
                self.cfg.model.min_pts_in_grid
            )
        assert len(self.volume_list.keys()) < 10, "memory will explore if handling too many scenes"

    def reset_volumes(self):
        for scene in self.scene_list:
            self.volume_list[scene].reset()

    def select_val_seq(self):
        rand_id = torch.randperm(len(self.scene_list))[0].item()
        self.val_seq = self.scene_list[rand_id]

    def get_point_cloud_volume(self, pc):
        min_coords = np.min(pc, axis=0) - self.voxel_size
        max_coords = np.max(pc, axis=0) + self.voxel_size
        volume_resolution = (max_coords - min_coords) / self.voxel_size
        volume_resolution = np.ceil(volume_resolution)
        return min_coords, max_coords, volume_resolution

    def __getitem__(self, idx):
        self.change_sampling_idx(self.num_pixels)
        if self.stage == "val":
            scene = self.val_seq
            frame_id = self.scene_images[scene][idx]
            idx_in_seq = idx
        else:
            seq_id = idx // self.num_images
            idx_in_seq = idx - (seq_id * self.num_images)
            scene = self.scene_list[seq_id]
            frame_id = self.scene_images[scene][idx_in_seq]

        dimension_path = os.path.join(
            self.data_root_dir, scene, "pose", "dimensions.txt"
        )
        with open(dimension_path, "r") as f:
            line = f.read().splitlines()[0].split(" ")
            dimensions = np.asarray([float(l) for l in line])
        mul_factor = int((torch.rand(1) * 2).item()) + 1
        if self.stage in ["train", "val"]:
            img_ids = np.arange(self.max_neighbor_imgs+1) - np.floor(self.max_neighbor_imgs / 2)
            img_ids = img_ids * mul_factor
            img_ids += idx_in_seq
            img_ids = np.clip(img_ids, a_min=0, a_max=self.num_images-1)
            img_ids = img_ids.astype(np.int32)
            img_ids = [self.scene_images[scene][t] for t in img_ids]
        else:
            img_ids = [frame_id]
        T_wc_list = []
        intr_mat_list = []
        rgbd_list = []
        uv_list = []
        rgb_list = []
        gt_pts_list = []
        gt_pts_frame_list = []
        mask_list = []
        frame_mask_list = []
        dist_start_frame_list = []
        dist_end_frame_list = []
        dist_start_list = []
        dist_end_list = []
        clean_depth_list = []
        routing_mask_list = []
        gradient_mask_list = []
        # assert len(img_ids) == 1, "single-view render should have batch size 1"
        world_min_coords, world_max_coords, world_volume_resolution = \
            voxel_utils.get_world_range(
                dimensions, self.voxel_size
            )
        old_sdfs = np.zeros(world_volume_resolution.astype(np.int32))
        old_weights = np.zeros(world_volume_resolution.astype(np.int32))

        for i, img_id in enumerate(img_ids):
            uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
            uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
            uv = uv.reshape(2, -1).transpose(1, 0)
            image_path = os.path.join(
                self.data_root_dir, scene, "image", f"{img_id}.jpg"
            )
            depth_path = os.path.join(
                self.data_root_dir, scene, "depth", f"{img_id}.png"
            )
            mask_path = os.path.join(
                self.data_root_dir, scene, "mask", f"{img_id}.png"
            )
            T_wc_path = os.path.join(
                self.data_root_dir, scene, "pose", f"T_wc_{img_id}.txt"
            )
            intr_mat_path = os.path.join(
                self.data_root_dir, scene, "pose", f"intr_mat_{img_id}.txt"
            )
            T_wc = self.read_pose(T_wc_path)
            intr_mat = self.read_pose(intr_mat_path)
            rgb = load_rgb(image_path)
            add_noise = True if self.stage != "test" else False
            clean_depth, noise_depth, frame_mask = load_depth(
                depth_path, 0, max_depth=self.max_depth,
                add_noise=add_noise
            )
            gradient_mask = binary_dilation(frame_mask, iterations=5)
            routing_mask = binary_dilation(frame_mask, iterations=8)
            img_h, img_w = clean_depth.shape
            rgbd = np.concatenate([rgb, noise_depth[None, ...]], axis=0)
            pts_c = geometry.depth2xyz(clean_depth, intr_mat).reshape(-1, 3)
            pts_w_frame = (T_wc @ geometry.get_homogeneous(pts_c).T)[:3, :].T
            valid_pts = pts_w_frame[frame_mask.reshape(-1)]
            gt_pts = pts_w_frame[self.sampling_idx, :]
            rgb = rgb.reshape(3, -1).transpose(1, 0)[self.sampling_idx, :]
            uv = uv[self.sampling_idx, :]
            mask = frame_mask.reshape(-1)[self.sampling_idx]
            cam_min_coords, cam_max_coords, cam_volume_resolution = \
                voxel_utils.get_frustrum_range(
                    intr_mat, img_h, img_w,
                    max_depth=self.max_depth,
                    voxel_size=self.voxel_size
                )
            frame_mask_list.append(frame_mask)
            T_wc_list.append(T_wc)
            intr_mat_list.append(intr_mat)
            rgbd_list.append(rgbd)
            uv_list.append(uv)
            rgb_list.append(rgb)
            gt_pts_list.append(gt_pts)
            mask_list.append(mask)
            gt_pts_frame_list.append(pts_w_frame)
            clean_depth_list.append(clean_depth)
            routing_mask_list.append(routing_mask)
            gradient_mask_list.append(gradient_mask)
            sdfs, weights = voxel_utils.depth_to_tsdf(
                noise_depth, T_wc, intr_mat[:3, :3], world_min_coords,
                world_max_coords, world_volume_resolution, self.voxel_size,
                device=self.cfg.device_type,
            )
            old_sdfs += sdfs
            old_weights += weights
        x, y, z = np.nonzero(old_weights == 0)
        sdfs = old_sdfs / np.clip(old_weights, a_min=1, a_max=len(img_ids))
        sdfs[x, y, z] = 5 * self.voxel_size

        T_wc = np.stack(T_wc_list, axis=0).astype(np.float32)
        intr_mat = np.stack(intr_mat_list, axis=0).astype(np.float32)
        rgbd = np.stack(rgbd_list, axis=0).astype(np.float32)
        uv = torch.stack(uv_list, dim=0)
        rgb = np.stack(rgb_list, axis=0).astype(np.float32)
        gt_pts = np.stack(gt_pts_list, axis=0).astype(np.float32)
        frame_mask = np.stack(frame_mask_list, axis=0).astype(np.float32)
        mask = np.stack(mask_list, axis=0).astype(np.float32)
        gt_pts_frame = np.stack(gt_pts_frame_list, axis=0).astype(np.float32)
        clean_depth = np.stack(clean_depth_list, axis=0).astype(np.float32)
        routing_mask = np.stack(routing_mask_list, axis=0).astype(np.float32)
        gradient_mask = np.stack(gradient_mask_list, axis=0).astype(np.float32)

        # start_dists_frame = np.stack(dist_start_frame_list, axis=0)
        # end_dists_frame = np.stack(dist_end_frame_list, axis=0)
        # start_dists = np.stack(dist_start_list, axis=0).astype(np.float32)
        # end_dists = np.stack(dist_end_list, axis=0).astype(np.float32)
        frame = {
            "scene_id": scene,
            "frame_id": frame_id,
            "T_wc": T_wc,
            "intr_mat": intr_mat,
            "rgbd": rgbd,
            "cam_min_coords": cam_min_coords,
            "cam_max_coords": cam_max_coords,
            "cam_volume_resolution": cam_volume_resolution,
            "world_min_coords": world_min_coords,
            "world_max_coords": world_max_coords,
            "world_volume_resolution": world_volume_resolution,
            "mask": frame_mask,
            "sdfs": sdfs,
            "sdf_weights": old_weights,
            "gt_pts": gt_pts_frame,
            "gt_depth": clean_depth,
            "routing_mask": routing_mask,
            "gradient_mask": gradient_mask
            # "start_dists": start_dists_frame,
            # "end_dists": end_dists_frame
        }
        rays = {
            "uv": uv,
            "rgb": rgb,
            "gt_pts": gt_pts,
            "intr_mat": intr_mat,
            "T_wc": T_wc,
            "mask": mask,
        }
        return frame, rays


class FusionRefinerAbstractDataset(FusionBaseDataset):
    def __init__(self, cfg, stage):
        super().__init__(cfg, stage)
        self.scan_id = cfg.dataset.scan_id
        self.voxel_size = cfg.model.voxel_size
        self.max_depth = cfg.model.ray_tracer.ray_max_dist
        self.depth_scale = -1
        self.image_paths = []
        self.depth_paths = []
        self.T_wc_paths = []
        self.intr_mat_paths = []

    def __len__(self):
        return len(self.image_paths)

    def read_extr_pose(self):
        raise NotImplementedError

    def read_intr_pose(self):
        raise NotImplementedError

    def load_rgb(self, path, downsample_scale):
        rgb = cv2.imread(path, -1)[:, :, ::-1]
        img_h, img_w, _ = rgb.shape
        if downsample_scale > 0:
            reduced_w = int(img_w * downsample_scale)
            reduced_h = int(img_h * downsample_scale)
            rgb = cv2.resize(rgb, dsize=(reduced_w, reduced_h))
        rgb = rgb.transpose(2, 0, 1)
        return rgb

    def load_depth(self, depth_path, depth_scale, max_depth, downsample_scale):
        depth = cv2.imread(depth_path, -1)
        depth = depth / depth_scale
        if downsample_scale > 0:
            img_h, img_w = depth.shape
            reduced_w = int(img_w * downsample_scale)
            reduced_h = int(img_h * downsample_scale)
            depth = cv2.resize(
                depth,
                dsize=(reduced_w, reduced_h),
                interpolation=cv2.INTER_NEAREST
            )
        mask = np.logical_and(depth > 0, depth < max_depth)
        return depth, mask

    def get_neighbor_xyz(self, xyz_map, mask, uv, kernel_size=15):
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
        out_mask = out_mask.astype(np.bool)
        return out, out_mask

    def __getitem__(self, idx):
        self.change_sampling_idx(self.num_pixels)
        world_min_coords, world_max_coords, world_volume_resolution = \
            voxel_utils.get_world_range(
                self.dimensions, self.voxel_size
            )

        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)
        uv = uv[self.sampling_idx, :]

        T_wc_path = self.T_wc_paths[idx]
        T_wc = self.read_extr_pose(T_wc_path)
        intr_mat_path = self.intr_mat_paths[idx]
        intr_mat = self.read_intr_pose(intr_mat_path)[:3, :3]
        intr_mat[2][2] = 1
        intr_mat[:2, :3] = intr_mat[:2, :3] * self.downsample_scale

        image_path = self.image_paths[idx]
        rgb = self.load_rgb(image_path, self.downsample_scale)

        depth_path = self.depth_paths[idx]
        assert self.depth_scale != -1, "depth scale should be overwritten"
        depth, frame_mask = self.load_depth(
            depth_path, self.depth_scale,
            self.max_depth, self.downsample_scale
        )
        normal = depth_to_normals(
            torch.from_numpy(depth).unsqueeze(0).unsqueeze(0),
            torch.from_numpy(intr_mat).unsqueeze(0)
        )[0].permute(1, 2, 0).numpy()
        normal_w = (T_wc[:3, :3] @ normal.reshape(-1, 3).T).T
        pts_c = geometry.depth2xyz(depth, intr_mat).reshape(-1, 3)
        pts_w_frame = (T_wc @ geometry.get_homogeneous(pts_c).T)[:3, :].T
        input_pts = np.concatenate(
            [pts_w_frame, normal_w],
            axis=-1
        )
        input_pts = input_pts[frame_mask.reshape(-1)]

        rgbd = np.concatenate([rgb, depth[None, ...]], axis=0)

        gt_pts = pts_w_frame[self.sampling_idx, :]
        rgb = rgb.reshape(3, -1).transpose(1, 0)[self.sampling_idx, :]
        mask = frame_mask.reshape(-1)[self.sampling_idx]

        img_h, img_w = depth.shape
        gt_xyz_map_w = pts_w_frame.reshape(img_h, img_w, 3)
        neighbor_pts, neighbor_mask = self.get_neighbor_xyz(
            gt_xyz_map_w, frame_mask, uv, 15)

        frame = {
            "scene_id": self.scan_id,
            "frame_id": idx,
            "input_pts": input_pts,
            "T_wc": T_wc,
            "intr_mat": intr_mat,
            "rgbd": rgbd,
            "world_min_coords": world_min_coords,
            "world_max_coords": world_max_coords,
            "world_volume_resolution": np.asarray(world_volume_resolution),
            "mask": frame_mask,
            "gt_pts": pts_w_frame,
        }
        rays = {
            "uv": uv,
            "rgb": rgb,
            "gt_pts": gt_pts,
            "intr_mat": intr_mat,
            "T_wc": T_wc,
            "mask": mask,
            "neighbor_pts": neighbor_pts,
            "neighbor_masks": neighbor_mask
        }
        return frame, rays

@register("fusion_refiner_dataset")
class FusionRefinerDataset(FusionRefinerAbstractDataset):
    def __init__(self, cfg, stage):
        super().__init__(cfg, stage)

        img_dir = os.path.join(self.data_root_dir, self.scan_id, "image")
        num_images = len(os.listdir(img_dir))
        if stage == "train":
            img_ids = np.arange(self.shift, num_images, self.skip)
        else:
            img_ids = np.arange(self.shift, num_images, self.skip)[:2]
        self.num_images = len(img_ids)
        self.depth_scale = self.cfg.dataset.depth_scale
        dimension_path = os.path.join(
            self.data_root_dir, self.scan_id, "pose", "dimensions.txt"
        )
        with open(dimension_path, "r") as f:
            line = f.read().splitlines()[0].split(" ")
            dimensions = np.asarray([float(l) for l in line])
        self.dimensions = dimensions
        for img_id in img_ids:
            self.image_paths.append(
                os.path.join(
                    self.data_root_dir, self.scan_id, "image", f"{img_id}.jpg"
                )
            )
            self.depth_paths.append(
                os.path.join(
                    self.data_root_dir, self.scan_id, "depth", f"{img_id}.png"
                )
            )
            self.T_wc_paths.append(
                os.path.join(
                    self.data_root_dir, self.scan_id, "pose", f"T_wc_{img_id}.txt"
                )
            )
            self.intr_mat_paths.append(
                os.path.join(
                    self.data_root_dir, self.scan_id, "pose", f"intr_mat_{img_id}.txt"
                )
            )
        self.sampling_idx = torch.arange(0, self.num_pixels)

    def read_pose(self, path):
        with open(path, "r") as f:
            line = f.read().splitlines()[0].split(" ")
            pose = np.asarray([float(t) for t in line])
            nrow = int(np.sqrt(len(pose)))
        return pose.reshape(nrow, nrow).astype(np.float32)

    def read_extr_pose(self, path):
        return self.read_pose(path)

    def read_intr_pose(self, path):
        return self.read_pose(path)


@register("fusion_refiner_scannet_dataset")
class FusionRefinerScanNetDataset(FusionRefinerAbstractDataset):
    def __init__(self, cfg, stage):
        super().__init__(cfg, stage)
        img_dir = os.path.join(self.data_root_dir, self.scan_id, "frames", "color")
        num_images = len(os.listdir(img_dir))
        img_ids = np.arange(0, num_images, self.skip)
        self.img_h = self.img_res[0]
        self.img_w = self.img_res[1]
        self.num_images = len(img_ids)
        self.depth_scale = self.cfg.dataset.depth_scale
        axis_align_mat = scannet_helper.read_meta_file(
            os.path.join(
                self.data_root_dir, self.scan_id, f"{self.scan_id}.txt")
        )
        mesh_path = os.path.join(self.data_root_dir, self.scan_id, f"{self.scan_id}_vh_clean_2.ply")
        mesh = trimesh.load(mesh_path)
        mesh.vertices = (axis_align_mat @ geometry.get_homogeneous(mesh.vertices).T)[:3, :].T
        max_pts = np.max(mesh.vertices, axis=0)
        min_pts = np.min(mesh.vertices, axis=0)
        center = (min_pts + max_pts) / 2
        self.dimensions = np.asarray(max_pts - min_pts)
        recenter_mat = np.eye(4)
        recenter_mat[:3, 3] = -center
        self.axis_align_mat = recenter_mat @ axis_align_mat

        for img_id in img_ids:
            self.image_paths.append(
                os.path.join(
                    self.data_root_dir, self.scan_id, "frames", "color", f"{img_id}.jpg"
                )
            )
            self.depth_paths.append(
                os.path.join(
                    self.data_root_dir, self.scan_id, "frames", "depth", f"{img_id}.png"
                )
            )
            self.T_wc_paths.append(
                os.path.join(
                    self.data_root_dir, self.scan_id, "frames", "pose", f"{img_id}.txt"
                )
            )
            self.intr_mat_paths.append(
                os.path.join(
                    self.data_root_dir, self.scan_id, "frames", "intrinsic", "intrinsic_depth.txt"
                )
            )
        self.sampling_idx = torch.arange(0, self.num_pixels)

    def load_rgb(self, path, downsample_scale=None):
        rgb = cv2.imread(path, -1)[:, :, ::-1]
        rgb = cv2.resize(rgb, (self.img_w, self.img_h))
        rgb = rgb.transpose(2, 0, 1)
        return rgb

    def read_extr_pose(self, path):
        T_cw = scannet_helper.read_extrinsic(path)
        T_wc = np.linalg.inv(T_cw)
        T_wc = self.axis_align_mat @ T_wc
        return T_wc

    def read_intr_pose(self, path):
        return scannet_helper.read_intrinsic(path)[:3, :3]


if __name__ == "__main__":
    from easydict import EasyDict
    from torch.utils.data import DataLoader
    import open3d as o3d
    import src.utils.o3d_helper as o3d_helper

    cfg = EasyDict({
        "dataset": {
            "img_res": [240, 320],
            "num_pixels": 10240,
            "data_dir": "./data",
            "subdomain": "fusion"
        },
        "model": {
            "feature_vector_size": 32,
            "feature_resolution": 128,
        }
    })
    dataset = FusionDataset(cfg, "train")
    loader = DataLoader(dataset, batch_size=1)
    for data in loader:
        frame, volume, rays = data
        rgb = rays['rgb'][0]
        gt_pts = rays['gt_pts'][0]
        pts = o3d_helper.np2pc(
            gt_pts.numpy().reshape(-1, 3),
            rgb.numpy().reshape(-1, 3)/2 + 0.5)
        o3d.visualization.draw_geometries([pts])