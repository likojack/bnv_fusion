import cv2
import os
import torch
import trimesh
import numpy as np
from kornia.geometry.depth import depth_to_normals, depth_to_3d

from src.models.sparse_volume import SparseVolume, VolumeList
from src.datasets import register
import src.utils.geometry as geometry
import src.utils.voxel_utils as voxel_utils
from src.utils.common import load_depth, load_rgb
import src.utils.scannet_helper as scannet_helper


@register("fusion_inference_dataset")
class FusionInferenceDataset(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self, cfg, stage):
        super().__init__()
        self.cfg = cfg
        self.data_root_dir = os.path.join(
            cfg.dataset.data_dir,
            cfg.dataset.subdomain,
        )
        self.dense_volume = cfg.trainer.dense_volume
        scan_id = cfg.dataset.scan_id
        self.scan_id = scan_id
        dimension_path = os.path.join(
            self.data_root_dir, self.scan_id, "pose", "dimensions.txt"
        )
        with open(dimension_path, "r") as f:
            line = f.read().splitlines()[0].split(" ")
            dimensions = np.asarray([float(l) for l in line])
        self.dimensions = dimensions
        self.skip = cfg.dataset.skip_images
        self.shift = cfg.dataset.sample_shift
        self.downsample_scale = cfg.dataset.downsample_scale
        self.stage = stage
        self.feat_dim = cfg.model.feature_vector_size
        self.img_res = cfg.dataset.img_res
        self.total_pixels = self.img_res[0] * self.img_res[1]
        self.num_pixels = cfg.dataset.num_pixels
        self.max_depth = cfg.model.ray_tracer.ray_max_dist
        self.voxel_size = cfg.model.voxel_size
        num_images = len(os.listdir(os.path.join(
            self.data_root_dir, self.scan_id, "image")))
        img_ids = np.arange(self.shift, num_images, self.skip)
        self.num_images = len(img_ids)
        self.scene_images = {
            scan_id: [str(i) for i in img_ids]
        }
        self.sampling_idx = torch.arange(0, self.num_pixels)
        if stage == "val":
            self.val_seq = self.scan_id
        self.init_volumes(dimensions)

    def __len__(self):
        return len(self.scene_images[self.scan_id])

    def read_pose(self, path):
        with open(path, "r") as f:
            line = f.read().splitlines()[0].split(" ")
            pose = np.asarray([float(t) for t in line])
            nrow = int(np.sqrt(len(pose)))
        return pose.reshape(nrow, nrow).astype(np.float32)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def init_volumes(self, dimensions):
        self.volume_list = {}
        min_coords, max_coords, n_xyz = voxel_utils.get_world_range(
            dimensions, self.voxel_size)
        if self.dense_volume:
            self.volume_list[self.scan_id] = Volume(
                self.feat_dim, n_xyz, min_coords, max_coords, device_type=self.cfg.device_type
            )
        else:
            self.volume_list[self.scan_id] = VolumeList(
                self.feat_dim,
                self.voxel_size,
                dimensions,
                self.cfg.model.min_pts_in_grid
            )

    def reset_volumes(self):
        self.volume_list[self.scan_id].reset()

    def get_neighbor_xyz(self, xyz_map, mask, uv, kernel_size=15):
        uv = uv.numpy().astype(np.int32)
        mask = mask.astype(np.float32)
        n_pts = len(uv)
        half_length = np.floor(kernel_size / 2).astype(np.int32)
        img_h, img_w = xyz_map.shape[:2]
        out = np.zeros((n_pts, int(kernel_size ** 2), 3)) - 1000
        out_mask = np.zeros((n_pts, int(kernel_size ** 2))) - 1
        for i in range(n_pts):
            x, y = uv[i]
            y_start = y - half_length
            y_end = y + half_length
            y_range = np.arange(y_start, y_end+1)
            y_range = np.clip(y_range, a_min=0, a_max=img_h-1)
            x_start = x - half_length
            x_end = x + half_length
            x_range = np.arange(x_start, x_end+1)
            x_range = np.clip(x_range, a_min=0, a_max=img_w-1)
            indices = np.stack(np.meshgrid(y_range, x_range), axis=-1).reshape(-1, 2)
            out[i] = xyz_map[indices[:, 0], indices[:, 1], :3]
            out_mask[i] = mask[indices[:, 0], indices[:, 1]]

        assert (out_mask != -1).all()
        assert (out != -1000).all()
        out_mask = out_mask.astype(np.bool)
        return out, out_mask

    def get_uv_block(self, img_h, img_w, kernel_size):
        half_length = int(kernel_size/2)
        x_min = half_length
        x_max = img_w - half_length
        y_min = half_length
        y_max = img_h - half_length

        x = torch.randint(x_min, x_max, (1,)).item()
        y = torch.randint(y_min, y_max, (1,)).item()
        x_start = x - half_length
        x_end = x + half_length
        y_start = y - half_length
        y_end = y + half_length

        x_range = np.arange(x - half_length, x + half_length)
        assert np.min(x_range) >= 0
        assert np.max(x_range) < img_w
        y_range = np.arange(y - half_length, y + half_length)
        assert np.min(y_range) >= 0
        assert np.max(y_range) < img_h
        indices = np.stack(np.meshgrid(y_range, x_range), axis=-1).reshape(-1, 2)
        # switch x and y
        indices = indices[:, [1,0]]
        return indices

    def __getitem__(self, idx):
        scene = self.scan_id
        frame_id = self.scene_images[scene][idx]
        # prepare data for frame
        dimension_path = os.path.join(
            self.data_root_dir, scene, "pose", "dimensions.txt"
        )
        with open(dimension_path, "r") as f:
            line = f.read().splitlines()[0].split(" ")
            dimensions = np.asarray([float(l) for l in line])
        min_coords, max_coords, n_xyz = voxel_utils.get_world_range(
            dimensions, self.voxel_size)

        image_path = os.path.join(
            self.data_root_dir, scene, "image", f"{frame_id}.jpg")
        depth_path = os.path.join(
            self.data_root_dir, scene, "depth", f"{frame_id}.png")
        mask_path = os.path.join(
            self.data_root_dir, scene, "mask", f"{frame_id}.png")
        T_wc_path = os.path.join(
            self.data_root_dir, scene, "pose", f"T_wc_{frame_id}.txt")
        intr_mat_path = os.path.join(
            self.data_root_dir, scene, "pose", f"intr_mat_{frame_id}.txt")
        T_wc = self.read_pose(T_wc_path)
        intr_mat = self.read_pose(intr_mat_path)[:3, :3]
        intr_mat[:2, :3] = intr_mat[:2, :3] * self.downsample_scale
        rgb = load_rgb(image_path, downsample_scale=self.downsample_scale)
        clean_depth, noise_depth, frame_mask = load_depth(
            depth_path, self.downsample_scale, max_depth=self.max_depth, add_noise=False)
        normal = depth_to_normals(
            torch.from_numpy(clean_depth).unsqueeze(0).unsqueeze(0),
            torch.from_numpy(intr_mat).unsqueeze(0)
        )[0].permute(1, 2, 0).numpy()
        gt_xyz_map = depth_to_3d(
            torch.from_numpy(clean_depth).unsqueeze(0).unsqueeze(0),
            torch.from_numpy(intr_mat).unsqueeze(0)
        )[0].permute(1, 2, 0).numpy()
        img_h, img_w = clean_depth.shape
        gt_xyz_map_w = (T_wc @ geometry.get_homogeneous(gt_xyz_map.reshape(-1, 3)).T)[:3, :].T
        gt_xyz_map_w = gt_xyz_map_w.reshape(img_h, img_w, 3)
        # # get observed voxels
        # sdf, weight = voxel_utils.depth_to_tsdf(
        #     clean_depth,
        #     T_wc,
        #     intr_mat,
        #     min_coords,
        #     max_coords,
        #     np.asarray(n_xyz),
        #     self.voxel_size
        # )

        # NOTE: VERY IMPORTANT TO * -1 for normal due to a bug in data preparation in
        # data_prepare_depth_shapenet.py!
        normal_w = (T_wc[:3, :3] @ normal.reshape(-1, 3).T).T
        rgbd = np.concatenate([rgb, noise_depth[None, ...]], axis=0)
        pts_c = geometry.depth2xyz(clean_depth, intr_mat).reshape(-1, 3)
        pts_w_frame = (T_wc @ geometry.get_homogeneous(pts_c).T)[:3, :].T
        input_pts = np.concatenate(
            [pts_w_frame, normal_w],
            axis=-1
        )
        input_pts = input_pts[frame_mask.reshape(-1)]
        frame = {
            "depth_path": depth_path,
            "img_path": image_path,
            "scene_id": scene,
            "frame_id": frame_id,
            "T_wc": T_wc,
            "intr_mat": intr_mat,
            "rgbd": rgbd,
            "mask": frame_mask,
            "gt_pts": pts_w_frame,
            "gt_depth": clean_depth,
            "input_pts": input_pts,
        }

        return frame, {}


@register("fusion_inference_dataset_scannet")
class FusionInferenceDatasetScanNet(torch.utils.data.Dataset):
    def __init__(self, cfg, stage):
        super().__init__()
        self.scan_id = cfg.dataset.scan_id
        self.skip = cfg.dataset.skip_images
        self.shift = cfg.dataset.sample_shift
        self.downsample_scale = cfg.dataset.downsample_scale
        self.stage = stage
        self.feat_dim = cfg.model.feature_vector_size
        self.img_res = cfg.dataset.img_res
        self.total_pixels = self.img_res[0] * self.img_res[1]
        self.num_pixels = cfg.dataset.num_pixels
        self.max_depth = cfg.model.ray_tracer.ray_max_dist
        self.voxel_size = cfg.model.voxel_size
        
        img_dir = os.path.join(cfg.dataset.data_dir, self.scan_id, "frames", "color")
        
        num_images = len(os.listdir(img_dir))
        img_ids = np.arange(0, num_images, self.skip)
        self.img_h = self.img_res[0]
        self.img_w = self.img_res[1]
        self.num_images = len(img_ids)
        self.depth_scale = cfg.dataset.depth_scale
        axis_align_mat = scannet_helper.read_meta_file(
            os.path.join(
                cfg.dataset.data_dir, self.scan_id, f"{self.scan_id}.txt")
        )
        mesh_path = os.path.join(cfg.dataset.data_dir, self.scan_id, f"{self.scan_id}_vh_clean_2.ply")
        mesh = trimesh.load(mesh_path)
        mesh.vertices = (axis_align_mat @ geometry.get_homogeneous(mesh.vertices).T)[:3, :].T
        max_pts = np.max(mesh.vertices, axis=0)
        min_pts = np.min(mesh.vertices, axis=0)
        center = (min_pts + max_pts) / 2
        self.dimensions = np.asarray(max_pts - min_pts)
        recenter_mat = np.eye(4)
        recenter_mat[:3, 3] = -center
        self.axis_align_mat = recenter_mat @ axis_align_mat
        self.image_paths = []
        self.depth_paths = []
        self.T_wc_paths = []
        self.intr_mat_paths = []
        for img_id in img_ids:
            self.image_paths.append(
                os.path.join(
                    cfg.dataset.data_dir, self.scan_id, "frames", "color", f"{img_id}.jpg"
                )
            )
            self.depth_paths.append(
                os.path.join(
                    cfg.dataset.data_dir, self.scan_id, "frames", "depth", f"{img_id}.png"
                )
            )
            self.T_wc_paths.append(
                os.path.join(
                    cfg.dataset.data_dir, self.scan_id, "frames", "pose", f"{img_id}.txt"
                )
            )
            self.intr_mat_paths.append(
                os.path.join(
                    cfg.dataset.data_dir, self.scan_id, "frames", "intrinsic", "intrinsic_depth.txt"
                )
            )
        self.sampling_idx = torch.arange(0, self.num_pixels)

    def __len__(self):
        return len(self.image_paths)

    def read_extr_pose(self, path):
        T_cw = scannet_helper.read_extrinsic(path)
        T_wc = np.linalg.inv(T_cw)
        T_wc = self.axis_align_mat @ T_wc
        return T_wc

    def read_intr_pose(self, path):
        return scannet_helper.read_intrinsic(path)[:3, :3]

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        depth_path = self.depth_paths[idx]
        T_wc_path = self.T_wc_paths[idx]
        intr_mat_path = self.intr_mat_paths[idx]
        T_wc = self.read_extr_pose(T_wc_path)
        intr_mat = self.read_intr_pose(intr_mat_path)[:3, :3]
        intr_mat[:2, :3] = intr_mat[:2, :3] * self.downsample_scale
        clean_depth, noise_depth, frame_mask = load_depth(
            depth_path, self.downsample_scale, max_depth=self.max_depth, add_noise=False)
        rgb = np.zeros((3, clean_depth.shape[0], clean_depth.shape[1]))
        normal = depth_to_normals(
            torch.from_numpy(clean_depth).unsqueeze(0).unsqueeze(0),
            torch.from_numpy(intr_mat).unsqueeze(0)
        )[0].permute(1, 2, 0).numpy()
        gt_xyz_map = depth_to_3d(
            torch.from_numpy(clean_depth).unsqueeze(0).unsqueeze(0),
            torch.from_numpy(intr_mat).unsqueeze(0)
        )[0].permute(1, 2, 0).numpy()
        img_h, img_w = clean_depth.shape
        gt_xyz_map_w = (T_wc @ geometry.get_homogeneous(gt_xyz_map.reshape(-1, 3)).T)[:3, :].T
        gt_xyz_map_w = gt_xyz_map_w.reshape(img_h, img_w, 3)

        # NOTE: VERY IMPORTANT TO * -1 for normal due to a bug in data preparation in
        # data_prepare_depth_shapenet.py!
        normal_w = (T_wc[:3, :3] @ normal.reshape(-1, 3).T).T
        rgbd = np.concatenate([rgb, noise_depth[None, ...]], axis=0)
        pts_c = geometry.depth2xyz(clean_depth, intr_mat).reshape(-1, 3)
        pts_w_frame = (T_wc @ geometry.get_homogeneous(pts_c).T)[:3, :].T
        input_pts = np.concatenate(
            [pts_w_frame, normal_w],
            axis=-1
        )
        input_pts = input_pts[frame_mask.reshape(-1)]
        frame = {
            "depth_path": depth_path,
            "img_path": image_path,
            "scene_id": self.scan_id,
            "frame_id": idx,
            "T_wc": T_wc,
            "intr_mat": intr_mat,
            "rgbd": rgbd,
            "mask": frame_mask,
            "gt_pts": pts_w_frame,
            "gt_depth": clean_depth,
            "input_pts": input_pts,
        }
        return frame, {}


@register("fusion_inference_dataset_arkit")
class FusionInferenceDatasetARKit(torch.utils.data.Dataset):
    def __init__(self, cfg, stage):
        self.cfg = cfg
        self.data_root_dir = os.path.join(
            cfg.dataset.data_dir,
            cfg.dataset.subdomain,
        )
        self.dense_volume = cfg.trainer.dense_volume
        scan_id = cfg.dataset.scan_id
        self.scan_id = scan_id
        gt_mesh_path = os.path.join(self.data_root_dir, scan_id, f"{scan_id}_3dod_mesh.ply")
        gt_mesh = trimesh.load(gt_mesh_path)
        max_pts = np.max(gt_mesh.vertices, axis=0)
        min_pts = np.min(gt_mesh.vertices, axis=0)
        center = (min_pts + max_pts) / 2
        dimensions = max_pts - min_pts
        self.axis_align_mat = np.eye(4)
        self.axis_align_mat[:3, 3] = -center
        self.dimensions = dimensions
        self.skip = cfg.dataset.skip_images
        self.shift = cfg.dataset.sample_shift
        self.downsample_scale = cfg.dataset.downsample_scale
        self.stage = stage
        self.feat_dim = cfg.model.feature_vector_size
        self.img_res = cfg.dataset.img_res
        self.total_pixels = self.img_res[0] * self.img_res[1]
        self.num_pixels = cfg.dataset.num_pixels
        self.max_depth = cfg.model.ray_tracer.ray_max_dist
        self.voxel_size = cfg.model.voxel_size
        num_images = len(os.listdir(os.path.join(
            self.data_root_dir, self.scan_id, "image")))
        img_ids = np.arange(self.shift, num_images, self.skip)
        self.num_images = len(img_ids)
        self.scene_images = {
            scan_id: [str(i) for i in img_ids]
        }
        self.sampling_idx = torch.arange(0, self.num_pixels)
        if stage == "val":
            self.val_seq = self.scan_id
        self.init_volumes(dimensions)


class IterableInferenceDataset(torch.utils.data.Dataset):
    def __init__(self, img_list, ray_max_dist, bound_min, bound_max, n_xyz, sampling_size):
        self.img_list = img_list
        self.ray_max_dist = ray_max_dist
        self.bound_min = bound_min
        self.bound_max = bound_max
        self.n_xyz = n_xyz
        self.sampling_size = sampling_size
        self.n_iters = -1
        self.last_frame = -1

    def __len__(self):
        return self.n_iters

    def __getitem__(self, idx):
        print(self.last_frame, len(self.img_list))
        if self.last_frame != -1:
            img_id = torch.randint(self.last_frame, len(self.img_list), size=(1,))
        else:
            img_id = torch.randint(0, len(self.img_list), size=(1,))
        return self._sample_key_frame(self.img_list[img_id])

    def _load_rgb(self, path):
        rgb = cv2.imread(path, -1)[:, :, ::-1].transpose(2, 0, 1)
        return rgb

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

    def _sample_key_frame(self, meta_frame):
        depth, _, frame_mask = load_depth(
            meta_frame['depth_path'],
            downsample_scale=0,
            max_depth=self.ray_max_dist)
        rgb = np.zeros((3, depth.shape[0], depth.shape[1]))
        rgbd = np.concatenate([rgb, depth[None, ...]], axis=0)
        pts_c = geometry.depth2xyz(
            depth,
            meta_frame['intr_mat'].numpy()[0]
        ).reshape(-1, 3)
        pts_w_frame = (meta_frame['T_wc'][0].numpy() @ geometry.get_homogeneous(pts_c).T)[:3, :].T
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
            gt_xyz_map_w, frame_mask, uv, 3)
        rays = {
            "uv": uv.unsqueeze(0),
            "rgb": torch.from_numpy(rgb_ray).unsqueeze(0),
            "gt_pts": torch.from_numpy(gt_pts_ray).unsqueeze(0).float(),
            "intr_mat": meta_frame['intr_mat'].clone(),
            "T_wc": meta_frame['T_wc'].clone(),
            "mask": torch.from_numpy(mask_ray).unsqueeze(0).float(),
            "neighbor_pts": torch.from_numpy(neighbor_pts).unsqueeze(0).float(),
            "neighbor_masks": torch.from_numpy(neighbor_mask).unsqueeze(0).float()
        }
        return rays

