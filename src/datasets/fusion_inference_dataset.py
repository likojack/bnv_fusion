import json
import cv2
import os
import torch
import trimesh
import numpy as np
from kornia.geometry.depth import depth_to_normals, depth_to_3d

from src.datasets import register
import src.utils.geometry as geometry
import src.utils.voxel_utils as voxel_utils
from src.utils.common import load_depth, load_rgb
import src.utils.scannet_helper as scannet_helper


class FusionInferenceAbstractDataset(torch.utils.data.Dataset):
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
        self.depth_scale = cfg.dataset.depth_scale
        self.image_paths = []
        self.depth_paths = []
        self.T_wc_paths = []
        self.intr_mat_paths = []
        self.sampling_idx = torch.arange(0, self.num_pixels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        depth_path = self.depth_paths[idx]
        T_wc_path = self.T_wc_paths[idx]
        intr_mat_path = self.intr_mat_paths[idx]
        T_wc = self.read_extr_pose(T_wc_path)
        intr_mat = self.read_intr_pose(intr_mat_path)[:3, :3]
        rgb = self.read_rgb(image_path)
        depth, mask = self.read_depth(depth_path)
        normal = depth_to_normals(
            torch.from_numpy(depth).unsqueeze(0).unsqueeze(0),
            torch.from_numpy(intr_mat).unsqueeze(0)
        )[0].permute(1, 2, 0).numpy()
        gt_xyz_map = depth_to_3d(
            torch.from_numpy(depth).unsqueeze(0).unsqueeze(0),
            torch.from_numpy(intr_mat).unsqueeze(0)
        )[0].permute(1, 2, 0).numpy()
        img_h, img_w = depth.shape
        gt_xyz_map_w = (T_wc @ geometry.get_homogeneous(gt_xyz_map.reshape(-1, 3)).T)[:3, :].T
        gt_xyz_map_w = gt_xyz_map_w.reshape(img_h, img_w, 3)

        # NOTE: VERY IMPORTANT TO * -1 for normal due to a bug in data preparation in
        # data_prepare_depth_shapenet.py!
        normal_w = (T_wc[:3, :3] @ normal.reshape(-1, 3).T).T
        rgbd = np.concatenate([rgb, depth[None, ...]], axis=0)
        pts_c = geometry.depth2xyz(depth, intr_mat).reshape(-1, 3)
        pts_w_frame = (T_wc @ geometry.get_homogeneous(pts_c).T)[:3, :].T
        input_pts = np.concatenate(
            [pts_w_frame, normal_w],
            axis=-1
        )
        input_pts = input_pts[mask.reshape(-1)]
        frame = {
            "depth_path": depth_path,
            "img_path": image_path,
            "scene_id": self.scan_id,
            "frame_id": idx,
            "T_wc": T_wc,
            "intr_mat": intr_mat,
            "rgbd": rgbd,
            "mask": mask,
            "gt_pts": pts_w_frame,
            "gt_depth": depth,
            "input_pts": input_pts,
        }
        return frame, {}

    def read_rgb(self, path):
        raise NotImplementedError("subclass should implement this")
    
    def read_depth(self, path):
        raise NotImplementedError("subclass should implement this")
    
    def read_extr_pose(self, path):
        raise NotImplementedError("subclass should implement this")
    
    def read_intr_pose(self, path):
        raise NotImplementedError("subclass should implement this")


@register("fusion_inference_dataset")
class FusionInferenceDataset(FusionInferenceAbstractDataset):
    def __init__(self, cfg, stage):
        super().__init__(cfg, stage)
        root_dir = os.path.join(cfg.dataset.data_dir, self.scan_id)
        dimension_path = os.path.join(root_dir, "pose", "dimensions.txt")
        with open(dimension_path, "r") as f:
            line = f.read().splitlines()[0].split(" ")
            self.dimensions = np.asarray([float(l) for l in line])
        num_images = len(os.listdir(os.path.join(root_dir, "image")))
        for i in range(num_images):
            self.image_paths.append(os.path.join(root_dir, "image", f"{i}.jpg"))
            self.depth_paths.append(os.path.join(root_dir, "depth", f"{i}.png"))
            self.T_wc_paths.append(os.path.join(root_dir, "pose",
                                                f"T_wc_{i}.txt"))
            self.intr_mat_paths.append(os.path.join(root_dir, "pose",
                                                    f"intr_mat_{i}.txt"))

    def read_pose(self, path):
        with open(path, "r") as f:
            line = f.read().splitlines()[0].split(" ")
            pose = np.asarray([float(t) for t in line])
            nrow = int(np.sqrt(len(pose)))
        return pose.reshape(nrow, nrow).astype(np.float32)

    def read_extr_pose(self, path):
        return self.read_pose(path)

    def read_intr_pose(self, path):
        intr_mat = self.read_pose(path)
        intr_mat[:2, :3] = intr_mat[:2, :3] * self.downsample_scale
        return intr_mat

    def read_depth(self, path):
        depth, _, mask = load_depth(path, self.downsample_scale,
                                    max_depth=self.max_depth, add_noise=False)
        return depth, mask

    def read_rgb(self, path):
        return load_rgb(path, downsample_scale=self.downsample_scale)


@register("fusion_inference_dataset_scannet")
class FusionInferenceDatasetScanNet(FusionInferenceAbstractDataset):
    def __init__(self, cfg, stage):
        super().__init__(cfg, stage)
        img_dir = os.path.join(cfg.dataset.data_dir, self.scan_id, "frames", "color")
        num_images = len(os.listdir(img_dir))
        img_ids = np.arange(0, num_images, self.skip)
        self.num_images = num_images
        axis_align_mat = scannet_helper.read_meta_file(
            os.path.join(
                cfg.dataset.data_dir, self.scan_id, f"{self.scan_id}.txt")
        )
        mesh_path = os.path.join(cfg.dataset.data_dir,
                                 self.scan_id,
                                 f"{self.scan_id}_vh_clean_2.ply")
        mesh = trimesh.load(mesh_path)
        mesh.vertices = (axis_align_mat @\
            geometry.get_homogeneous(mesh.vertices).T)[:3, :].T
        
        max_pts = np.max(mesh.vertices, axis=0)
        min_pts = np.min(mesh.vertices, axis=0)
        center = (min_pts + max_pts) / 2
        self.dimensions = np.asarray(max_pts - min_pts)
        recenter_mat = np.eye(4)
        recenter_mat[:3, 3] = -center
        self.axis_align_mat = recenter_mat @ axis_align_mat
        frame_dir = os.path.join(cfg.dataset.data_dir, self.scan_id, "frames")
        for img_id in img_ids:
            self.image_paths.append(
                os.path.join(frame_dir, "color", f"{img_id}.jpg"))
            self.depth_paths.append(
                os.path.join(frame_dir, "depth", f"{img_id}.png"))
            self.T_wc_paths.append(
                os.path.join(frame_dir, "pose", f"{img_id}.txt"))
            self.intr_mat_paths.append(
                os.path.join(frame_dir, "intrinsic", "intrinsic_depth.txt"))

    def read_extr_pose(self, path):
        T_cw = scannet_helper.read_extrinsic(path)
        T_wc = np.linalg.inv(T_cw)
        T_wc = self.axis_align_mat @ T_wc
        return T_wc

    def read_intr_pose(self, path):
        intr_mat = scannet_helper.read_intrinsic(path)[:3, :3]
        intr_mat[:2, :3] *= self.downsample_scale
        return intr_mat

    def read_rgb(self, path):
        rgb = cv2.imread(path, -1)[:, :, ::-1] / 255. * 2 - 1
        rgb = cv2.resize(rgb, (self.img_res[1], self.img_res[0])).transpose(2, 0, 1)
        return rgb

    def read_depth(self, path):
        depth, _, mask = load_depth(path, self.downsample_scale, max_depth=self.max_depth, add_noise=False)
        return depth, mask

@register("fusion_inference_dataset_synthetic")
class FusionInferenceDatasetSynthetic(FusionInferenceAbstractDataset):    
    def __init__(self, cfg, stage):
        super().__init__(cfg, stage)
        root_dir = os.path.join(cfg.dataset.data_dir, self.scan_id) 
        num_images = len(os.listdir(os.path.join(root_dir, "image")))
        img_ids = np.arange(0, num_images, self.skip)
        cams = np.load(os.path.join(root_dir, "cameras_sphere.npz"))
        dimensions = float(2 * cams['scale_factor_0'])
        self.dimensions = np.asarray([dimensions, dimensions, dimensions])
        for i in img_ids:
            self.image_paths.append(os.path.join(root_dir, "image",
                                    "{:03d}.png".format(i)))
            self.depth_paths.append(os.path.join(root_dir, "depth",
                                    "{:03d}.png".format(i)))
            P = cams[f'world_mat_{i}'] @ cams[f'scale_mat_{i}']
            P = P[:3, :4]
            intrinsics, pose = geometry.load_K_Rt_from_P(None, P)
            self.T_wc_paths.append(pose)
            self.intr_mat_paths.append(intrinsics)

    def read_extr_pose(self, path):
        return path

    def read_intr_pose(self, path):
        return path

    def read_rgb(self, path):
        rgb = cv2.imread(path, -1)[:, :, ::-1] / 255. * 2 - 1
        rgb = cv2.resize(rgb, (self.img_res[1], self.img_res[0])).transpose(2, 0, 1)
        return rgb

    def read_depth(self, path):
        depth, _, mask = load_depth(path, self.downsample_scale, max_depth=self.max_depth, add_noise=False)
        return depth, mask


@register("fusion_inference_dataset_arkit")
class FusionInferenceDatasetARKit(torch.utils.data.Dataset):
    """ Dataset to process sequences captured by iphone/ipad via the "3D scanner" app
    """

    def __init__(self, cfg, stage):
        self.cfg = cfg
        self.data_root_dir = cfg.dataset.data_dir
        self.dense_volume = cfg.trainer.dense_volume

        self.seq_dir = os.path.join(self.data_root_dir, scan_id)
        rough_mesh_path = os.path.join(self.seq_dir, "export.obj")
        gt_mesh = trimesh.load(rough_mesh_path)
        max_pts = np.max(gt_mesh.vertices, axis=0)
        min_pts = np.min(gt_mesh.vertices, axis=0)
        center = (min_pts + max_pts) / 2
        dimensions = max_pts - min_pts
        self.axis_align_mat = np.eye(4)
        self.axis_align_mat[:3, 3] = -center
        self.dimensions = dimensions
        self.downsample_scale = cfg.dataset.downsample_scale
        self.stage = stage
        self.feat_dim = cfg.model.feature_vector_size
        self.img_res = cfg.dataset.img_res
        self.total_pixels = self.img_res[0] * self.img_res[1]
        self.num_pixels = cfg.dataset.num_pixels
        self.max_depth = cfg.model.ray_tracer.ray_max_dist
        self.voxel_size = cfg.model.voxel_size
        self.confidence_level = cfg.dataset.confidence_level
        self.mask_paths = []
        self.depth_paths = []
        self.img_paths = []
        self.cam_paths = []
        img_names = [f.split("_")[1].split(".")[0] for f in os.listdir(self.seq_dir) if f.startswith("depth_")]
        img_names = sorted(img_names, key=lambda a: int(a))
        self.num_images = len(img_names)
        self.sampling_idx = torch.arange(0, self.num_pixels)
        for img_name in img_names:
            self.depth_paths.append(os.path.join(self.seq_dir, f"depth_{img_name}.png"))
            self.mask_paths.append(os.path.join(self.seq_dir, f"conf_{img_name}.png"))
            self.cam_paths.append(os.path.join(self.seq_dir, f"frame_{img_name}.json"))
            img_path = os.path.join(self.seq_dir, f"frame_{img_name}.jpg")
            if not os.path.exists(img_path):
                self.img_paths.append(img_path)
            else:
                self.img_paths.append(None)
        self.intr_scale = 1 / 7.5  # the scale factor between the intr_mat for highres RGB and lowres depth
        
        if stage == "val":
            self.val_seq = self.scan_id

    def load_confidence(self, mask_path):
        confidence = cv2.imread(mask_path, -1)
        return confidence >= self.confidence_level
        
    def __getitem__(self, idx):
        depth_path = self.depth_paths[idx]
        img_path = self.img_paths[idx]
        if img_path is None:
            img_path = "rgb_not_exist"

        mask_path = self.mask_paths[idx]
        cam_path = self.cam_paths[idx]
        with open(cam_path, "r") as f:
            cam = json.load(f)
        intr_mat = np.asarray(cam['intrinsics']).reshape(3, 3)
        intr_mat[:2, :3] *= self.intr_scale * self.downsample_scale
        T_wc = np.asarray(cam['cameraPoseARFrame']).reshape(4, 4)
        T_align = np.eye(4)
        T_align[1, 1] = -1
        T_align[2, 2] = -1
        T_wc = self.axis_align_mat @ T_wc @ T_align
        clean_depth, noise_depth, frame_mask = load_depth(
            depth_path, self.downsample_scale, max_depth=self.max_depth, add_noise=False)

        frame_mask *= self.load_confidence(mask_path)

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
            "img_path": img_path,
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

