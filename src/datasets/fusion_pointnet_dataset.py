import os
import cv2
import torch
import numpy as np
import pickle
import trimesh
from kornia.geometry.depth import depth_to_normals, depth_to_3d

from src.datasets import register
import src.utils.geometry as geometry
from src.utils.shapenet_helper import read_pose


@register("fusion_pointnet_dataset")
class FusionPointNetDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, stage):
        super().__init__()
        self.n_local_samples = cfg.dataset.n_local_samples
        self.voxel_size = cfg.model.voxel_size
        self.stage = stage
        self.max_depth = cfg.model.ray_tracer.ray_max_dist
        self.data_root_dir = os.path.join(
            cfg.dataset.data_dir,
            cfg.dataset.subdomain,
        )
        # cats = ["03001627", "03636649"]
        cats = ["03001627_noise", "03636649_noise"]

        seq_dirs = [os.path.join(self.data_root_dir, f) for f in cats]    
        file_paths = []

        if stage == "test":
            self.data_root_dir = f"/home/kejie/repository/fast_sdf/data/rendering/03636649"
            seqs = sorted(os.listdir(self.data_root_dir))[:10]
            seqs = ["1a5ebc8575a4e5edcc901650bbbbb0b5"]
            for seq in seqs:
                file_names = os.listdir(os.path.join(self.data_root_dir, seq))
                file_paths.append(
                    [os.path.join(self.data_root_dir, seq, f) for f in file_names])
        elif stage == "val":    
            for seq_dir in seq_dirs:
                seqs = sorted(os.listdir(seq_dir))[:10]
                for seq in seqs:
                    file_names = os.listdir(os.path.join(seq_dir, seq))
                    file_names = sorted(file_names)
                    file_paths.append(
                        [os.path.join(seq_dir, seq, f) for f in file_names])
        else:  # stage == "train"
            for seq_dir in seq_dirs:
                seqs = sorted(os.listdir(seq_dir))[10:]
                for seq in seqs:
                    file_names = os.listdir(os.path.join(seq_dir, seq))
                    file_names = sorted(file_names)
                    file_paths.extend(
                        [os.path.join(seq_dir, seq, f) for f in file_names])
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def _resize_input_pts(self, pts):
        """ pts: [N, 3]
        """
        pts = torch.from_numpy(np.asarray(pts)).float()
        if len(pts) < self.n_local_samples:
            inds = torch.randint(0, len(pts), size=(self.n_local_samples,))
            pts = pts[inds]
            pts = pts.numpy()
        pts = np.random.permutation(pts)[:self.n_local_samples]
        return pts

    def __getitem__(self, idx):
        if self.stage == "test":
            max_depth = self.max_depth
            img_path = self.file_paths[idx][0]
            img_name = img_path.split("/")[-1].split(".")[0]
            instance_name = img_path.split("/")[-2]
            T_wc, intr_mat = read_pose(img_name)
            gt_depth = cv2.imread(img_path, -1) / 5000.
            mask = np.logical_and((gt_depth > 0), (gt_depth < max_depth)).reshape(-1)
            gt_depth[gt_depth > max_depth] = 0
            # compute the gt normal
            gt_normal = depth_to_normals(
                torch.from_numpy(gt_depth).unsqueeze(0).unsqueeze(0),
                torch.from_numpy(intr_mat).unsqueeze(0)
            )[0].permute(1, 2, 0).numpy()

            gt_xyz_map = depth_to_3d(
                torch.from_numpy(gt_depth).unsqueeze(0).unsqueeze(0),
                torch.from_numpy(intr_mat).unsqueeze(0)
            )[0].permute(1, 2, 0).numpy()

            gt_pts_c = gt_xyz_map.reshape(-1, 3)[mask]
            gt_pts_c[:, 2] *= -1
            gt_pts_c[:, 1] *= -1
            gt_pts_w = (T_wc @ geometry.get_homogeneous(gt_pts_c).T)[:3, :].T
            min_ = np.min(gt_pts_w, axis=0)
            max_ = np.max(gt_pts_w, axis=0)
            center = (min_ + max_) / 2
            gt_pts_w = gt_pts_w - center[None, :]
            gt_normal_w = (T_wc[:3, :3] @ gt_normal.reshape(-1, 3).T).T
            gt_normal_w = gt_normal_w[mask]
            gt_normal_w = gt_normal_w * -1
            input_pts = np.concatenate(
                [gt_pts_w, gt_normal_w],
                axis=-1
            )
            min_ = np.min(gt_pts_w, axis=0) - 2 * self.voxel_size
            max_ = np.max(gt_pts_w, axis=0) + 2 * self.voxel_size
            n_xyz = np.ceil((max_ - min_) / self.voxel_size).astype(int).tolist()
            bound_min = min_
            bound_max = bound_min + self.voxel_size * np.asarray(n_xyz)
            # input_pts = []
            # sample_centers = []
            # sample_ids = np.random.permutation(
            #     np.arange(len(gt_pts_w)))[:500]
            # for ind, sample_id in enumerate(sample_ids):
            #     dist = np.sqrt(np.sum(
            #         (gt_pts_w[sample_id:sample_id+1] - gt_pts_w) ** 2,
            #         axis=-1)
            #     )
            #     valid_neighbors = dist < self.voxel_size * 2
            #     neighbor_ids = np.random.permutation(
            #         np.nonzero(valid_neighbors)[0]
            #     )[:128]
            #     neighbor_pts = gt_pts_w[neighbor_ids]
            #     neighbor_normals = gt_normal_w[neighbor_ids]
            #     sample_pts = np.concatenate(
            #         [neighbor_pts, neighbor_normals],
            #         axis=-1
            #     )
            #     sample_pts = self._resize_input_pts(sample_pts)
            #     sample_center = gt_pts_w[sample_id:sample_id+1]
            #     sample_pts[:, :3] = sample_pts[:, :3] - sample_center
            #     input_pts.append(sample_pts)
            #     sample_centers.append(sample_center)
            # input_pts = np.stack(input_pts, axis=0)
            # sample_centers = np.concatenate(sample_centers, axis=0)

            return {
                "scene_id": instance_name,
                "input_pts": input_pts,
                # "sample_center": sample_centers,
                "bound_min": bound_min,
                "bound_max": bound_max,
            }
        elif self.stage == "val":
            file_paths = self.file_paths[idx]
            # randomly select 500 local patches of a shape
            sample_paths = np.random.permutation(file_paths)[:500]
            batch_input_pts = []
            batch_center = []
            batch_training_pts = []
            batch_gt = []
            for p in sample_paths:
                with open(p, "rb") as f:
                    data = pickle.load(f)
                if len(data['input_pts']) < 16:
                    continue
                input_pts = self._resize_input_pts(data['input_pts'])
                batch_input_pts.append(input_pts)
                batch_center.append(data['center'][0])
                batch_training_pts.append(np.asarray(data['training_pts']))
                batch_gt.append(np.asarray(data['gt_sdf']))
            batch_input_pts = np.stack(batch_input_pts, axis=0)
            batch_center = np.stack(batch_center, axis=0)
            batch_training_pts = np.stack(batch_training_pts, axis=0)
            batch_gt = np.stack(batch_gt, axis=0)
            data = {
                "scene_id": p.split("/")[-2],
                "input_pts": batch_input_pts,
                'sample_center': batch_center,
                "training_pts": batch_training_pts,
                "gt": batch_gt
            }
        else:
            file_path = self.file_paths[idx]
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            input_pts = self._resize_input_pts(data['input_pts'])
            # DEBUG:
            # import open3d as o3d
            # import src.utils.o3d_helper as o3d_helper
            # points = data['training_pts']
            # sdfs = data['gt_sdf']
            # surface_pts = data['input_pts']
            # visual_list = [
            #     o3d_helper.np2pc(surface_pts[:, :3], surface_pts[:, -3:] / 0.5 + 0.5)
            # ]
            # color = np.zeros_like(points)
            # color[sdfs > 0, 0] = 1
            # color[sdfs <= 0, 1] = 1
            # visual_list.append(o3d_helper.np2pc(points, color))
            # o3d.visualization.draw_geometries(visual_list)
            data = {
                "scene_id": file_path.split("/")[-1],
                "input_pts": input_pts,
                'sample_center': data['center'][0],
                "training_pts": np.asarray(data['training_pts']),
                "gt": np.asarray(data['gt_sdf'])
            }

        return data
