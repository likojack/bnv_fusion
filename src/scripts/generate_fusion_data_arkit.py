"""
What idr needs:
    cameras.npz
        scale_mat_{i}
        world_mat_{i}
    image
    mask
"""
import argparse
import os
import sys
from tqdm import tqdm
import open3d as o3d
import trimesh

import cv2
import numpy as np

from src.utils.geometry import get_homogeneous, depth2xyz
from src.datasets.arkitscene_dataset import get_association
from src.utils.common import load_depth


def make_dir(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)


def read_cam_traj(path, n_imgs):
    T_wcs = []
    start_line = 1
    end_line = 5
    with open(path, "r") as f:
        lines = f.read().splitlines()
    assert len(lines) / 5 == n_imgs
    for i in range(n_imgs):
        if "\t" in lines[start_line]:
            line = [
                line.split("\t") for line in lines[start_line:end_line]
            ]
        else:
            line = [
                [l for l in line.split(" ") if len(l) > 0] for line in lines[start_line:end_line]
            ]
        T_wc = np.asarray(line).astype(np.float32)
        start_line += 5
        end_line += 5
        T_wcs.append(T_wc)
    return T_wcs


ROOT_DIR = "/home/kejie/Datasets_ssd/raw/Training/"
out_base_dir = "/home/kejie/repository/fast_sdf/data/fusion/arkit"

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--seq")
args = arg_parser.parse_args()
seq_names = [args.seq]

for name in tqdm(seq_names):

    gt_mesh_path = os.path.join(ROOT_DIR, name, f"{name}_3dod_mesh.ply")
    gt_mesh = trimesh.load(gt_mesh_path)
    max_pts = np.max(gt_mesh.vertices, axis=0)
    min_pts = np.min(gt_mesh.vertices, axis=0)
    center = (min_pts + max_pts) / 2
    dimensions = max_pts - min_pts
    axis_align_mat = np.eye(4)
    axis_align_mat[:3, 3] = -center
    base_dir = os.path.join(ROOT_DIR, name)
    frames = get_association(base_dir, name)
    n_imgs = len(frames)
    out_dir = os.path.join(out_base_dir, name)
    out_rgb_dir = os.path.join(out_dir, "image")
    out_mask_dir = os.path.join(out_dir, "mask")
    out_depth_dir = os.path.join(out_dir, "depth")
    out_pose_dir = os.path.join(out_dir, "pose")
    make_dir(out_dir)
    make_dir(out_rgb_dir)
    make_dir(out_mask_dir)
    make_dir(out_depth_dir)
    make_dir(out_pose_dir)

    gt_mesh.vertices = (axis_align_mat @ get_homogeneous(gt_mesh.vertices).T)[:3, :].T
    gt_mesh.export(os.path.join(out_dir, "gt_mesh.ply"))
    # get the 3D bounding box of the scene
    cameras_new = {}
    skip = 0
    i = 0
    for _ in range(0, n_imgs):
        if not (os.path.exists(frames[i]['rgb_path']) and os.path.exists(frames[i]['depth_path'])):
            skip += 1
            continue
        
        rgb = cv2.imread(frames[i]['rgb_path'], -1)
        depth_map, _, _ = load_depth(frames[i]['depth_path'], 1, max_depth=1000.)
        mask = cv2.imread(frames[i]['confidence_path'], -1)
        depth_map = depth_map * (mask >= 2)
        ind_y, ind_x = np.nonzero(depth_map != 0)
        mask = (depth_map > 0).astype(np.float32)
        img_h, img_w = mask.shape
        T_wc = np.linalg.inv(frames[i]['T_cw'])
        T_wc = axis_align_mat @ T_wc
        out_rgb_path = os.path.join(out_rgb_dir, f"{i}.jpg")
        cv2.imwrite(out_rgb_path, rgb)
        out_mask_path = os.path.join(out_mask_dir, f"{i}.png")
        cv2.imwrite(out_mask_path, mask.astype(np.uint8)*255)
        out_depth_path = os.path.join(out_depth_dir, f"{i}.png")
        cv2.imwrite(out_depth_path, (depth_map * 1000).astype(np.uint16))

        intr_mat = np.eye(4)
        intr_mat[:3, :3] = frames[i]['intr_mat']
        cameras_new['intr_mat_%d'%i] = intr_mat
        cameras_new['T_wc_%d'%i] = T_wc
        intr_path = os.path.join(out_pose_dir, f"intr_mat_{i}.txt")
        with open(intr_path, "w") as f:
            f.write(" ".join([str(t) for t in intr_mat.reshape(-1)]))
        extr_path = os.path.join(out_pose_dir, f"T_wc_{i}.txt")
        with open(extr_path, "w") as f:
            f.write(" ".join([str(t) for t in T_wc.reshape(-1)]))
        i += 1
    print(f"skip {skip} images")
    cameras_new['dimensions'] = dimensions
    np.savez('{0}/{1}.npz'.format(out_dir, "cameras"), **cameras_new)
    dimension_path = os.path.join(out_pose_dir, "dimensions.txt")
    with open(dimension_path, "w") as f:
        f.write(" ".join([str(t) for t in dimensions.reshape(-1)]))

