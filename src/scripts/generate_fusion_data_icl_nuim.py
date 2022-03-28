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
import open3d as o3d
import sys
from tqdm import tqdm
import trimesh
from scipy.spatial.transform import Rotation
import cv2
import numpy as np

# from src.datasets.scenenet import SceneNet
from src.utils.geometry import get_homogeneous, depth2xyz
import src.utils.scannet_helper as scannet_helper


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
        T_wc = np.asarray([line.split(" ") for line in lines[start_line:end_line]]).astype(np.float32)
        start_line += 5
        end_line += 5
        T_wcs.append(T_wc)
    return T_wcs


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--noise", action="store_true")
arg_parser.add_argument("--scan_id", required=True)
args = arg_parser.parse_args()


ROOT_DIR = "/home/kejie/repository/fast_sdf/data/icl_nuim/"
RENDER_PATH = args.scan_id
GT_MESH_PATH = os.path.join(ROOT_DIR, RENDER_PATH[:-1] + ".ply")
# gt_mesh = o3d.io.read_triangle_mesh(GT_MESH_PATH)
gt_mesh = trimesh.load(GT_MESH_PATH)
max_pts = np.max(np.asarray(gt_mesh.vertices), axis=0)
min_pts = np.min(np.asarray(gt_mesh.vertices), axis=0)
center = (min_pts + max_pts) / 2
dimensions = max_pts - min_pts
axis_align_mat = np.eye(4)
axis_align_mat[:3, 3] = -center

out_base_dir = "/home/kejie/repository/fast_sdf/data/fusion/icl_nuim"
DEPTH_SCALE = 1000.
SKIP_IMAGES = 1
intr_mat = np.eye(3)
intr_mat[0, 0] = 525
intr_mat[0, 2] = 319.5
intr_mat[1, 1] = 525.
intr_mat[1, 2] = 239.5

if args.noise:
    out_dir = os.path.join(out_base_dir, f"{RENDER_PATH}_noise")
else:
    out_dir = os.path.join(out_base_dir, f"{RENDER_PATH}")

gt_mesh.vertices = (axis_align_mat @ get_homogeneous(np.asarray(gt_mesh.vertices)).T)[:3, :].T
gt_mesh.export(os.path.join(out_dir, "gt_mesh.ply"))

out_rgb_dir = os.path.join(out_dir, "image")
out_mask_dir = os.path.join(out_dir, "mask")
out_depth_dir = os.path.join(out_dir, "depth")
out_pose_dir = os.path.join(out_dir, "pose")
make_dir(out_dir)
make_dir(out_rgb_dir)
make_dir(out_mask_dir)
make_dir(out_depth_dir)
make_dir(out_pose_dir)

seq_dir = os.path.join(ROOT_DIR, RENDER_PATH)
img_dir = os.path.join(seq_dir, f"{RENDER_PATH}-color")
if args.noise:
    depth_dir = os.path.join(seq_dir, f"{RENDER_PATH}-depth-simulated")
else:
    depth_dir = os.path.join(seq_dir, f"{RENDER_PATH}-depth-clean")

pose_path = os.path.join(seq_dir, "pose.txt")

img_names = [f.split(".")[0] for f in os.listdir(img_dir)]
img_names = sorted(img_names, key=lambda a: int(a))
n_imgs = len(img_names)

T_wcs = read_cam_traj(pose_path, n_imgs)
min_pts = []
max_pts = []

cameras = {}
# get the 3D bounding box of the scene
used_id = 0

pts_o3d = []
cameras_new = {}
for i in range(0, n_imgs, SKIP_IMAGES):
    rgb = cv2.imread(
        os.path.join(img_dir, img_names[i] + ".jpg"), -1)[:, :, ::-1]
    depth = cv2.imread(
        os.path.join(depth_dir, img_names[i] + ".png"), -1) / DEPTH_SCALE
    mask = (depth > 0).astype(np.float32)
    img_h, img_w = mask.shape
    y, x = np.nonzero(depth)
    valid_pixels = np.stack([x, y], axis=-1)
    img_h, img_w = depth.shape
    rgb = cv2.resize(rgb, (img_w, img_h))

    T_wc = T_wcs[i]
    T_wc = axis_align_mat @ T_wc
    cameras['intr_mat_%d'%used_id] = intr_mat
    cameras['T_wc_%d'%used_id] = T_wc

    pts_c = depth2xyz(depth, intr_mat)
    pts_c = pts_c[valid_pixels[:, 1], valid_pixels[:, 0], :].reshape(-1, 3)
    pts_w = (T_wc @ get_homogeneous(pts_c).T)[:3, :].T
    _min = np.min(pts_w, axis=0)
    _max = np.max(pts_w, axis=0)
    min_pts.append(_min)
    max_pts.append(_max)

    out_rgb_path = os.path.join(out_rgb_dir, f"{used_id}.jpg")
    cv2.imwrite(out_rgb_path, rgb[:, :, ::-1])
    out_mask_path = os.path.join(out_mask_dir, f"{used_id}.png")
    cv2.imwrite(out_mask_path, mask.astype(np.uint8)*255)
    out_depth_path = os.path.join(out_depth_dir, f"{used_id}.png")
    cv2.imwrite(out_depth_path, (depth * 1000).astype(np.uint16))

    extr_path = os.path.join(out_pose_dir, f"T_wc_{used_id}.txt")
    with open(extr_path, "w") as f:
        f.write(" ".join([str(t) for t in T_wc.reshape(-1)]))
    cameras_new['T_wc_%d'%used_id] = T_wc
    cameras_new['intr_mat_%d'%used_id] = cameras['intr_mat_%d'%used_id]
    intr_path = os.path.join(out_pose_dir, f"intr_mat_{used_id}.txt")
    with open(intr_path, "w") as f:
        f.write(" ".join([str(t) for t in intr_mat.reshape(-1)]))
    used_id += 1

cameras_new['dimensions'] = dimensions
np.savez('{0}/{1}.npz'.format(out_dir, "cameras"), **cameras_new)

dimension_path = os.path.join(out_pose_dir, "dimensions.txt")
with open(dimension_path, "w") as f:
    f.write(" ".join([str(t) for t in dimensions.reshape(-1)]))