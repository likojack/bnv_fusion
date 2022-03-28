import argparse
import os
import open3d as o3d
import numpy as np
import trimesh
from tqdm import tqdm

from src.utils.common import load_depth, load_rgb


def read_pose(path):
    with open(path, "r") as f:
        line = f.read().splitlines()[0].split(" ")
        pose = np.asarray([float(t) for t in line])
        nrow = int(np.sqrt(len(pose)))
    return pose.reshape(nrow, nrow).astype(np.float32)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--scan_id", required=True)
    arg_parser.add_argument("--dataset_name", required=True)
    arg_parser.add_argument("--skip", type=int, required=True)
    arg_parser.add_argument("--voxel", type=float, required=True)
    arg_parser.add_argument("--max_depth", type=float, required=True)
    arg_parser.add_argument("--shift", type=int, required=True)
    args = arg_parser.parse_args()

    root_dir = "/home/kejie/repository/fast_sdf/data/fusion/"

    voxel_size = args.voxel
    scale = args.skip
    render_path = f"{args.dataset_name}_{args.skip}_{args.shift}"
    max_depth = args.max_depth
    shift = args.shift
    sdf_trunc = min(voxel_size * 5, 0.05)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    seq_dir = os.path.join(root_dir, args.dataset_name, args.scan_id)
    img_dir = os.path.join(seq_dir, "image")
    depth_dir = os.path.join(seq_dir, "depth")
    mask_dir = os.path.join(seq_dir, "mask")
    pose_dir = os.path.join(seq_dir, "pose")

    n_imgs = len(os.listdir(img_dir))
    test_imgs = np.arange(shift, n_imgs, scale)
    cameras = np.load(os.path.join(seq_dir, "cameras.npz"))
    intr_mat = cameras['intr_mat_0']
    dimension = cameras['dimensions'] / 2.
    downsample_scale = 1
    intr_mat[:2, :3] = intr_mat[:2, :3] * downsample_scale
    for img in tqdm(test_imgs):
        rgb_path = os.path.join(img_dir, f"{img}.jpg")
        rgb = load_rgb(rgb_path, downsample_scale).transpose(1, 2, 0)
        rgb = (rgb / 2 + 0.5) * 255.
        T_wc_path = os.path.join(
            pose_dir, f"T_wc_{img}.txt"
        )
        T_wc = read_pose(T_wc_path)
        depth_path = os.path.join(depth_dir, f"{img}.png")
        depth_map, _, mask = load_depth(
            depth_path, downsample_scale, max_depth=max_depth)
        img_h, img_w = depth_map.shape
        depth_map = depth_map * (depth_map < 5)
        color = o3d.geometry.Image(rgb.astype(np.uint8))
        depth = o3d.geometry.Image((depth_map * 1000).astype(np.uint16))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=max_depth, convert_rgb_to_intensity=False)
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        img_h, img_w = depth_map.shape
        intrinsic.set_intrinsics(
            width=img_w,
            height=img_h,
            fx=intr_mat[0, 0],
            fy=intr_mat[1, 1],
            cx=intr_mat[0, 2],
            cy=intr_mat[1, 2],
        )
        # T_wc = cameras[f"T_wc_{img}"]
        T_cw = np.linalg.inv(T_wc)
        volume.integrate(
            rgbd,
            intrinsic,
            T_cw
        )
    print("Extract a triangle mesh from the volume and visualize it.")

    mesh_o3d = volume.extract_triangle_mesh()
    mesh_o3d.compute_vertex_normals()
    out_dir = f"./logs/tsdf_fusion/{render_path}/{args.dataset_name}/{args.scan_id}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh_o3d.vertices),
        faces=np.asarray(mesh_o3d.triangles),
        vertex_normals=np.asarray(mesh_o3d.vertex_normals)
    )
    mesh = trimesh.intersections.slice_mesh_plane(
        mesh, np.array([1, 0, 0]), np.array([-dimension[0], 0, 0]))
    mesh = trimesh.intersections.slice_mesh_plane(
        mesh, np.array([-1, 0, 0]), np.array([dimension[0], 0, 0]))
    mesh = trimesh.intersections.slice_mesh_plane(
        mesh, np.array([0, 1, 0]), np.array([0, -dimension[1], 0]))
    mesh = trimesh.intersections.slice_mesh_plane(
        mesh, np.array([0, -1, 0]), np.array([0, dimension[1], 0]))
    mesh = trimesh.intersections.slice_mesh_plane(
        mesh, np.array([0, 0, 1]), np.array([0, 0, -dimension[2]]))
    mesh = trimesh.intersections.slice_mesh_plane(
        mesh, np.array([0, 0, -1]), np.array([0, 0, dimension[2]]))
    mesh.export(
        os.path.join(out_dir, f"scene_scale_{scale}_voxel_size_{int(voxel_size*1000)}_max_depth_{int(max_depth*10)}_shift_{int(shift)}.ply")
    )
