"""The highres depth is not error free. Some pixels are totally out-of-bound.
"""


import numpy as np
import os
import os.path as osp
from scipy.spatial.transform import Rotation


def read_extr(info):
    rot = np.asarray(info[1:4]).astype(np.float32)
    T_cw = np.eye(4)
    rot_mat = Rotation.from_rotvec(rot).as_matrix()
    T_cw[:3, :3] = rot_mat
    trans = np.asarray(info[4:7])
    T_cw[:3, 3] = trans
    return T_cw


def read_intr(path):
    with open(path, "r") as f:
        intr_info = f.read().split(" ")
    intr_mat = np.eye(3)
    intr_mat[0][0] = intr_info[2]
    intr_mat[1][1] = intr_info[3]
    intr_mat[0][2] = intr_info[4]
    intr_mat[1][2] = intr_info[5]
    return intr_mat


def get_frame_from_highres_time_stamps(dir, highres_dir, seq_name, poses, time_stamps):
    """
    get frame given the input time stamp in the highres folder.
    The pose is linearly interpolate if not given in the pose traj.

    Args:
        poses (_type_): _description_
        time_stamps (_type_): _description_
    """
    
    img_dir = osp.join(dir, "lowres_wide")
    depth_dir = osp.join(dir, "lowres_depth")
    confidence_dir = osp.join(dir, "confidence")
    high_res_img_dir = osp.join(highres_dir, "wide")
    high_res_depth_dir = osp.join(highres_dir, "highres_depth")

    time_stamps = np.asarray(time_stamps)
    src_time_stamps = np.asarray([float(p) for p in poses.keys()])
    time_diffs =np.abs(time_stamps[:, None] - src_time_stamps[None, :])
    match_ids = np.argsort(time_diffs, axis=-1)
    highres_frames = []
    counter = 0
    for i, time_stamp in enumerate(time_stamps):
        # calculate the weights for pose interpolation
        weights = np.abs(time_stamp - src_time_stamps[match_ids[i, :2]])
        weights = weights / np.sum(weights)
        weights = 1 - weights
        if weights[0] == 1:
            counter += 1
    
        assert np.abs(np.sum(weights) - 1) <= 1e-6
        T_cw = np.eye(4)
        id_0 = "{:.3f}".format(src_time_stamps[match_ids[i][0]])
        id_1 = "{:.3f}".format(src_time_stamps[match_ids[i][1]])
        p0 = poses[id_0]
        p1 = poses[id_1]
        quat = weights[0] * Rotation.from_matrix(p0[:3, :3]).as_quat() + \
            weights[1] * Rotation.from_matrix(p1[:3, :3]).as_quat()
        rot_mat = Rotation.from_quat(quat).as_matrix()
        T_cw[:3, :3] = rot_mat
        trans = weights[0] * p0[:3, 3] + weights[1] * p1[:3, 3]
        T_cw[:3, 3] = trans
        time_stamp = "{:.3f}".format(time_stamp)
        depth_name = f"{seq_name}_{time_stamp}.png"
        rgb_name = f"{seq_name}_{time_stamp}.png"
        high_res_rgb_name = f"{seq_name}_{time_stamp}.png"
        if (not os.path.exists(osp.join(img_dir, rgb_name))) or \
            (not os.path.exists(osp.join(depth_dir, depth_name))):

            print("lowres image not available")
            assert False

        intr_path = osp.join(dir, "lowres_wide_intrinsics", f"{seq_name}_{time_stamp}.pincam")
        intr_mat = read_intr(intr_path)
        intr_path = osp.join(dir, "wide_intrinsics", f"{seq_name}_{time_stamp}.pincam")
        highres_intr_mat = read_intr(intr_path)

        highres_frame = {
            "confidence_path": osp.join(confidence_dir, depth_name),
            "rgb_path": osp.join(img_dir, rgb_name),
            "depth_path": osp.join(depth_dir, depth_name),
            "high_res_rgb_path": osp.join(high_res_img_dir, high_res_rgb_name),
            "high_res_depth_path": osp.join(high_res_depth_dir, depth_name),
            "T_cw": T_cw,
            "intr_mat": intr_mat,
            "high_res_intr_mat": highres_intr_mat,
            "time_stamp": time_stamp
        }
        highres_frames.append(highres_frame)
    print(counter)
    return highres_frames


def read_poses(path):
    with open(path, "r") as f:
        lines = f.read().splitlines()
    
    poses = {}

    for l in lines:
        info = l.split(" ")
        time_stamp = "{:.3f}".format(round(float(info[0]), 3))
        T_cw = read_extr(info)
        poses[time_stamp] = T_cw
    return poses


def get_association(dir, poses, seq_name):
    img_dir = osp.join(dir, "lowres_wide")
    high_res_img_dir = osp.join(dir, "vga_wide")
    depth_dir = osp.join(dir, "lowres_depth")
    confidence_dir = osp.join(dir, "confidence")
    
    frames = []
    available_rgbs = []
    available_rgbs = np.asarray([float(f.split("_")[1][:-4]) for f in os.listdir(img_dir)])
    available_depths = np.asarray([float(f.split("_")[1][:-4]) for f in os.listdir(depth_dir)])
    n_skipped_imgs = 0
    for time_stamp in poses:
        depth_name = f"{seq_name}_{time_stamp}.png"
        rgb_name = f"{seq_name}_{time_stamp}.png"
        high_res_rgb_name = f"{seq_name}_{time_stamp}.png"
        if (not os.path.exists(osp.join(img_dir, rgb_name))) or \
            (not os.path.exists(osp.join(depth_dir, depth_name))) or \
            (not os.path.exists(osp.join(high_res_img_dir, high_res_rgb_name))):

            n_skipped_imgs += 1
            continue

        T_cw = poses[time_stamp]
        intr_path = osp.join(dir, "lowres_wide_intrinsics", f"{seq_name}_{time_stamp}.pincam")
        intr_mat = read_intr(intr_path)
        high_res_intr_mat = read_intr(
            osp.join(dir, "vga_wide_intrinsics", f"{seq_name}_{time_stamp}.pincam")
        )
        frame = {
            "confidence_path": osp.join(confidence_dir, depth_name),
            "rgb_path": osp.join(img_dir, rgb_name),
            "high_res_rgb_path": osp.join(img_dir, high_res_rgb_name),
            "depth_path": osp.join(depth_dir, depth_name),
            "T_cw": T_cw,
            "intr_mat": intr_mat,
            "high_res_intr_mat": high_res_intr_mat,
            "time_stamp": time_stamp
        }
        frames.append(frame)
    print(f"{len(poses)} poses, {len(available_rgbs)} rgb, {len(available_depths)} depths")
    print(f"skipped {n_skipped_imgs}/{len(available_rgbs)} due to missing poses")
    return frames


def get_pose_by_time_stamp(frames, time_stamp):
    poses = [f['T_cw'] for f in frames]
    time_stamps = [f['time_stamp'] for f in frames]
    time_diff = np.abs(time_stamps - time_stamp)
    ids = np.argsort(time_diff)
    

def get_frame_by_time_stamp(frames, time_stamp):
    for frame in frames:
        if frame['time_stamp'] == time_stamp:
            return frame
    print("no frame found")


def get_nearby_frames(frames, time_stamp, N=10, min_angle=15, min_distance=0.1):
    """
    get nearby frames given a time stamp. The frames should have
    enough parallax and covisibility.

    Args:
        frames (_type_): _description_
        time_stamp (_type_): _description_
        N (int, optional): _description_. Defaults to 10.
    """

    n = 0
    i = 0
    times = np.asarray([f['time_stamp'] for f in frames])
    time_diffs = np.abs(time_stamp - times)
    ref_frame = get_frame_by_time_stamp(frames, time_stamp)
    assert ref_frame is not None

    ids = np.argsort(time_diffs)
    pose_ref = np.linalg.inv(ref_frame['T_cw'])
    nearby_frames = []
    while n < N:
        source_frame = frames[ids[i]]
        T_wc = np.linalg.inv(source_frame['T_cw'])
        angle = np.arccos(
            ((np.linalg.inv(T_wc[:3, :3]) @ pose_ref[:3, :3] @ np.array([0, 0, 1]).T) * np.array(
                [0, 0, 1])).sum())
        dis = np.linalg.norm(T_wc[:3, 3] - pose_ref[:3, 3])
        if angle > (min_angle / 180) * np.pi or dis > min_distance:
            nearby_frames.append(source_frame)
            n += 1
        i += 1
    return nearby_frames


def get_high_res_time_stamp(in_dir):
    img_dir = osp.join(in_dir, "highres_depth")
    return sorted([float(f.split("_")[1][:-4]) for f in os.listdir(img_dir)])


class ARKitSceneDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __len__(self):

    def __getitem__(self, idx):
        

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cv2
    import open3d as o3d
    import src.utils.o3d_helper as o3d_helper
    import src.utils.geometry as geometry


    # seq_ids = os.listdir("./data/fusion/arkit")
    seq_ids = ["41048190"]
    for seq_id in seq_ids:
        root_dir = f"/home/kejie/Datasets_ssd/raw/Training/{seq_id}"
        upsample_dir = f"/home/kejie/Datasets_ssd/raw/Training/{seq_id}"
        traj_path = osp.join(root_dir, "lowres_wide.traj")
        poses = read_poses(traj_path)
        frames = get_association(root_dir, poses, seq_id)
        highres_time_stamps = get_high_res_time_stamp(upsample_dir)
        training_pairs = []
        highres_frames = get_frame_from_highres_time_stamps(root_dir, upsample_dir, seq_id, poses, highres_time_stamps)
        
        fusion = o3d_helper.TSDFFusion() 
        pts_list = []
        for frame in highres_frames:
            print(frame['high_res_depth_path'])
            depth = cv2.imread(frame['high_res_depth_path'], -1) / 1000.
            mask = (depth > 0).astype(np.float32)
            # color = cv2.imread(frame['high_res_rgb_path'], -1)
            # T_wc = np.linalg.inv(frame['T_cw'])
            # intr_mat = frame['high_res_intr_mat']
            
            depth = cv2.imread(frame['depth_path'], -1) / 1000.
            confidence = cv2.imread(frame["confidence_path"], -1)
            depth = depth * (confidence >= 2)
            mask = cv2.resize(mask, dsize=(depth.shape[1], depth.shape[0]))
            depth = depth * mask
            color = cv2.imread(frame['rgb_path'], -1)
            T_wc = np.linalg.inv(frame['T_cw'])
            intr_mat = frame['intr_mat']
            
            fusion.integrate(depth, color, T_wc, intr_mat)
            
            # xyz = geometry.depth2xyz(depth, intr_mat).reshape(-1, 3)
            # xyz = (T_wc @ geometry.get_homogeneous(xyz).T)[:3, :].T
            # pts_list.append(o3d_helper.np2pc(xyz))
            # o3d.visualization.draw_geometries(pts_list)
            # lowres_depth = cv2.imread(frame['depth_path'], -1) / 1000.
            # mask = cv2.imread(frame["confidence_path"], -1)
            # lowres_depth = lowres_depth * (mask >= 2)
            # _, axes = plt.subplots(1, 2)
            # axes[0].imshow(depth)
            # axes[1].imshow(lowres_depth)
            # plt.show()
        mesh = fusion.marching_cube("./low_res.ply")
        o3d.visualization.draw_geometries([o3d_helper.trimesh2o3d(mesh)])

        # for time_stamp in highres_time_stamps:
        #     ref_frame = get_frame_by_time_stamp(frames, time_stamp)
        #     if ref_frame is not None:
        #         source_frames = get_nearby_frames(frames, time_stamp)
        #         training_pairs.append([source_frames, ref_frame])
        #         # fig, axes = plt.subplots(3, 5)
        #         # src_color = cv2.imread(ref_frame['high_res_rgb_path'], -1)[:, :, ::-1]
        #         # axes[0][2].imshow(src_color)
        #         # print(ref_frame['time_stamp'])
        #         # for i in range(len(source_frames)):
        #         #     print(source_frames[i]['time_stamp'])
        #         #     src_color = cv2.imread(source_frames[i]['high_res_rgb_path'], -1)[:, :, ::-1]
        #         #     y = i // 5
        #         #     x = i - y * 5
        #         #     axes[y+1][x].imshow(src_color)
        #         # plt.show()
        # print(f"{len(training_pairs)}/{len(highres_time_stamps)}")

