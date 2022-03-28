import numpy as np
import open3d as o3d
import matplotlib.cm
from src.utils.motion_utils import Isometry


def pointcloud(pc, color: np.ndarray = None, normal: np.ndarray = None):
    if isinstance(pc, o3d.geometry.PointCloud):
        if pc.has_normals() and normal is None:
            normal = np.asarray(pc.normals)
        if pc.has_colors() and color is None:
            color = np.asarray(pc.colors)
        pc = np.asarray(pc.points)

    assert pc.shape[1] == 3 and len(pc.shape) == 2, f"Point cloud is of size {pc.shape} and cannot be displayed!"
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc)
    if color is not None:
        assert color.shape[0] == pc.shape[0], f"Point and color must have same size {color.shape[0]}, {pc.shape[0]}"
        point_cloud.colors = o3d.utility.Vector3dVector(color)
    if normal is not None:
        point_cloud.normals = o3d.utility.Vector3dVector(normal)

    return point_cloud


def frame(transform: Isometry = Isometry(), size=1.0):
    frame_obj = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    frame_obj.transform(transform.matrix)
    return frame_obj


def merged_linesets(lineset_list: list):
    merged_points = []
    merged_inds = []
    merged_colors = []
    point_acc_ind = 0
    for ls in lineset_list:
        merged_points.append(np.asarray(ls.points))
        merged_inds.append(np.asarray(ls.lines) + point_acc_ind)
        if ls.has_colors():
            merged_colors.append(np.asarray(ls.colors))
        else:
            merged_colors.append(np.zeros((len(ls.lines), 3)))
        point_acc_ind += len(ls.points)

    geom = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.vstack(merged_points)),
        lines=o3d.utility.Vector2iVector(np.vstack(merged_inds))
    )
    geom.colors = o3d.utility.Vector3dVector(np.vstack(merged_colors))
    return geom


def trajectory(traj1: list, traj2: list = None, ucid: int = -1):
    if len(traj1) > 0 and isinstance(traj1[0], Isometry):
        traj1 = [t.t for t in traj1]
    if traj2 and isinstance(traj2[0], Isometry):
        traj2 = [t.t for t in traj2]

    traj1_lineset = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(np.asarray(traj1)),
                                         lines=o3d.utility.Vector2iVector(np.vstack((np.arange(0, len(traj1) - 1),
                                                                                     np.arange(1, len(traj1)))).T))
    if ucid != -1:
        color_map = np.asarray(matplotlib.cm.get_cmap('tab10').colors)
        traj1_lineset.paint_uniform_color(color_map[ucid % 10])

    if traj2 is not None:
        assert len(traj1) == len(traj2)
        traj2_lineset = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(np.asarray(traj2)),
                                             lines=o3d.utility.Vector2iVector(np.vstack((np.arange(0, len(traj2) - 1),
                                                                                         np.arange(1, len(traj2)))).T))
        traj_diff = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.vstack((np.asarray(traj1), np.asarray(traj2)))),
            lines=o3d.utility.Vector2iVector(np.arange(2 * len(traj1)).reshape((2, len(traj1))).T))
        traj_diff.colors = o3d.utility.Vector3dVector(np.array([[1.0, 0.0, 0.0]]).repeat(len(traj_diff.lines), axis=0))

        traj1_lineset = merged_linesets([traj1_lineset, traj2_lineset, traj_diff])
    return traj1_lineset


def camera(T_wc, wh_ratio: float = 4.0 / 3.0, scale: float = 1.0, fovx: float = 90.0,
           color_id: int = -1):
    pw = np.tan(np.deg2rad(fovx / 2.)) * scale
    ph = pw / wh_ratio
    all_points = np.asarray([
        [0.0, 0.0, 0.0],
        [pw, ph, scale],
        [pw, -ph, scale],
        [-pw, ph, scale],
        [-pw, -ph, scale],
    ])
    line_indices = np.asarray([
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [1, 3], [3, 4], [2, 4]
    ])
    geom = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(all_points),
        lines=o3d.utility.Vector2iVector(line_indices))

    if color_id == -1:
        my_color = np.zeros((3,))
    else:
        my_color = np.asarray(matplotlib.cm.get_cmap('tab10').colors)[color_id, :3]
    geom.colors = o3d.utility.Vector3dVector(np.repeat(np.expand_dims(my_color, 0), line_indices.shape[0], 0))

    geom.transform(T_wc)
    return geom


def wireframe_bbox(extent_min=None, extent_max=None, color_id=-1):
    if extent_min is None:
        extent_min = [0.0, 0.0, 0.0]
    if extent_max is None:
        extent_max = [1.0, 1.0, 1.0]

    if color_id == -1:
        my_color = np.zeros((3,))
    else:
        my_color = np.asarray(matplotlib.cm.get_cmap('tab10').colors)[color_id, :3]

    all_points = np.asarray([
        [extent_min[0], extent_min[1], extent_min[2]],
        [extent_min[0], extent_min[1], extent_max[2]],
        [extent_min[0], extent_max[1], extent_min[2]],
        [extent_min[0], extent_max[1], extent_max[2]],
        [extent_max[0], extent_min[1], extent_min[2]],
        [extent_max[0], extent_min[1], extent_max[2]],
        [extent_max[0], extent_max[1], extent_min[2]],
        [extent_max[0], extent_max[1], extent_max[2]],
    ])
    line_indices = np.asarray([
        [0, 1], [2, 3], [4, 5], [6, 7],
        [0, 4], [1, 5], [2, 6], [3, 7],
        [0, 2], [4, 6], [1, 3], [5, 7]
    ])
    geom = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(all_points),
        lines=o3d.utility.Vector2iVector(line_indices))
    geom.colors = o3d.utility.Vector3dVector(np.repeat(np.expand_dims(my_color, 0), line_indices.shape[0], 0))

    return geom
