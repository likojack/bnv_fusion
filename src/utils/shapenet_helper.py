import numpy as np
from scipy.spatial.transform import Rotation


def read_pose(img):
    """ see render_depths.py for translation and intrinsic
    """
    img = img[:-1]
    x_rot, y_rot = [float(f) for f in img.split("_")]
    T_wo = np.eye(4)
    T_wo[2, 3] = -1
    rot_mat_0 = Rotation.from_euler(
        "y", y_rot, degrees=True).as_matrix()
    rot_mat_1 = Rotation.from_euler(
        "x", x_rot, degrees=True).as_matrix()
    T_wo[:3, :3] = rot_mat_1 @ rot_mat_0
    intr_mat = np.eye(3)
    intr_mat[0, 0] = 128
    intr_mat[1, 1] = 128
    intr_mat[0, 2] = 128
    intr_mat[1, 2] = 128
    T_ow = np.linalg.inv(T_wo)
    return T_ow.astype(np.float32), intr_mat.astype(np.float32)
