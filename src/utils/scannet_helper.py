''' Ref: https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts '''
import os
import sys
import json
import csv
import numpy as np
import quaternion
from plyfile import PlyData

import src.utils.geometry as geo_utils


def read_meta_file(meta_file):
    lines = open(meta_file).readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) \
                for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
    return axis_align_matrix


def read_intrinsic(file_path):
    with open(file_path, "r") as f:
        intrinsic = np.asarray(
            [list(map(lambda x: float(x), f.split())) for f in f.read().splitlines()]
        )
    return intrinsic


def read_extrinsic(file_path):
    with open(file_path, "r") as f:
        T_cam_scan = np.linalg.inv(
            np.asarray(
                [list(map(lambda x: float(x), f.split())) for f in f.read().splitlines()]
            )
        )
    return T_cam_scan


def read_mesh_vertices(filename):
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
    return vertices


def read_aggregation(filename):
    assert os.path.isfile(filename)
    object_id_to_segs = {}
    object_id_to_class = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        for i in range(num_objects):
            object_id = data['segGroups'][i]['objectId'] + 1 # instance ids should be 1-indexed
            label = data['segGroups'][i]['label']
            segs = data['segGroups'][i]['segments']
            object_id_to_segs[object_id] = segs
            object_id_to_class[object_id] = label
    return object_id_to_segs, object_id_to_class


def read_segmentation(filename):
    assert os.path.isfile(filename)
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data['segIndices'])
        for i in range(num_verts):
            seg_id = data['segIndices'][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts


def get_cam_azi(T_wc):
    """ get camera azimuth rotation. Assume z axis is upright
    """
    cam_orientation = np.array([[0, 0, 1], [0, 0, 0]])
    cam_orientation = (geo_utils.get_homogeneous(cam_orientation) @ T_wc.T)[:, :3]
    cam_orientation = cam_orientation[0] - cam_orientation[1]
    cam_orientation[2] = 0
    cam_orientation = cam_orientation / np.linalg.norm(cam_orientation)
    theta = np.arctan2(cam_orientation[1], cam_orientation[0])
    return theta


def make_M_from_tqs(t, q, s):
    q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    M = T.dot(R).dot(S)
    return M