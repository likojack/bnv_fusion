import argparse
import open3d as o3d
import trimesh
import numpy as np
from sklearn.neighbors import NearestNeighbors

import src.utils.o3d_helper as o3d_helper


def visualize_errors(distances, gt_pts, max_dist=0.05, file_out=None):
    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap("plasma")
    distances = np.clip(distances, a_min=0, a_max=max_dist)
    colors = cmap(distances / max_dist)
    mesh = trimesh.Trimesh(
        vertices=gt_pts,
        faces=np.zeros_like(gt_pts),
        process=False
    )
    mesh.visual.vertex_colors = (colors * 255).astype(np.uint8)
    if file_out is not None:
        mesh.export(file_out)
    pts_o3d = o3d_helper.np2pc(mesh.vertices, mesh.visual.vertex_colors[:, :3] / 255.)
    # mesh_o3d = o3d_helper.trimesh2o3d(mesh)
    o3d.visualization.draw_geometries([pts_o3d])


args_parser = argparse.ArgumentParser()
args_parser.add_argument("--pred")
args_parser.add_argument("--gt")
args_parser.add_argument("--vertice_only", action="store_true")
args_parser.add_argument("--compute_normal", action="store_true")
args = args_parser.parse_args()


pred_mesh = trimesh.load(args.pred)
gt_mesh = trimesh.load(args.gt)
n_samples = 100000
threshold = 0.025
if args.vertice_only:
    gt_points = np.random.permutation(gt_mesh.vertices)[:n_samples]
else:
    gt_points, gt_face_id = trimesh.sample.sample_surface(gt_mesh, count=n_samples)
    gt_normal = gt_mesh.face_normals[gt_face_id]
    gt_face = gt_mesh.faces[gt_face_id]
pred_points, pred_face_id = trimesh.sample.sample_surface(pred_mesh, count=n_samples)
pred_normal = pred_mesh.face_normals[pred_face_id]
pred_face = pred_mesh.faces[pred_face_id]

nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(gt_points)
distances, indices = nbrs.kneighbors(pred_points)
# distances = np.clip(distances, a_min=0, a_max=0.05)
pred_gt_dist = np.mean(distances)
print("pred -> gt: ", pred_gt_dist)
precision = np.sum(distances < threshold) / len(distances)
print(f"precision @ {threshold}:", precision)
# pred_mesh_out = os.path.join(
#     "/".join(args.pred.split("/")[:-1]),
#     "pred_error.ply"
# )
# visualize_errors(distances[:, 0], pred_points, file_out=pred_mesh_out)

nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(pred_points)
distances, indices = nbrs.kneighbors(gt_points)
gt_pred_dist = np.mean(distances)
print("gt -> pred: ", gt_pred_dist)
recall = np.sum(distances < threshold) / len(distances)
print(f"recall @ {threshold}:", recall)
F1 = 2 * precision * recall / (precision + recall)
print("F1: ", F1)
print("{:.3f}/{:.4f}/{:.3f}/{:.4f}/{:.4f}".format(pred_gt_dist, precision, gt_pred_dist, recall, F1))
pred_normal = pred_normal[indices[:, 0]]
if args.compute_normal:
    assert not args.vertice_only
    print(np.mean(np.sum(gt_normal * pred_normal, axis=-1)))

# gt_mesh_out = os.path.join(
#     "/".join(args.pred.split("/")[:-1]),
#     "gt_error.ply"
# )
# visualize_errors(distances[:, 0], gt_points, max_dist=0.05, file_out=gt_mesh_out)
