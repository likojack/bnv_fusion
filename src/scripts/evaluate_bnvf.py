import argparse
import os
import open3d as o3d
import trimesh
import numpy as np
from sklearn.neighbors import NearestNeighbors


def evaluate(pred_points, gt_points, threshold, out, verbose=False):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(gt_points)
    distances, indices = nbrs.kneighbors(pred_points)
    pred_gt_dist = np.mean(distances)
    precision = np.sum(distances < threshold) / len(distances)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(pred_points)
    distances, indices = nbrs.kneighbors(gt_points)
    gt_pred_dist = np.mean(distances)
    recall = np.sum(distances < threshold) / len(distances)
    F1 = 2 * precision * recall / (precision + recall)
    if verbose:
        print("pred -> gt: ", pred_gt_dist)
        print(f"precision @ {threshold}:", precision)
        print("gt -> pred: ", gt_pred_dist)
        print(f"recall @ {threshold}:", recall)
        print("F1: ", F1)
        print("{:.3f}/{:.4f}/{:.3f}/{:.4f}/{:.4f}".format(pred_gt_dist, precision, gt_pred_dist, recall, F1))
    out['pred_gt'].append(pred_gt_dist)
    out['accuracy'].append(precision)
    out['gt_pred'].append(gt_pred_dist)
    out['recall'].append(recall)
    out['F1'].append(F1)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--pred_dir", required=True)
    arg_parser.add_argument("--gt_dir", required=True)
    arg_parser.add_argument("--file_name", required=True)
    args = arg_parser.parse_args()
    
    gt_dir = args.gt_dir
    pred_dir = args.pred_dir
    sequences = sorted(os.listdir(pred_dir))

    thresholds = [0.025]
    n_samples = 100000

    out = {
        "pred_gt": [],
        "accuracy": [],
        "gt_pred": [],
        "recall": [],
        "F1": []
    }
    for threshold in thresholds:
        for seq in sequences:
            print(f"{seq}:")
            gt_path = os.path.join(gt_dir, seq, "gt_mesh.ply")
            gt_mesh = o3d.io.read_triangle_mesh(gt_path)
            # gt_pts_poisson = np.asarray(gt_mesh.sample_points_poisson_disk(n_samples).points)
            gt_pts_uniform, _ = trimesh.sample.sample_surface(trimesh.load(gt_path), count=n_samples)
            # gt_pts_uniform = np.asarray(gt_mesh.sample_points_uniformly(n_samples).points)
            pred_seq_dir = os.path.join(pred_dir, seq)
            pred_file = [f for f in os.listdir(pred_seq_dir) if args.file_name in f]
            if len(pred_file) != 1:
                continue 
            pred_path = os.path.join(pred_seq_dir, pred_file[0])
            pred_mesh = o3d.io.read_triangle_mesh(pred_path)
            # pred_pts_poisson = np.asarray(pred_mesh.sample_points_poisson_disk(n_samples).points)
            pred_pts_uniform, _ = trimesh.sample.sample_surface(trimesh.load(pred_path), count=n_samples)
            # pred_pts_uniform = np.asarray(pred_mesh.sample_points_uniformly(n_samples).points)
            evaluate(pred_pts_uniform, gt_pts_uniform, threshold, out)
            # evaluate(pred_pts_poisson, gt_pts_poisson, threshold, out)
            print("sequence result:")
            print(out["pred_gt"][-1], out["accuracy"][-1], out["gt_pred"][-1], out["recall"][-1], out['F1'][-1])
            print("average result:")
            print(np.mean(out["pred_gt"]), np.mean(out["accuracy"]), np.mean(out["gt_pred"]), np.mean(out["recall"]), np.mean(out['F1']))
            

if __name__ == "__main__":
    main()