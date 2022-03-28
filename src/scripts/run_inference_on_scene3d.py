import numpy as np
import os
import subprocess
import trimesh

import src.utils.geometry as geometry
import src.utils.scannet_helper as scannet_helper

sequences = ["lounge", "cactusgarden", "stonewall", "copyroom", "burghers"]

for seq in sequences:
    for skip in [10]:
        for shift in range(0, 15, 5):
            exp_name = f"scene3d_{skip}_{shift}"
            out_root = f"/home/kejie/repository/fast_sdf/logs/test/{exp_name}/"
            commands = (
                "python src/test.py model=fusion_pointnet_model dataset=fusion_inference_dataset "
                "dataset.num_workers=0 "
                "dataset.downsample_scale=1 "
                "model.ray_tracer.ray_max_dist=3 "
                "model.voxel_size=0.01 "
                "model.min_pts_in_grid=8 "
                "trainer.checkpoint=/home/kejie/repository/fast_sdf/logs/train/2021-10-21/22-37-03/lightning_logs/version_0/checkpoints/last.ckpt"
            )
            commands = commands.split(" ")
            commands += [f"dataset.scan_id=scene3d/{seq}"]
            commands += [f"dataset.out_root={out_root}"]
            commands += [f"dataset.skip_images={skip}"]
            commands += [f"dataset.sample_shift={shift}"]

            try:
                subprocess.run(commands, check=True)
                print(f"finish {seq}")
            except subprocess.CalledProcessError:
                import pdb
                pdb.set_trace()

            commands = (
                "python src/train.py model=fusion_refiner_model dataset=fusion_refiner_dataset "
                "dataset.num_workers=4 "
                "dataset.downsample_scale=1 "
                "model.ray_tracer.ray_max_dist=3 "
                "model.voxel_size=0.01 "
                "model.min_pts_in_grid=8 "
                "trainer.max_epochs=30 "
                "trainer.check_val_every_n_epoch=10 "
                "dataset.num_pixels=5000 "
                "model.train_ray_splits=2500 "
                "model.sdf_delta_weight=0.1 "
                "model.pretrained_model=/home/kejie/repository/fast_sdf/logs/train/2021-10-21/22-37-03/lightning_logs/version_0/checkpoints/last.ckpt"
            )
            commands = commands.split(" ")            
            commands += [f"dataset.scan_id=scene3d/{seq}"]
            
            volume_path = f"/home/kejie/repository/fast_sdf/logs/test/{exp_name}/scene3d/{seq}"
            commands += [f"model.volume_dir={volume_path}"]
            
            commands += [f"dataset.skip_images={skip}"]
            commands += [f"dataset.sample_shift={shift}"]
            
            out_root = f"/home/kejie/repository/fast_sdf/logs/train/{exp_name}/"
            commands += [f"dataset.out_root={out_root}"]
            try:
                subprocess.run(commands, check=True)
                print(f"finish {seq}")
            except subprocess.CalledProcessError:
                import pdb
                pdb.set_trace()