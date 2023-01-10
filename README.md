<p align="center">

  <h1 align="center">BNV-Fusion: Dense 3D Reconstruction using Bi-level Neural Volume Fusion</h1>
  <p align="center">
    <a href="https://likojack.github.io/kejieli/#/home"><strong>Kejie Li</strong></a>
    ~
    <a href="https://andytang15.github.io/"><strong>Yansong Tang</strong></a>
    ~
    <a href="https://www.robots.ox.ac.uk/~victor/"><strong>Victor Adrian Prisacariu</strong></a>
    ~
    <a href="https://torrvision.com/"><strong>Philip H.S. Torr</strong></a>
  </p>
</p>

## BNV-Fusion ([Video](https://www.youtube.com/watch?v=ptx5vtQ9SvM) | [Paper](https://arxiv.org/pdf/2204.01139.pdf))

This repo implements the CVPR 2022 paper [Bi-level Neural Volume Fusion (BNV-Fusion)](https://arxiv.org/abs/2204.01139). BNV-Fusion leverages recent advances in neural implicit representations and neural rendering for dense 3D reconstruction. The keys to BNV-Fusion are 1) a sparse voxel grid of local shape codes to model surface geometry; 2) a well-designed bi-level fusion mechanism to integrate raw depth observations to the implicit grid efficiently and effectively. As a result, BNV-Fusin can run at a relatively **high frame rate** (2-5 frames per second on a desktop GPU) and reconstruct the 3D environment with **high accuracy**, where fine details missed by recent neural implicit based methods or traditional TSDF-Fusion are captured by BNV-Fusion.

## Requirements

Setup anaconda environment using the following command:

`
conda env create -f environment.yml -p CONDA_DIR/envs/bnv_fusion (CONDA_DIR is the folder where anaconda is installed)
`

You will need to the [torch-scatter](https://github.com/rusty1s/pytorch_scatter) additionally since conda doesn't seem to handle this package particularly well.


Alternatively, you can build a docker image using the DockerFile provided (Work in progress. We can't get the Open3D working within the docker image. Any help is appreciated!).

[IMPORTANT] Setup the PYTHONPATH before running the code:

`
export PYTHONPATH=$PYTHONPATH:$PWD
`

If you don't want to run this command everytime using a new terminal, you can also setup an alias in Bash to setup PYTHONPATH and activate the environment at one go as follows:

`
alias bnv_fusion="export PYTHONPATH=$PYTHONPATH:PROJECT_DIR;conda activate bnv_fusion"
`

PROJECT_DIR is the root directory of this repo.


**New: Running with sequences captured by iPhone/iPad**
------
We are happy to share that you can run BNV-Fusion reasonably easily on any sequences you captured using an iOS device with a lidar sensor (e.g., iPhone 12/13 Pro, iPad Pro). The instructions are as follows:
1. Download the [3D scanner app](https://apps.apple.com/us/app/3d-scanner-app/id1419913995) to an iOS device.
2. You can then capture a sequence using this app.
3. After recoding, you need to transfer the raw data (e.g., depth images, camera poses) to a desktop with a GPU. To do this, tap "scans" at the bottom left of the app. Select "Share Model" after clicking the "..." button. There are various formats you can use to share the data, but what we need is the raw data, so select "All Data". You can then choose your favorite way, such as google drive, for sharing. 
4. After you unpack the data at your desktop, run BNV-Fusion using the following command:
```
python src/run_e2e.py model=fusion_pointnet_model dataset=fusion_inference_dataset_arkit trainer.checkpoint=$PWD/pretrained/pointnet_tcnn.ckpt 'dataset.scan_id="xxxxx"' dataset.data_dir=yyyyy model.tcnn_config=$PWD/src/models/tcnn_config.json
```
Obviously, you need to specify the scan_id and where you hold the data. You should be able to see the reconstruction provided by BNV-Fusion after this step. Hope you have fun!


## Datasets and pretrained models
We tested BVF-Fusion on three datasets: 3D scene, ICL-NUIM, and ScanNet. Please go to the respective dataset repos to download data.
After downloading the data, run preprocessing scripts: 
```
python src/script/generate_fusion_data_{DATASET}.py
```

Instead of downloading the those datasets, we also provide some preprocessed data for one of the sequences in 3D scene in this link for quickly trying out BNV-Fusion. We can download the preprocessed data [here](https://drive.google.com/file/d/1nmdkK-mMpxebAO1MriCD_UbpLwbXYxah/view?usp=sharing).

You can also run the following command at the project root dir:
```
mkdir -p data/fusion/scene3d
cd data/fusion/scene3d/
pip install gdown (if gdown was not installed)
gdown https://drive.google.com/uc?id=1nmdkK-mMpxebAO1MriCD_UbpLwbXYxah
unzip lounge.zip && rm lounge.zip
```


## Running
<!-- The following script is an example of running the system on all sequences in the 3D scene dataset.
```
export PYTHONPATH=$PYTHONPATH:$PWD
conda activate bnv_fusion
python src/script/run_inference_on_scene3d.py
``` -->

To process a sequence, use the following command:
```
python src/run_e2e.py model=fusion_pointnet_model dataset=fusion_inference_dataset dataset.scan_id="scene3d/lounge" trainer.checkpoint=$PWD/pretrained/pointnet_tcnn.ckpt model.tcnn_config=$PWD/src/models/tcnn_config.json model.mode="demo"
```

## Evaluation
The results and GT meshes are availalbe here: https://drive.google.com/drive/folders/1gzsOIuCrj7ydX2-XXULQ61KjtITipYb5?usp=sharing

After downloading the data, you can run evaluation using the ```evaluate_bnvf.py```.

## Training the embedding (optional)
Instead of using the pretrained model provided, you can also train the local embedding yourself by running the following command
```
python src/train.py model=fusion_pointnet_modeldataset=fusion_pointnet_dataset model.voxel_size=0.01 model.min_pts_in_grid=8 model.train_ray_splits=1000 model.tcnn_config=$PWD/src/models/tcnn_config.json
```

## FAQ
- **Do I have to have a rough mesh, as requested [here](https://github.com/likojack/bnv_fusion/blob/9178e8c36743d6bf9a7828087553d365f50a6d7f/src/datasets/fusion_inference_dataset.py#L253), when running with my own data?**

No, We only use the mesh to determin the dimensions of the sceen to be reconstructed. You can manually set the boundary if you know the dimensions.

- **How to set an appropriate voxel size?**

The reconstruction quality apparently depends on the voxel size. If the voxel size is too small, there won't be enough points within each local region for the local embedding. If it is too large, the system fail to recover fine details. Therefore, we select the ideal voxel size based on the number of 3D points in a voxel. You will get a statistic on 3D points used in the local embedding after running system (see [here](https://github.com/likojack/bnv_fusion/blob/9178e8c36743d6bf9a7828087553d365f50a6d7f/src/models/sparse_volume.py#L515)). Empirically, we found out that the voxel size satisfying the following requirements gives better results: 1) the ```min``` is larger than 4, and 2) the ```mean``` is ideally larger than 8.  

## Citation
If you find our code or paper useful, please cite
```bibtex
@inproceedings{li2022bnv,
  author    = {Li, Kejie and Tang, Yansong and Prisacariu, Victor Adrian and Torr, Philip HS},
  title     = {BNV-Fusion: Dense 3D Reconstruction using Bi-level Neural Volume Fusion},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022}
}
```
