# BNV-Fusion

This repo implements the CVPR 2022 paper Bi-level Neural Volume Fusion.

# Requirements

Setup anaconda environment using the following command:

`
conda env create -f environment.yml -p CONDA_DIR/envs/bnv_fusion (CONDA_DIR is the folder where anaconda is installed)
`

You will need to the [torch-scatter](https://github.com/rusty1s/pytorch_scatter) additionally since conda doesn't seem to handle this package particularly well.


Alternatively, you can build a docker image using the DockerFile provided.


# Datasets and pretrained models
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


# Running
The following script is an example of running the system on all sequences in the 3D scene dataset.
```
export PYTHONPATH=$PYTHONPATH:$PWD
conda activate bnv_fusion
python src/script/run_inference_on_scene3d.py
```

To process just a sequence, use the following command:
```
python src/run_e2e.py model=fusion_pointnet_model dataset=fusion_inference_dataset trainer.checkpoint=$PWD/pretrained/pointnet.ckpt
```

# Training the embedding (optional)
Instead of using the pretrained model provided, you can also train the local embedding yourself by running the following command
```
python src/train.py model=fusion_pointnet_modeldataset=fusion_pointnet_dataset model.voxel_size=0.01 model.min_pts_in_grid=8 model.train_ray_splits=1000 model.tcnn_config=$PWD/src/models/tcnn_config.json
```