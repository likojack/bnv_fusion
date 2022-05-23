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

## BNV-Fusion

This repo implements the CVPR 2022 paper [Bi-level Neural Volume Fusion (BNV-Fusion)](https://arxiv.org/abs/2204.01139). BNV-Fusion leverages recent advances in neural implicit representations and neural rendering for dense 3D reconstruction. The keys to BNV-Fusion are 1) a sparse voxel grid of local shape codes to model surface geometry; 2) a well-designed bi-level fusion mechanism to integrate raw depth observations to the implicit grid efficiently and effectively. As a result, BNV-Fusin can run at a relatively **high frame rate** (2-5 frames per second on a desktop GPU) and reconstruct the 3D environment with **high accuracy**, where fine details missed by recent neural implicit based methods or traditional TSDF-Fusion are captured by BNV-Fusion.

## Requirements

Setup anaconda environment using the following command:

`
conda env create -f environment.yml -p CONDA_DIR/envs/bnv_fusion (CONDA_DIR is the folder where anaconda is installed)
`

You will need to the [torch-scatter](https://github.com/rusty1s/pytorch_scatter) additionally since conda doesn't seem to handle this package particularly well.


Alternatively, you can build a docker image using the DockerFile provided (Work in progress. We can't get the Open3D working within the docker image. Any help is appreciated!).


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
