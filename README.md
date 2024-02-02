# <p align="center"> <font color=#008000>GALA</font>: Generating Animatable Layered Assets <br> from a Single Scan </p>

## [Project Page](https://snuvclab.github.io/gala/) &nbsp;|&nbsp; [Paper](https://arxiv.org/abs/2401.12979) 

![teaser.png](./assets/teaser.png)

This is the official code for the paper "GALA: Generating Animatable Layered Assets from a Single Scan".

## News
- [2024/01/24] Initial release.

## Installation
Setup the environment using conda. We used a single 24GB gpu in our work, but you may adjust the batch size to fit your gpus.
``` 
conda env create -f env.yaml
conda activate gala
```

Install and download required libraries and data. For downloading SMPL-X, you must register [here](https://smpl-x.is.tue.mpg.de/register.php). Installing xformers reduces training time, but it takes extremely long. Remove it from "scripts/setup.sh" if needed.
```
bash scripts/setup.sh
```
Download "ViT-H HQ-SAM model" checkpoint [here](https://github.com/SysCV/sam-hq#model-checkpoints), and place it in ``./segmentation``.


## Running the code
### Prepare THuman2.0 Dataset
We use THuman2.0 in our demo since it's publicly accessible. The same pipeline also works for commercial dataset like RenderPeople, as used in our paper. Get access to Thuman2.0 scans and smplx parameters [here](https://github.com/ytrock/THuman2.0-Dataset) and organize the folder as below.
```
./data
├── thuman
│   └── 0001
│       └── 0001.obj
│       └── material0.mtl
│       └── material0.jpeg
│       └── smplx_param.pkl
│   └── 0002
│   └── ...
```

### Preprocess scans
For preprocessing THuman 2.0 scans, run the script below. Preprocessing includes normalization and segmentation of the input scan.
```
bash scripts/preprocess_data_single.sh thuman data/thuman/$SCAN_NAME $TARGET_OBJECT
# example
bash scripts/preprocess_data_single.sh thuman data/thuman/0001 jacket
```

### Run 
For canonicalized decomposition, run the commands below.
```
# Geometry Stage
python train.py config/th_0001_geo.yaml

# Appearance Stage
python train.py config/th_0001_tex.yaml
```
You can check the outputs in ``./results``. You can modify input text conditions in "config/th_0001_geo.yaml" or "config/th_0001_tex.yaml", and change experimental settings in "config/default_geo.yaml" or "config/default_geo.yaml".

## Citation

If you find this work useful, please cite our paper:

```
@misc{kim2024gala,
  title={GALA: Generating Animatable Layered Assets from a Single Scan}, 
  author={Taeksoo Kim and Byungjun Kim and Shunsuke Saito and Hanbyul Joo},
  year={2024},
  eprint={2401.12979},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

## Acknowledgement
We sincerely thank the authors of
- [nvdiffrec](https://github.com/NVlabs/nvdiffrec) 
- [Fantasia3D](https://github.com/Gorilla-Lab-SCUT/Fantasia3D) 
- [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) 
- [SCARF](https://github.com/yfeng95/SCARF) 
- [TeCH](https://github.com/huangyangyi/TeCH)

for their amazing work and codes!

## License
Codes are available only for non-commercial research purposes.
