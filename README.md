# Codebase for Mixing Ratio Prediction

## Description

This is the codebase for image-based machine learning for predicting the mixing ratios of solid chemicals.

## Dataset

The corresponding image dataset `mixratio_dataset_20230526.tar.xz` (2.25G) is available at figshare as [doi:10.6084/m9.figshare.20521224.v2](https://doi.org/10.6084/m9.figshare.20521224.v2).

## How to use this?

Data preparation

```bash
$ curl -L -o mixratio_dataset_20230526.tar.xz https://figshare.com/ndownloader/files/40911584
$ mkdir data
$ tar Jxf mixratio_dataset_20230526.tar.xz -C data
```

Run (requires NVIDIA GPU, CUDA 11.5, cuDNN)

```bash
$ git clone https://github.com/itakigawa/mixing_ratio_prediction.git
$ cd mixing_ratio_prediction
$ docker pull itakigawa/cu115_torch_timm
$ docker run --gpus all -it --rm -v $PWD:/home/takigawa/work -v `dirname $(pwd)`/data/input:/home/takigawa/work/input itakigawa/cu115_torch_timm bash
$ cd work
$ . ./script_allgen.sh
```

## Note

- docker 20.10.16
- mamba 0.24.0 (conda 4.13.0)
- Python 3.10.5
- PyTorch 1.11.0+cu115
- OpenCV 4.6.0
- pytorch-image-models 0.6.2.dev0
- albumentations 1.2.0
- numpy 1.23.0
- hydra 1.2.0
- mlflow 1.26.1
- tensorboard 2.9.1

## Licence

[MIT](https://github.com/tcnksm/tool/blob/master/LICENCE)

## Authors

Ichigaku Takigawa (https://itakigawa.github.io/)

