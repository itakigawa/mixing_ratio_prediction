# Codebase for Mixing Ratio Prediction

## Description

This is a codebase for image-based machine learning for predicting the mixing ratios of solid chemicals.

## Dataset

The corresponding image dataset is available at .

## How to use this?

The details will be available.

```bash
$ git clone https://github.com/itakigawa/mixing_ratio_prediction.git
$ cd mixing_ratio_prediction
$ cd docker
$ . ./build_image.sh
$ cd ..
$ docker run --gpus all -it --rm -p 8888:8888 -v $PWD:/home/takigawa/work -v "zenodo_data_folder"/input:/home/takigawa/work/input inokuma/test bash
$ cd work
$ . ./script_allgen.sh
```

## Note

- docker 20.10.16
- mamba 0.24.0
- conda 4.13.0
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

- Ichigaku Takigawa (https://itakigawa.github.io/)
- Yasuhide Inokuma
- Yuki Ide



