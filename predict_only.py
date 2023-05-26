import datetime
import logging
import math
import os
import pickle
import re
import sys
import time
from operator import itemgetter

import albumentations as A
import glob
import hydra
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import skimage
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from hydra.utils import to_absolute_path
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import ImageGrid
from skimage import io
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from timm.loss.cross_entropy import SoftTargetCrossEntropy
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import (
    BalancedCE_MSELoss,
    Data,
    ImageDataset,
    Prediction,
    SymmetricSoftTargetCrossEntropy,
    calc_pred,
    create_net,
    plot_prediction,
    csv_export,
    print_out_images,
    test,
    train_step,
    tta_predict,
)


@hydra.main(config_name="config.yaml")
def main(cfg):
    with mlflow.start_run() as mlrun:
        t0 = time.time()
        run(cfg)
        elapsed = time.time() - t0

        logging.info(f"MLflow run id: {mlrun.info.run_id}")
        # versions
        for lib in [np, pd, matplotlib, torch, timm, A, hydra, mlflow]:
          logging.info(f"version {lib.__name__}: {lib.__version__}")
        logging.info(f"Elapsed time: {datetime.timedelta(seconds=elapsed)}")
        print(f"MLflow run id: {mlrun.info.run_id}", file=sys.stderr)
        print(f"Elapsed time: {datetime.timedelta(seconds=elapsed)}", file=sys.stderr)


def run(cfg):
    outdir = prepare_outdir()
    device = prepare_device()

    logging.info(f"workdir = {os.getcwd()}")
    logging.info(f"outdir = {outdir}")
    mlflow.log_param("outdir", outdir)

    writer = SummaryWriter(f"{outdir}/tblog")
    criterion, softmax = setup_loss(cfg["loss"])

    result_summary = {}
    target_info = {}

    outdim = cfg["outdim"]
    num_tta = cfg["num_tta"]

    if cfg["grayscale"]:
        in_chans = 1
    else:
        in_chans = 3

    if cfg["pretrain"] == "ImageNet" or cfg["pretrain"] == "None":
      print('specify pretrained model')
      sys.exit(-1)
    else: 
        model = create_net(
            cfg["model_name"], cfg["head"], cfg["concat_pool"], outdim, softmax, in_chans, pretrained=False
        )
        pretrain_model = to_absolute_path(cfg["pretrain"])
        model.load_state_dict(torch.load(pretrain_model))

    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"# params: {num_params:,}", file=sys.stderr)
    logging.info(f"# params: {num_params:,}")
    mlflow.log_param("num_params", num_params)

    # PREDICTION
    da_transform = get_da_filters(cfg)
    key = "test_dir"
    target_info[key] = cfg[key]
    test_dir = to_absolute_path(cfg["test_dir"])
    test_imgs = glob.glob(f"{test_dir}/*.jpg")

    test_dataset = ImageDataset(
        test_imgs, transform=None, grayscale=cfg["grayscale"]
    )
    test_predcsv = f"{outdir}/test_tta_prediction.csv"

    print("calc test_tta", file=sys.stderr)
    test_tta_pkl = f"{outdir}/test_tta.pkl"

    # tta_predict(model, device, dataset, transform, num_tta, softmax):
    y_pred = []

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for img in tqdm(test_dataset):
            images = torch.stack(
                [da_transform(image=img)["image"] for i in range(num_tta)]
            )
            images = images.to(device)
            out = model(images)
            pred_out = out.mean(dim=0).detach().cpu()
            if softmax == False:
                pred_out = pred_out.softmax(dim=-1)
            y_pred.append(pred_out.numpy())
    pred_test = np.array(y_pred)

    df = pd.DataFrame(columns=['img'] + [f'Pred_{o+1}' for o in range(outdim)])
    for i, img_path in enumerate(test_imgs):
        img_name = os.path.basename(img_path)
        df.loc[i] = [img_name] + [pred_test[i,j] for j in range(outdim)]
        print(i, pred_test[i,0], pred_test[i,1])
    df.to_csv(test_predcsv)
    print(f'csv_export to {test_predcsv}')

    with open(test_tta_pkl, "wb") as f:
        pickle.dump(pred_test, f)
    mlflow.log_artifact(test_tta_pkl)

    # OUTPUT SUMMARY
    print("-" * 10, file=sys.stderr)
    logging.info(outdir)

    for k in cfg.keys():
        logging.info(f"{k} = {cfg[k]}")

    for metric in result_summary.keys():
        mlflow.log_metric(metric, result_summary[metric])
        logging.info(f"{metric}: {result_summary[metric]:.5}")

    logging.info("test_dir: " + target_info["test_dir"])
    logging.info(f"num test images: {len(test_imgs)}")
    logging.info(f"working dir: {os.getcwd()}")
    logging.info(f"device: {torch.cuda.get_device_name(device)}")

def prepare_outdir():
    # output dir
    outdir = "output"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    return outdir


def prepare_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        cuda_id = torch.cuda.current_device()
        logging.info(f"cuda device: {torch.cuda.get_device_name(cuda_id)}")
    else:
        device = "cpu"
    return device


def setup_loss(loss):
    if loss in ["SoftTargetCrossEntropy", "SymmetricSoftTargetCrossEntropy"]:
        criterion = getattr(sys.modules[__name__], loss)()
        softmax = False
    elif loss == "BalancedCE_MSELoss":
        criterion = BalancedCE_MSELoss(alpha=0.5)
        softmax = False
    elif loss in ["MSELoss", "L1Loss"]:
        criterion = getattr(nn, loss)()
        softmax = True
    else:
        print("loss is undefined")
    return criterion, softmax


def learn_setup(mode, model, num, cfg):
    if mode == "warmup":
        optimizer = optim.Adam(
            model.parameters(), lr=cfg["lr_init_warmup"], weight_decay=0
        )
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg["lr_max_warmup"],
            steps_per_epoch=num,
            epochs=cfg["warmup_epochs"],
        )
    elif mode == "uptrain":
        optimizer = optim.Adam(
            model.parameters(), lr=cfg["lr_init_uptrain"], weight_decay=0
        )
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg["lr_max_uptrain"],
            steps_per_epoch=num,
            epochs=cfg["uptrain_epochs"],
        )
    else:
        print("error")

    return optimizer, scheduler


def get_da_filters(cfg):
    cent_crop_size = cfg["center_crop_size"]
    crop_size = cfg["crop_size"]
    input_size = cfg["input_size"]
    da_filters = [
        A.CenterCrop(cent_crop_size, cent_crop_size),
        A.RandomCrop(crop_size, crop_size),
        A.Resize(input_size, input_size),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=True, p=1.0),
        A.RandomRotate90(p=0.5)
    ]
    if cfg["flip"]:
        da_filters.extend([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ])
    if cfg["normalize"]:
        da_filters.append(
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            )
        )
    else:
        da_filters.append(A.ToFloat(max_value=255.0))
    da_filters.append(ToTensorV2()),
    print(da_filters, file=sys.stderr)
    logging.info(da_filters)
    return A.Compose(da_filters)


if __name__ == "__main__":
    s = f"file://{os.getcwd()}/mlruns"
    mlflow.set_tracking_uri(s)
    main()
