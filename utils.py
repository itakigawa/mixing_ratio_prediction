import glob
import math
import os
import random
import re
import sys
from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import ImageGrid
from skimage import io
from skimage.color import rgb2gray
from timm.loss.cross_entropy import SoftTargetCrossEntropy
from timm.models.layers import SelectAdaptivePool2d
from torch.utils.data import Dataset
from tqdm import tqdm, trange

# https://github.com/rwightman/pytorch-image-models/blob/master/timm/loss/cross_entropy.py

# Symmetric Cross Entropy for Robust Learning with Noisy Labels (ICCV2019)
# https://arxiv.org/abs/1908.06112


class SymmetricSoftTargetCrossEntropy(nn.Module):
    def __init__(self, alpha=0.1, beta=1.0):
        super(SymmetricSoftTargetCrossEntropy, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, target):
        ce = torch.sum(-target * torch.log_softmax(x, dim=-1), dim=-1)
        t_logsoftmax = torch.log(target).nan_to_num(neginf=-6)
        rce = torch.sum(-torch.softmax(x, dim=-1) * t_logsoftmax, dim=-1)
        return self.alpha * ce.mean() + self.beta * rce.mean()


class SoftmaxMSELoss(nn.Module):
    def __init__(self):
        super(SoftmaxMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, x, target):
        y_pred = torch.softmax(x, dim=-1)
        return self.mse(y_pred, target)


class BalancedCE_MSELoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(BalancedCE_MSELoss, self).__init__()
        self.ce = SoftTargetCrossEntropy()
        self.mse = SoftmaxMSELoss()
        self.alpha = alpha

    def forward(self, x, target):
        return self.mse(x, target) + self.alpha * self.ce(x, target)


def get_id(fname):
    pat = re.compile("(img.+).jpg")
    result = pat.search(fname)
    return result.group(1)


def process(line):
    token = line.strip().split()
    return token[0], [float(x) / 100.0 for x in token[1:]]


def include_imgs(d):
    img_files = glob.glob(f"{d}/*.jpg")
    if len(img_files) > 0:
        return True
    else:
        return False


def load_data(target):
    img_files = glob.glob(f"{target}/*.jpg")
    file_heads = [get_id(fname) for fname in img_files]
    ydict = {}
    yfile = glob.glob(f"{target}/*.txt")[0]
    with open(yfile, "r") as f:
        names = f.readline().strip().split()
        for line in f:
            k, v = process(line)
            ydict[k] = v
    img_labels = [ydict[h] for h in file_heads]
    return img_files, img_labels, names


def auto_brightness(img, min_brightness=0.6, max_value=255):
    brightness = np.sum(img) / (max_value * img.shape[0] * img.shape[1])
    ratio = brightness / min_brightness
    bright_img = cv2.convertScaleAbs(img, alpha=1 / ratio, beta=0)
    return bright_img


@dataclass
class Data:
    train_imgs: list
    train_labels: list
    valid_imgs: list
    valid_labels: list
    test_imgs: list
    test_labels: list
    names: list


@dataclass
class Prediction:
    imgs: list
    y_true: np.ndarray
    y_pred: np.ndarray
    outdim: int
    names: list


class MyDataset(Dataset):
    def __init__(
        self, img_paths, labels, transform=None, auto_brightness=True, grayscale=False
    ):
        self.transform = transform
        self.img_paths = img_paths
        self.labels = labels
        self.auto_brightness = auto_brightness
        self.grayscale = grayscale

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        label = self.labels[index]
        x = io.imread(img_path)
        if self.grayscale:
            x = rgb2gray(x)
        if self.auto_brightness:
            x = auto_brightness(x)
        if self.transform:
            augmented = self.transform(image=x)
            x = augmented["image"]

        return x, torch.tensor(label)


class CustomModel(nn.Module):
    def __init__(self, timm_model, gpool, head):
        super(CustomModel, self).__init__()
        self.backbone = timm_model
        self.gpool = gpool
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.forward_features = self.backbone.forward_features
        self.head = head

    def forward(self, x):
        f = self.forward_features(x)
        if hasattr(self.backbone, "global_pool"):
            if self.gpool:
                f = self.gpool(f)
            else:
                f = self.backbone.global_pool(f)
        y = self.head(f)
        return y


def create_net(
    model_name,
    head="bestfitting",
    concat_pool=False,
    outdim=1,
    softmax=True,
    in_chans=3,
    pretrained=True
):
    model_timm = timm.create_model(model_name, pretrained=pretrained, in_chans=in_chans)
    num_ftrs = model_timm.num_features
    if concat_pool:
        neck = SelectAdaptivePool2d(output_size=1, pool_type="catavgmax", flatten=True)
        num_ftrs *= 2
    else:
        neck = SelectAdaptivePool2d(output_size=1, pool_type="avg", flatten=True)
    if head == "bestfitting":
        layers = [
            nn.BatchNorm1d(num_ftrs),
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, int(num_ftrs / 2)),
            nn.ReLU(),
            nn.BatchNorm1d(int(num_ftrs / 2)),
            nn.Dropout(p=0.5),
            nn.Linear(int(num_ftrs / 2), outdim),
        ]
    elif head == "bn_linear":
        layers = [
            nn.BatchNorm1d(num_ftrs),
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, outdim),
        ]
    else:
        layers = [nn.Dropout(p=0.5), nn.Linear(num_ftrs, outdim)]
    if softmax:
        layers.append(nn.Softmax(dim=-1))
    clf = nn.Sequential(*layers)
    model = CustomModel(model_timm, neck, clf)

    return model


def plot_image(images, img_names, y_true, y_pred, outdim, m=4, s=4):
    n = math.ceil(len(images) / m)
    w = images[0].shape[1]

    fig = plt.figure(figsize=(s * m, s * n))
    grid = ImageGrid(fig, 111, nrows_ncols=(n, m), axes_pad=0.1)

    for ax, img, name, true, pred in zip(grid, images, img_names, y_true, y_pred):
        str_true = " ".join([f"{100.0*true[j]:.2f}%" for j in range(outdim)])
        str_pred = " ".join([f"{100.0*pred[j]:.2f}%" for j in range(outdim)])

        ax.set_axis_off()
        ax.imshow(np.vstack([np.zeros((220, w, 3), dtype=int), img]))
        ax.text(
            0.02,
            0.97,
            f"true {str_true}",
            ha="left",
            va="center",
            size=10,
            color="white",
            fontname="monospace",
            transform=ax.transAxes,
        )
        ax.text(
            0.02,
            0.92,
            f"pred {str_pred}",
            ha="left",
            va="center",
            size=10,
            color="yellow",
            fontname="monospace",
            transform=ax.transAxes,
        )
        ax.text(
            0.02,
            0.85,
            name[:20] + "..",
            ha="left",
            va="center",
            size=10,
            color="white",
            fontname="monospace",
            transform=ax.transAxes,
        )

    for i in range(m * n - len(images)):
        grid[m * n - i - 1].set_axis_off()


def csv_export(pred, fname="out.csv"):
    imgs, y_true, y_pred, names = pred.imgs, pred.y_true, pred.y_pred, pred.names
    df = pd.DataFrame(columns=['img'] + [f'{o}_{c}' for o in names for c in ['true', 'pred']])
    for i, img_path in enumerate(imgs):
        img_name = os.path.basename(img_path)
        df.loc[i] = [img_name] + [y[i,j] for j in range(len(names)) for y in [y_true, y_pred]]
    df.to_csv(fname)
    print(f'csv_export to {fname}')


def print_out_images(images, y_true, y_pred, outdim, fname="out.pdf", img_per_page=16):
    with PdfPages(fname) as pdf:
        n_pages = math.ceil(len(images) / img_per_page)
        for i in trange(n_pages):
            igrid = [
                io.imread(j) for j in images[i * img_per_page : (i + 1) * img_per_page]
            ]
            names = [
                os.path.basename(j)
                for j in images[i * img_per_page : (i + 1) * img_per_page]
            ]
            itrue = [j for j in y_true[i * img_per_page : (i + 1) * img_per_page]]
            ipred = [j for j in y_pred[i * img_per_page : (i + 1) * img_per_page]]
            plot_image(igrid, names, itrue, ipred, outdim)
            pdf.savefig(bbox_inches="tight")
            plt.close()


def train_step(
    model, device, train_loader, criterion, optimizer, scheduler, epoch, grad_accum
):
    model.train()
    train_loss = 0
    optimizer.zero_grad()
    for i, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item()
        loss.backward()
        if (i + 1) % grad_accum == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    train_loss /= len(train_loader)
    return train_loss


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
    test_loss /= len(test_loader)
    return test_loss


def calc_pred(model, device, test_loader, criterion, softmax):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            if softmax == False:
                output = output.softmax(dim=-1)
            y_true.extend(target.detach().cpu().tolist())
            y_pred.extend(output.detach().cpu().tolist())
    return np.array(y_true), np.array(y_pred)


def tta_predict(model, device, dataset, transform, num_tta, softmax):
    y_true = []
    y_pred = []

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for img, label in tqdm(dataset):
            images = torch.stack(
                [transform(image=img)["image"] for i in range(num_tta)]
            )
            images = images.to(device)
            out = model(images)
            pred_out = out.mean(dim=0).detach().cpu()
            true_out = label
            if softmax == False:
                pred_out = pred_out.softmax(dim=-1)
            y_pred.append(pred_out.numpy())
            y_true.append(true_out.numpy())

    return np.array(y_true), np.array(y_pred)


def xy_plot(ax, y_true, y_pred, **kwargs):
    ax.plot((0, 1), (0, 1), ls="--", c="0.5")
    ax.scatter(y_true, y_pred, **kwargs)
    ax.set_xlabel("groundtruth")
    ax.set_ylabel("predicted")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")


def plot_prediction(pred, fnames, col):
    imgs = pred.imgs
    y_true = pred.y_true
    y_pred = pred.y_pred
    outdim = pred.outdim
    ynames = pred.names

    plt.ioff()
    fig, ax = plt.subplots(1, outdim, figsize=(4 * outdim, 4))
    for i, asub in enumerate(ax):
        xy_plot(asub, y_true[:, i], y_pred[:, i], color=col, alpha=0.5)
        asub.set_title(ynames[i])
    plt.suptitle(f"individual ({outdim} outputs)")
    plt.savefig(fnames[0])
    plt.close(fig)

    plt.ioff()
    fig, ax = plt.subplots()
    xy_plot(ax, y_true.flatten(), y_pred.flatten(), color=col, alpha=0.5)
    ax.set_title("all")
    plt.savefig(fnames[1])
    plt.close(fig)
