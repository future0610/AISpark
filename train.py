from datasets.Spark import Spark
from torchvision import transforms as T
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, random_split
import cv2
import numpy as np
import torch
from models.TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_Seg, CONFIGS as configs
import math
import torch.nn as nn
import torch.optim as optimizer
from loss.loss import *
from trainer.Trainer import Trainer
import os
import sys
import random
import torch.backends.cudnn as cudnn

if __name__ == "__main__":
    SEED = 0

    pwd = os.path.abspath(os.path.join(__file__, os.path.pardir))
    os.chdir(pwd)
    ROOT = os.path.abspath(os.path.join(pwd, "data/"))

    BATCH = 32
    EPOCH = 100
    NUM_WORKERS = 0
    LR = 0.001
    WEIGHT_DECAY = 0.09

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(SEED)

    def init_kaiming(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data)
            if m.bias != None:
                nn.init.constant_(m.bias.data, 0)

    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name()}")
    else:
        print(f"Device: cpu")

    transform = alb.Compose([
        alb.HorizontalFlip(p = 0.5),
        alb.VerticalFlip(p = 0.5),
        alb.Rotate(limit = 90, p = 0.5, border_mode = cv2.BORDER_REPLICATE),
        ToTensorV2(transpose_mask = True)
    ])

    dataSet = Spark(ROOT, train = True, transform = transform)
    # dataSet.scan()

    testSet = Spark(ROOT, train = False, transform = transform)
    # testSet.scan()

    L = len(dataSet)
    train_size = int(0.7 * L)
    val_size = L - train_size
    train_data, val_data = random_split(dataSet, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size = BATCH, shuffle = False, num_workers = NUM_WORKERS)
    if val_size > 0:
        val_loader = DataLoader(val_data, batch_size = BATCH, shuffle = False,  num_workers = NUM_WORKERS)
    else:
        val_loader = None
    test_loader = DataLoader(testSet, batch_size = BATCH, shuffle = False,  num_workers = NUM_WORKERS)

    # config_vit = configs["R50-ViT-B_16"]
    # config_vit.n_classes = 1
    # config_vit.patches.grid = (int(256 / 16), int(256 / 16))
    # model = ViT_Seg(config_vit, img_size = 256, num_classes = 1)

    # from models.model import Custom

    # model = Custom(3)
    from models.Unet import UNet
    model = UNet(3)

    model = model.to(device)
    model.apply(init_kaiming)
    optim = optimizer.Adam(model.parameters(), lr = 0.001)

    lr_lambda = lambda epoch: math.exp(-1 * (math.sin(10 * epoch * math.pi / 180) ** 2) * math.sqrt(epoch))
    scheduler = optimizer.lr_scheduler.LambdaLR(optimizer = optim, lr_lambda = lr_lambda)

    loss = BCEmIoULoss(1)
    eval_metric = mIoU()

    trainer = Trainer(
        model = model,
        optimizer = optim,
        loss_func = loss,
        eval_metric = eval_metric,
        device = device,
        scheduler = scheduler,
        log_dir = "runs",
        epoch = EPOCH,
        batch = BATCH
    )

    trainer.run(train_loader, val_loader)
    trainer.test(test_loader)