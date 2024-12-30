from datasets.Spark import Spark
from torchvision import transforms as T
import albumentations as alb
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import cv2
import numpy as np
import torch
from models.TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_Seg, CONFIGS as configs
import math
import torch.nn as nn
import torch.optim as optimizer
from loss.functional import *
import random
import torch.backends.cudnn as cudnn

seed = 1
deterministic = True

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
if deterministic:
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
     
def init_kaiming(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data)
        if m.bias != None:
            nn.init.constant_(m.bias.data, 0)


ROOT = "data/"

BS = 16
EPOCH = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = alb.Compose([
    alb.HorizontalFlip(p = 0.5),
    alb.VerticalFlip(p = 0.5),
    alb.Rotate(limit = 90, p = 0.5, border_mode=cv2.BORDER_REPLICATE),
    ToTensorV2(transpose_mask = True)
])

dataSet = Spark(ROOT, train = True, transform = transform)
testSet = Spark(ROOT, train = False, transform = transform)


L = len(dataSet)
train_size = int(0.7 * L)
val_size = L - train_size
train_data, val_data = random_split(dataSet, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size = BS, shuffle = False)
val_loader = DataLoader(val_data, batch_size = BS, shuffle = False)
test_loader = DataLoader(testSet, batch_size = BS, shuffle = False)

config_vit = configs["R50-ViT-B_16"]
config_vit.n_classes = 1
config_vit.patches.grid = (int(256 / 16), int(256 / 16))
model = ViT_Seg(config_vit, img_size = 256, num_classes = 1)
from models.model import Custom

# model = Custom(3)
model = model.to(device)
model.apply(init_kaiming)
optim = optimizer.Adam(model.parameters(), lr = 0.01)

print(model.transformer.embeddings.patch_embeddings.weight.data)

# process_dict = stable.load('./checkpoint/')
# on_epoch = process_dict['epoch']
# model.load_state_dict(process_dict['model'])
# optim.load_state_dict(process_dict['optimizer'])

lr_lambda = lambda epoch: 1#math.exp(-1 * math.tan(10 * epoch * math.pi / 180)) ** 2
sceduler = optimizer.lr_scheduler.LambdaLR(optimizer = optim, lr_lambda = lr_lambda)
# sceduler.last_epoch = on_epoch

barLen = 20
threshold = 0.3
for epoch in range(EPOCH):
    # if epoch <= on_epoch:
    #     continue
    model.train()
    cost = 0
    size = 0
    for batch_idx, (img, mask) in enumerate(train_loader):
        optim.zero_grad()
        plt.imshow(img[0].permute(1, 2, 0).numpy())
        plt.show()
        
        plt.imshow(mask[0][0].numpy())
        plt.show()
        img, mask = img.to(device), mask.to(device)
        predicted = model(img)

        loss = BCELoss2d(predicted, mask, 1) + 1 - soft_IoU(predicted, mask)
        cost += loss.item()
        size += img.size(0)

        loss.backward()
        print(f"\rEPOCH(lr: {optim.param_groups[0]['lr']:.4f}) {epoch + 1:{len(str(EPOCH))}d}/{EPOCH} {'Train Process':35s}:[{'=' * int(barLen * (size / train_size)) + '>':{barLen + 1}s}][{size:{len(str(train_size))}d}/{train_size}({100 * (size / train_size):6.2f}%)] Loss: {cost / (batch_idx + 1):11.4f}", end = "")
        print()
        optim.step()
        if batch_idx >= 5:
            break
    continue
    print()
    size = 0
    avg_mIoU = 0
    dataLen = val_size
    with torch.no_grad():
        model.eval()
        for batch_idx, (img, mask) in enumerate(val_loader):
            img, mask = img.to(device), mask.to(device)

            predicted = model(img)

            mIoU = IoU(predicted, mask, 0.5)
            avg_mIoU += mIoU
            size += img.size(0)
            print(f"\rEPOCH(lr: {optim.param_groups[0]['lr']:.4f}) {epoch + 1:{len(str(EPOCH))}d}/{EPOCH} {'Validation Process':35s}:[{'=' * int(barLen * (size / dataLen)) + '>':{barLen + 1}s}][{size:{len(str(val_size))}d}/{val_size}({100 * (size / dataLen):6.2f}%)] DC Score: {avg_mIoU / (batch_idx + 1):7.4f}", end = "")
    sceduler.step()
    print()