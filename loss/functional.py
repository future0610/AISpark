import numpy as np
import torch

# 픽셀 정확도를 계산 metric
def pixel_accuracy(y_true, y_pred):
    sum_n = np.sum(np.logical_and(y_pred, y_true))
    sum_t = np.sum(y_true)
 
    if (sum_t == 0):
        pixel_accuracy = 0
    else:
        pixel_accuracy = sum_n / sum_t
    return pixel_accuracy 

def DICECoeff(x: torch.Tensor, y: torch.Tensor, eps = 1e-8):
    GT = y.sum(dim = (2, 3))
    PD = x.sum(dim = (2, 3))
    IT = (y * x).sum(dim = (2, 3)) + eps
    Union = GT + PD + eps
    
    dc = ((2 * IT) / Union)
    return dc.mean()

def BCELoss2d(x, y, weight = 1):
    x = torch.sigmoid(x)
    eps = 1e-8
    l = - weight * ((y * torch.log(x + eps) + (1 - y) * torch.log(1 - x + eps)))

    return l.mean()

def mIoU(x, y, threshold = 0.5, eps = 1e-8):
    u = IoU(x = x, y = y, threshold = threshold, eps = eps)
    v = IoU(x = -1 * x, y = 1 - y, threshold = threshold, eps = eps)

    return (u + v) / 2

def IoU(x, y, threshold = 0.5, eps = 1e-8):
    union = (x + y).sum(dim = (-2, -1)) + eps
    it = (x * y).sum(dim = (-2, -1))
    union = union - it

    return (it / union).mean()

def BCE_IoU_Loss(x, y, threshold, weight = 1, eps = 1e-8):
    bce_loss = BCELoss2d(x, y, weight)
    x = torch.sigmoid(x)
    iou = IoU(x, y, eps)

    return bce_loss + 1 - iou

def BCE_Dice_Loss(x, y, weight = 1, eps = 1e-8):
    bce_loss = BCELoss2d(x, y, weight)
    x = torch.sigmoid(x)
    dice = DICECoeff(x, y, eps)

    return bce_loss + 1 - dice