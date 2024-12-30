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

class BCELoss2d:
    def __init__(self, weight):
        self.weight = weight
    
    def __call__(self, x, y):
        x = torch.sigmoid(x)
        eps = 1e-8
        l = - self.weight * ((y * torch.log(x + eps) + (1 - y) * torch.log(1 - x + eps)))

        return l.mean()

class DiffIoU:
    def __init__(self, threshole = 0.5, eps = 1e-8) -> None:
        self.threshold = threshole
        self.eps = eps

    def __call__(self, x, y):
        union = (x + y).sum(dim = (-2, -1)) + self.eps
        it = (x * y).sum(dim = (-2, -1))
        union = union - it

        area1 = union
        u = it / union
        
        x = 1 - 1 * x
        y = 1 - y
        union = (x + y).sum(dim = (-2, -1)) + self.eps
        it = (x * y).sum(dim = (-2, -1))
        union = union - it

        area2 = union
        v = it / union

        return ((area2 * u + area1 * v) / (area1 + area2)).mean()

class BCEDiffIoU:
    def __init__(self, threshold, weight = 1, eps = 1e-8) -> None:
        self.threshold = threshold
        self.weight = weight
        self.eps = eps
        self.BCELoss2d = BCELoss2d(self.weight)
        self.DiffIoU = DiffIoU(self.threshold, self.eps)

    def __call__(self, x, y):
        bce_loss = self.BCELoss2d(x, y)
        diffIoU = self.DiffIoU(x, y)

        return bce_loss + 1 - diffIoU
    
    def eval(self, x, y):
        return self.DiffIoU(x, y)

class mIoU:
    def __init__(self, eps = 1e-8):
        self.eps = eps
        self.IoU = IoU(self.eps)
    
    def __call__(self, x, y):
        u = self.IoU(x, y)
        v = self.IoU(-1 * x, 1 - y)

        return (u + v) / 2

class mIoU:
    def __init__(self, eps = 1e-8):
        self.eps = eps
        self.IoU = IoU(self.eps)

    def __call__(self, x, y):
        u = self.IoU(x, y)
        v = self.IoU(1 - x, 1 - y)

        return (u + v) / 2

class IoU:
    def __init__(self, eps = 1e-8):
        self.eps = eps

    def __call__(self, x, y):
        yarea = y.sum(dim = (-2, -1))
        xarea = x.sum(dim = (-2, -1))
        tp = (x * y).sum(dim = (-2, -1))
        union = xarea + yarea - tp
        
        iou = tp / union
        # iou[union == 0] = 1
        iou = iou[union != 0]

        return iou.mean()

class BCEmIoULoss:
    def __init__(self, weight = 1) -> None:
        self.weight = weight
        self.BCELoss2d = BCELoss2d(self.weight)
        self.mIoU = mIoU()

    def __call__(self, x, y):
        bce_loss = self.BCELoss2d(x, y)
        x = torch.sigmoid(x)
        miou = self.mIoU(x, y)

        return bce_loss + 1 - miou
    
    def eval(self, x, y):
        return self.mIoU(x, y)

class BCEIoULoss:
    def __init__(self, weight = 1) -> None:
        self.weight = weight
        self.BCELoss2d = BCELoss2d(self.weight)
        self.IoU = IoU()

    def __call__(self, x, y):
        bce_loss = self.BCELoss2d(x, y)
        x = torch.sigmoid(x)
        iou = self.IoU(x, y)

        return bce_loss + 1 - iou
    
    def eval(self, x, y):
        return self.IoU(x, y)

class BCEDiceLoss:
    def __init__(self, weight = 1) -> None:
        self.weight = weight
        self.BCELoss2d = BCELoss2d(self.weight)

    def __call__(self, x, y):
        bce_loss = self.BCELoss2d(x, y)
        x = torch.sigmoid(x)
        dice = DICECoeff(x, y)

        return bce_loss + 1 - dice
    
    def eval(self, x, y):
        return DICECoeff(x, y)