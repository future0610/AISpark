import torch
from torchvision import transforms as T

def center_sampling(img_size, target_size) -> tuple:
    y, x = target_size
    h, w = img_size
    i = torch.randint(y // 2 - (y + 1) % 2, h - y // 2).tolist()
    j = torch.randint(x // 2 - (x + 1) % 2, w - x // 2).tolist()

    return (i, j)

def size_sampling(img_size, target_size, decrease_rate = 1):
    y, x = target_size
    h, w = img_size
    target_h = torch.randint(int(decrease_rate * y), h + 1)
    target_w = torch.randint(int(decrease_rate * x), w + 1)

    return (target_h, target_w)

def crop(img: torch.Tensor, center, size):
    i, j = size
    y, x = center
    
    return img[x - (i // 2 - (i + 1) % 2) : y + i // 2 + 1, x - j // 2 + (j + 1) % 2 : x + j // 2 + 1, :]

class Compose(T.Compose):
    def __init__(self, transforms):
        super().__init__(transforms)

    def __call__(self, img, masks = None):
        for t, tm in self.transforms:
            img, masks = t(img), tm(masks)

        return img, masks

class Empty:
    def __init__(self) -> None:
        pass

    def __call__(self, x):
        return x
    
class Cropper:
    def __init__(self, target_size, mode = 0, rate = 1):
        """
        if mode == 0:
            이미지와 마스크의 중심부를 일정한 크기로 자름
        elif mode == 1:
            이미지와 마스크의 중심부를 임의의 크기로 자름
        elif mode == 2:
            이미지와 마스크 위의 임의의 위치를 중심으로 일정한 크기로 자름
        elif mode == 3:
            이미지와 마스크 위의 임의의 위치를 중심으로 임의의 크기로 자름
        """
        self.target_size = target_size
        self.mode = mode
        self.resize = T.Resize(self.target_size, antialias = True)

    def __call__(self, img, mask) -> torch.Any:
        if self.mode == 0:
            h, w = img.shape[-2], img.shape[-1]
            i, j = h // 2, w // 2
            img = crop(img, (i, j), self.target_size)
            mask = crop(mask, (i, j), self.target_size)

# class Cropper:
#     def __init__(self, target_size, custom_size = None, mode = "RandomCenter", batch_same = False, label = False) -> None:
#         self.ts = target_size
#         self.mode = mode
#         self.custom_size = custom_size
#         self.resizer = T.Resize(self.ts, antialias = True)
#         self.batch_same = batch_same
#         self.label = label

#     def __call__(self, img: torch.Tensor, mask: torch.Tensor = None):
#         if self.mode == "RandomCenter":
#             target = torch.zeros(size = (img.size(0),)) + (img.shape[-1] - self.ts[0])
#             if not self.batch_same:
#                 i, j = self.randomCenter(img.shape, self.custom_size)
#                 img = self.constSizedCrop(img, (i, j))
#                 if mask != None:
#                     mask = self.constSizedCrop(mask, (i, j))
#             else:
#                 i, j = self.randomCenter(torch.Size([1]) + img.shape[1 :], self.custom_size)
#                 i = i[0]
#                 j = j[0]
#                 img = self.constSizedCenterCrop(img, (i, j), self.custom_size)
#                 if mask != None:
#                     mask = self.constSizedCenterCrop(mask, (i, j), self.custom_size)
            
#         elif self.mode == "Center":
#             target = torch.zeros(size = (img.size(0),)) + (img.shape[-1] - self.ts[0])
#             h, w = img.shape[-2], img.shape[-1]
#             i, j = h // 2, w // 2
            
#             img = self.constSizedCenterCrop(img, (i, j), self.custom_size)
#             if mask != None:
#                 mask = self.constSizedCenterCrop(mask, (i, j), self.custom_size)

#         elif self.mode == "RandomSize":
#             h, w = img.shape[-2], img.shape[-1]
            
#             if not self.batch_same:
#                 i, j = self.randomSize(img.shape)
#                 target = (img.shape[-1] - torch.tensor(i, dtype = torch.int64))
#                 img = self.randomSizedCrop(img, (h, w), (i, j))
#                 if mask != None:
#                     mask = self.randomSizedCrop(mask, (h, w), (i, j))
#             else:
#                 i, j = self.randomSize(torch.Size([1]) + img.shape[1 :])
#                 i = i[0]
#                 j = j[0]
#                 target = torch.tensor([i for _ in range(img.size(0))], dtype = torch.int64)
#                 img = self.constSizedCenterCrop(img, (h, w), (i, j))
#                 img = self.resizer(img)
#                 if mask != None:
#                     mask = self.constSizedCenterCrop(mask, (h, w), (i, j))
#                     mask = self.resizer(mask)
        
#         elif self.mode == "RandomSizeCenter":
#             if not self.batch_same:
#                 i, j = self.randomSize(img.shape)
#                 target = (img.shape[-1] - torch.tensor(i, dtype = torch.int64))
#                 img, mask = self.randomSizedCenterCrop(img, mask, (i, j))
#             else:
#                 i, j = self.randomSize(torch.Size([1]) + img.shape[1 :])
#                 i = i[0]
#                 j = j[0]
#                 target = torch.tensor([i for _ in range(img.size(0))], dtype = torch.int64)
#                 h, w = self.randomCenter(torch.Size([1]) + img.shape[1 :], (i, j))
#                 h = h[0]
#                 w = w[0]
#                 img = self.constSizedCenterCrop(img, (h, w), (i, j))
#                 img = self.resizer(img)
#                 if mask != None:
#                     mask = self.constSizedCenterCrop(mask, (h, w), (i, j))
#                     mask = self.resizer(mask)
        
#         elif self.mode == "Resize":
#             target = torch.zeros(size = (img.size(0),), dtype = torch.int64)
#             img = self.resizer(img)
#             if mask != None:
#                 mask = self.resizer(mask)
#         return img, mask

#     def randomSize(self, data : torch.Size) -> tuple:
#         batch_size = data[0]
#         h, w = data[-2], data[-1]
#         i = torch.randint(self.ts[0], min(h, w) + 1, size = (batch_size,)).tolist()

#         return (i, i.copy())

#     def randomCenter(self, data : torch.Size, size = None) -> tuple:
#         if size == None:
#             size = self.ts
#         batch_size = data[0]
#         h, w = data[-2], data[-1]

#         i = torch.randint(size[0] // 2 - (size[0] + 1) % 2, h - size[0] // 2, size = (batch_size,)).tolist()
#         j = torch.randint(size[1] // 2 - (size[1] + 1) % 2, w - size[1] // 2, size = (batch_size,)).tolist()

#         return (i, j)
    
#     def constSizedCenterCrop(self, img, center, size = None):
#         initSize = size
#         if size == None:
#             size = self.ts
#         i, j = size
#         x, y = center

#         cp = img[:, :, x - i // 2 + (i + 1) % 2 + 1 : x + i // 2 + 2, y - j // 2 + (j + 1) % 2 + 1 : y + j // 2 + 2]
#         if initSize != None:
#             cp = self.resizer(cp)

#         return cp

#     def constSizedCrop(self, img: torch.Tensor, center: tuple, size = None):
#         initSize = size
#         if size == None:
#             size = self.ts
#         i, j = size
#         x, y = center
#         for it in range(img.size(0)):
#             cp = img[it, :, x[it] - (i // 2 - (i + 1) % 2) : x[it] + i // 2 + 1, y[it] - j // 2 + (j + 1) % 2 : y[it] + j // 2 + 1].unsqueeze(0)
#             if initSize != None:
#                 cp = self.resizer(cp)
#             if it == 0:
#                 cropped = cp
#             else:
#                 cropped = torch.cat([cropped, cp], dim = 0)
        
#         return cropped
    
#     def randomSizedCenterCrop(self, img: torch.Tensor, mask: torch.Tensor = None, size = None):
#         i, j = size
#         for it in range(img.size(0)):
#             x, y = self.randomCenter(torch.Size([1]) + img.shape[1 :], (i[it], j[it]))
#             x = x[0]
#             y = y[0]
#             # print((i[it], j[it]), (x, y))
#             cp_img = img[it, :, x - (i[it] // 2 - (i[it] + 1) % 2) : x + i[it] // 2 + 1, y - j[it] // 2 + (j[it] + 1) % 2 : y + j[it] // 2 + 1].unsqueeze(0)
#             cp_img = self.resizer(cp_img)
#             try:
#                 cropped_img = torch.cat([cropped_img, cp_img], dim = 0)
#             except:
#                 cropped_img = cp_img
#             if mask != None:
#                 cp_mask = mask[it, :, x - (i[it] // 2 - (i[it] + 1) % 2) : x + i[it] // 2 + 1, y - j[it] // 2 + (j[it] + 1) % 2 : y + j[it] // 2 + 1].unsqueeze(0)
#                 cp_mask = self.resizer(cp_mask)
#                 try:
#                     cropped_mask = torch.cat([cropped_mask, cp_mask], dim = 0)
#                 except:
#                     cropped_mask = cp_mask
        
#         return cropped_img, cropped_mask
    
#     def randomSizedCrop(self, img: torch.Tensor, center, size = None):
#         i, j = size

#         x, y = center
#         for it in range(img.size(0)):
#             cp = img[it, :, x - (i[it] // 2 - (i[it] + 1) % 2) : x + i[it] // 2 + 1, y - j[it] // 2 + (j[it] + 1) % 2 : y + j[it] // 2 + 1].unsqueeze(0)
#             cp = self.resizer(cp)
#             try:
#                 cropped = torch.cat([cropped, cp], dim = 0)
#             except:
#                 cropped = cp

#         return cropped

