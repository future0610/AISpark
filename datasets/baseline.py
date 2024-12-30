import rasterio
import threading
import numpy as np
from sklearn.utils import shuffle as shuffle_lists
from sklearn.model_selection import train_test_split
import pandas as pd
import os

MAX_PIXEL_VALUE = 65535 # 이미지 정규화를 위한 픽셀 최대값
RANDOM_STATE = 42 # seed 고정
# 데이터 위치
IMAGES_PATH = './data/train_img/'
MASKS_PATH = './data/train_mask/'
BATCH_SIZE = 64 # batch size 지정

train_meta = pd.read_csv('./data/train_meta.csv')
test_meta = pd.read_csv('./data/test_meta.csv')

class threadsafe_iter:
    """
    데이터 불러올떼, 호출 직렬화
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g

def get_img_arr(path):
    img = rasterio.open(path).read().transpose((1, 2, 0))    
    img = np.float32(img)/MAX_PIXEL_VALUE
    
    return img

def get_img_762bands(path):
    img = rasterio.open(path).read((7,6,2)).transpose((1, 2, 0))    
    img = np.float32(img)/MAX_PIXEL_VALUE
    
    return img
    
def get_mask_arr(path):
    img = rasterio.open(path).read().transpose((1, 2, 0))
    seg = np.float32(img)
    return seg



@threadsafe_generator
def generator_from_lists(images_path, masks_path, batch_size=32, shuffle = True, random_state=None, image_mode='10bands'):
   
    images = []
    masks = []

    fopen_image = get_img_arr
    fopen_mask = get_mask_arr

    if image_mode == '762':
        fopen_image = get_img_762bands

    i = 0 
    # 데이터 shuffle
    while True:
        
        if shuffle:
            if random_state is None:
                images_path, masks_path = shuffle_lists(images_path, masks_path)
            else:
                images_path, masks_path = shuffle_lists(images_path, masks_path, random_state= random_state + i)
                i += 1 


        for img_path, mask_path in zip(images_path, masks_path):
            
            img = fopen_image(img_path)
            mask = fopen_mask(mask_path)
            images.append(img)
            masks.append(mask)

            if len(images) >= batch_size:
                yield (np.array(images), np.array(masks))
                images = []
                masks = []

if __name__ == "__main__":
    # train : val = 8 : 2 나누기
    x_tr, x_val = train_test_split(train_meta, test_size=0.2, random_state=RANDOM_STATE)
    print(len(x_tr), len(x_val))
    # train : val 지정 및 generator
    images_train = [os.path.join(IMAGES_PATH, image) for image in x_tr['train_img'] ]
    masks_train = [os.path.join(MASKS_PATH, mask) for mask in x_tr['train_mask'] ]

    images_validation = [os.path.join(IMAGES_PATH, image) for image in x_val['train_img'] ]
    masks_validation = [os.path.join(MASKS_PATH, mask) for mask in x_val['train_mask'] ]

    train_generator = generator_from_lists(images_train, masks_train, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, image_mode="762")
    validation_generator = generator_from_lists(images_validation, masks_validation, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, image_mode="762")