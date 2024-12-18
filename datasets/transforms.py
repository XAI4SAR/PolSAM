import warnings
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2
import numpy as np
import numbers
import random
import collections
from collections.abc import Iterable


def get_data_info(dataset):
    if dataset == 'coarse':
        classes = 6
        colormap = np.array([[0, 0, 0],[100, 149, 237], [0, 30, 100],[255, 215, 0], [50, 205, 50],[0, 128, 0]])
        class_names = ['BK', 'urban surfaces', 'water', 'building', 'barren land', 'vegetation']
        path_train = r'datasets/PhySAR-Seg-1/PhySAR-Seg-1_train.csv'
        path_val = r'datasets/PhySAR-Seg-1/PhySAR-Seg-1_val.csv'
        path_test = r'datasets/PhySAR-Seg-1/PhySAR-Seg-1_test16.csv'
        weight = None
    elif dataset == 'SF':
        classes = 8
        colormap = np.array([(0, 0, 0), (0, 128, 0), (0, 255, 0),  (0, 255, 255), (255, 0, 255), (255, 0, 0), (255, 255, 0), (0, 0, 255)])
        class_names = ['BK', 'mountain', 'vegetation', 'developed', 'high-density urban', 'low-density urban', 'irregular urban', 'water']
        # path_train = r'datasets/PhySAR-Seg-2/pauli_Halpha_T6_train.csv'
        # path_val = r'datasets/PhySAR-Seg-2/pauli_Halpha_T6_val.csv'
        # path_train = r'datasets/PhySAR-Seg-2/pauli_T9_HT12_train.csv'
        # path_val = r'datasets/PhySAR-Seg-2/pauli_T9_HT12_val.csv'
        # path_train = r'datasets/PhySAR-Seg-2/pauli_scat_label_npy_train.csv'
        # path_val = r'datasets/PhySAR-Seg-2/pauli_scat_label_npy_val.csv'
        path_train = r'datasets/PhySAR-Seg-2/PhySAR-Seg-2_train.csv'
        path_val = r'datasets/PhySAR-Seg-2/PhySAR-Seg-2_val.csv'
        path_test = None
        weight = [0.0819,0.0642,0.1608,0.2339,0.3156,0.0666,0.0666,0.0104]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return {'classes':classes, 'colormap':colormap, 'class_names': class_names, 'weight':weight,
            'path':{'path_train':path_train, 'path_val':path_val, 'path_test':path_test}}


def is_single_channel(image_path):
        image = Image.open(image_path)
        if image.mode == 'L':
            return True
        else:
            return False

# Generate the corresponding color according to the given number of categories, the color is fixed
def get_class_colors(N):  # N=classes
        def uint82bin(n, count=8):
            return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        class_colors = cmap.tolist()
        return class_colors


def get_2dshape(shape, *, zero=True):
    if not isinstance(shape, Iterable):
        shape = int(shape)
        shape = (shape, shape)
    else:
        h, w = map(int, shape)
        shape = (h, w)
    if zero:
        minv = 0
    else:
        minv = 1

    assert min(shape) >= minv, 'invalid shape: {}'.format(shape)
    return shape

def random_crop_pad_to_shape(img, crop_pos, crop_size, pad_label_value):
    h, w = img.shape[:2]
    start_crop_h, start_crop_w = crop_pos
    assert ((start_crop_h < h) and (start_crop_h >= 0))
    assert ((start_crop_w < w) and (start_crop_w >= 0))

    crop_size = get_2dshape(crop_size)
    crop_h, crop_w = crop_size

    img_crop = img[start_crop_h:start_crop_h + crop_h,
               start_crop_w:start_crop_w + crop_w, ...]

    img_, margin = pad_image_to_shape(img_crop, crop_size, cv2.BORDER_CONSTANT,
                                      pad_label_value)

    return img_, margin

def generate_random_crop_pos(ori_size, crop_size):
    ori_size = get_2dshape(ori_size)
    h, w = ori_size

    crop_size = get_2dshape(crop_size)
    crop_h, crop_w = crop_size

    pos_h, pos_w = 0, 0

    if h > crop_h:
        pos_h = random.randint(0, h - crop_h + 1)

    if w > crop_w:
        pos_w = random.randint(0, w - crop_w + 1)

    return pos_h, pos_w

def pad_image_to_shape(img, shape, border_mode, value):
    margin = np.zeros(4, np.uint32)
    shape = get_2dshape(shape)
    pad_height = shape[0] - img.shape[0] if shape[0] - img.shape[0] > 0 else 0
    pad_width = shape[1] - img.shape[1] if shape[1] - img.shape[1] > 0 else 0

    margin[0] = pad_height // 2
    margin[1] = pad_height // 2 + pad_height % 2
    margin[2] = pad_width // 2
    margin[3] = pad_width // 2 + pad_width % 2

    img = cv2.copyMakeBorder(img, margin[0], margin[1], margin[2], margin[3],
                             border_mode, value=value)

    return img, margin

def random_mirror_single(img, gt):
    if random.random() >= 0.5:
        img = cv2.flip(img, 1)
        gt = cv2.flip(gt, 1)

    return img, gt,

def random_mirror(rgb, gt, modal_x):
    if random.random() >= 0.5:
        rgb = cv2.flip(rgb, 1)
        gt = cv2.flip(gt, 1)
        modal_x = cv2.flip(modal_x, 1)

    return rgb, gt, modal_x

def random_rotation_single(img, gt):
    angle = random.random() * 20 - 10
    h, w = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    img = cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
    gt = cv2.warpAffine(gt, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)

    return img, gt

def random_rotation(img, gt, mx):
    angle = random.random() * 20 - 10
    h, w = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    img = cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mx = cv2.warpAffine(mx, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    gt = cv2.warpAffine(gt, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)

    return img, gt, mx

def random_gaussian_blur(img):
    gauss_size = random.choice([1, 3, 5, 7])
    if gauss_size > 1:
        # do the gaussian blur
        img = cv2.GaussianBlur(img, (gauss_size, gauss_size), 0)

    return img

def center_crop(img, shape):
    h, w = shape[0], shape[1]
    y = (img.shape[0] - h) // 2
    x = (img.shape[1] - w) // 2
    return img[y:y + h, x:x + w]

def random_crop_single(img, gt, size):
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        size = size

    h, w = img.shape[:2]
    crop_h, crop_w = size[0], size[1]

    if h > crop_h:
        x = random.randint(0, h - crop_h + 1)
        img = img[x:x + crop_h, :, :]
        gt = gt[x:x + crop_h, :]

    if w > crop_w:
        x = random.randint(0, w - crop_w + 1)
        img = img[:, x:x + crop_w, :]
        gt = gt[:, x:x + crop_w]

    return img, gt

def random_crop(img, mx, gt, crop_size):
    if isinstance(crop_size, numbers.Number):
        crop_size = (int(crop_size), int(crop_size))
    else:
        crop_size = crop_size

    img = Image.fromarray(img)
    resized_img = img.resize(crop_size, Image.BICUBIC)
    resized_img = np.array(resized_img)

    if mx is not None:
        mx = Image.fromarray(mx)
        resized_mx = mx.resize(crop_size, Image.BICUBIC)
        resized_mx = np.array(resized_mx)
    else:
        resized_mx = None

    if gt is not None:
        gt = Image.fromarray(gt)
        resized_gt = gt.resize(crop_size, Image.NEAREST)
        resized_gt = np.array(resized_gt)
    else:
        resized_gt = None

    return resized_img, resized_mx, resized_gt

def normalize(img, mean, std):
    img = img.astype(np.float64) / 255.0
    img = img - mean
    img = img / std
    return img