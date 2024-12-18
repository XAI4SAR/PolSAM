import random
import time

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms.functional import to_tensor
from datasets.transforms import  normalize


def csv_list_multimodel(path):
    img_path = pd.read_csv(path)
    images_list = list(img_path.iloc[:, 0])
    scats_list = list(img_path.iloc[:, 1])
    labels_list = list(img_path.iloc[:, 2])
    return images_list, scats_list, labels_list

def csv_list_SF_Ablation(path):
    img_path = pd.read_csv(path)
    images_list = list(img_path.iloc[:, 0])
    Halpha_list = list(img_path.iloc[:, 1])
    T6_list = list(img_path.iloc[:, 2])
    labels_list = list(img_path.iloc[:, 3])
    return images_list, Halpha_list, T6_list, labels_list


# class MydatasetSF_Ablation(Dataset):
#     def __init__(self, data_info, cfg, train=True, aug=True):
#         self.train = train
#         self.do_aug = aug
#         self.cfg = cfg
#         if self.train:
#             self.images, self.Has, self.T9s, self.labels = csv_list_SF_Ablation(data_info['path']['path_train'])
#         else:
#             self.images, self.Has, self.T9s, self.labels = csv_list_SF_Ablation(data_info['path']['path_val'])
#         print('Read ' + str(len(self.images)) + ' valid examples') if len(self.images) == len(self.Has) == len(self.labels) else None
        
#         self.tsf = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize([0.391, 0.258, 0.593], [0.144, 0.129, 0.192])])
#         if cfg.ablation == 'T9':
#             self.tsf2 = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize([1.90983932e-04, 5.89392301e-05, 5.09879095e-05, 4.51099445e-01, 9.49438299e-01,
#                                       4.59979472e-01, 0.719042, 0.302984, 0.332024],
#                                      [0.00149292, 0.00085947, 0.00092904, 0.00099348, 0.00047795, 0.00083098, 0.000484935, 0.000438965 , 0.00030162])])
#         elif cfg.ablation == 'H_alpha_T12':
#             self.tsf2 = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.539, 0.417, 0.337, 1.90983932e-04, 5.89392301e-05, 5.09879095e-05, 4.51099445e-01,
#                                       9.49438299e-01, 4.59979472e-01, 0.719042, 0.302984, 0.332024],
#                                      [0.232, 0.209, 0.074, 0.00149292, 0.00085947, 0.00092904, 0.00099348,
#                                       0.00047795, 0.00083098, 0.000484935, 0.000438965, 0.00030162])])

#     def __getitem__(self, idx):
#         image = self.images[idx]
#         label = self.labels[idx]
#         name = image.split('/')[-1]
#         image = Image.open(image).convert('RGB')
#         label = Image.open(label).convert('RGB')
#         if self.cfg.ablation == 'H_alpha_T12':
#             Ha = self.Has[idx]
#             mx = np.load(Ha)
#         elif self.cfg.ablation == 'T9':
#             T9 = self.T9s[idx]
#             mx = np.load(T9)
#         if self.train and self.do_aug:
#             image, label, mx= custom_random_affine(image, label, mx, havescat=True)
#         label = label_indices(dataset_SF='SF')(label)
#         label = torch.from_numpy(label).long()
#         mx = self.tsf2(mx).to(dtype=torch.float32)
#         return{
#             'image':self.tsf(image),
#             'scatrgb':mx,
#             'scatonehot':mx,
#             'label':label,
#             'name':name}

#     def __len__(self):
#         return len(self.images)


class MydatasetSF_Ablation(Dataset):
    def __init__(self, data_info, cfg, train=True, aug=True):
        self.train = train
        self.do_aug = aug
        self.cfg = cfg
        if self.train:
            self.images, self.Has, self.T6s, self.labels = csv_list_SF_Ablation(data_info['path']['path_train'])
        else:
            self.images, self.Has, self.T6s, self.labels = csv_list_SF_Ablation(data_info['path']['path_val'])
        print('Read ' + str(len(self.images)) + ' valid examples') if len(self.images) == len(self.Has) == len(self.labels) else None

        self.tsf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.361, 0.258, 0.563], [0.144, 0.129, 0.162])])
        if cfg.ablation == 'T6':
            self.tsf2 = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([1.90983932e-04, 5.89362301e-05, 5.09876065e-05, 4.51096445e-01, 6.49438296e-01, 4.56676472e-01],
                                     [0.00146262, 0.00085947, 0.00092604, 0.00096348, 0.00047795, 0.00083068])])
        elif cfg.ablation == 'H_alpha':
            self.tsf2 = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.539, 0.417, 0.337], [0.232, 0.209, 0.074])])

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        name = image.split('/')[-1]
        image = Image.open(image).convert('RGB')
        label = Image.open(label).convert('RGB')
        if self.cfg.ablation == 'H_alpha':
            Ha = self.Has[idx]
            mx = np.load(Ha)
        elif self.cfg.ablation == 'T6':
            T6 = self.T6s[idx]
            mx = np.load(T6)
        if self.train and self.do_aug:
            image, label, mx= custom_random_affine(image, label, mx, havescat=True)
        label = label_indices(dataset_SF='SF')(label)
        label = torch.from_numpy(label).long()
        mx = self.tsf2(mx).to(dtype=torch.float32)
        return{
            'image':self.tsf(image),
            'scatrgb':mx,
            'scatonehot':mx,
            'label':label,
            'name':name}

    def __len__(self):
        return len(self.images)


class MydatasetP(Dataset):
    def __init__(self, inputnum, data_info, cfg, bs=4, train=True, aug=True, onehot=False, npy=False):
        self.bs = bs
        self.train = train
        self.inputnum = inputnum
        self.onehot = onehot
        self.npy = npy
        self.data_info = data_info
        self.do_aug = aug
        self.cfg = cfg
        if self.train:
            self.images, self.scats, self.labels = csv_list_multimodel(data_info['path']['path_train'])
        else:
            self.images, self.scats, self.labels = csv_list_multimodel(data_info['path']['path_val'])
        print('Read ' + str(len(self.images)) + ' valid examples') if len(self.images) == len(self.scats) == len(self.labels) else None
        
        if self.cfg.dataset_name =='SF':
            self.tsf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.361, 0.258, 0.563], [0.144, 0.129, 0.162])])
            self.tsf_scat = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.245, 0.153, 0.395], [0.145, 0.111, 0.081])
            ])
        elif self.cfg.dataset_name == 'coarse':
            self.tsf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.440, 0.372, 0.442], [0.488, 0.424, 0.484])
            ])
            self.tsf_scat = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.289, 0.282, 0.154], [0.375, 0.460, 0.350])
            ])
        self.ten = transforms.ToTensor()
        self.tsff = transforms.Normalize([0.364, 0.327, 0.298], [0.405, 0.413, 0.362])

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        name = image.split('/')[-1]
        image = Image.open(image).convert('RGB')
        label = Image.open(label).convert('RGB')
        if self.do_aug:
            if self.inputnum == 1:
                if self.train:
                    image, label, _= custom_random_affine(image, label, label, havescat=False)
                label = label_indices(dataset_SF=self.cfg.dataset_name)(label)
                label = torch.from_numpy(label).long()
                return self.tsf(image), label
            if self.inputnum == 2:
                scat = self.scats[idx]
                name1 = scat.split('/')[-1]
                # scat = np.load(scat)
                scat = Image.open(scat).convert('RGB')
                if self.train:
                    image, label, scat= custom_random_affine(image, label, scat, havescat=True)
                label = label_indices(dataset_SF=self.cfg.dataset_name)(label)
                label = torch.from_numpy(label).long()
                scatrgb = self.tsf_scat(scat)
                imagea = np.zeros((6,512,512))
                imagea[0:3, :, :] = self.tsf(image)
                imagea[3:6, :, :] = scatrgb
                if self.onehot:
                    scatonehot = scat_indices_coarse_onehot(onehotnum=8, dataset=self.cfg.dataset_name)(scat)
                    # scatonehot = scat_indices_coarse_onehot(onehotnum=4, dataset=self.SF)(scat)
                    scatonehot = torch.from_numpy(scatonehot)
                    scatonehot = scatonehot.permute(2, 0, 1)
                    return{
                        'image':self.tsf(image),
                        'scatrgb':scatrgb,
                        'scatonehot':scatonehot.float(),
                        'pauli_scat_concat':imagea.astype('float32'),
                        'label':label,
                        'name':name,
                        'name1':name1}
                else:
                    if self.bs == 4:
                        return{
                            'image':self.tsf(image),
                            'scatrgb':scatrgb,
                            'pauli_scat_concat':imagea.astype('float32'),
                            'label':label,
                            'name':name}
                    else:
                        return imagea.astype('float32'), label
        else:
            label = label_indices(dataset_SF=self.cfg.dataset_name)(label)
            label = np.array(label).astype('int32')
            label = torch.from_numpy(label).long()
            if self.inputnum == 1:
                return self.tsf(image), label
            if self.inputnum == 2:
                scat = self.scats[idx]
                name1 = scat.split('/')[-1]
                if self.npy:
                    scat = np.load(scat)
                    scat = self.ten(scat)
                    scat = scat/9.5-1
                    if self.bs == 4:
                        return self.tsf(image), label, scat
                    else:
                        imagea = np.zeros((4,512,512))
                        imagea[0:3, :, :] = self.tsf(image)
                        imagea[3:4, :, :] = scat
                        return imagea.astype('float32'), label
                else:
                    scat = np.load(scat)
                    # scat = Image.open(scat).convert('RGB')
                    scatrgb = self.tsf_scat(scat)
                    imagea = np.zeros((6,512,512))
                    imagea[0:3, :, :] = self.tsf(image)
                    imagea[3:6, :, :] = scatrgb
                    if self.onehot:
                        scatonehot = scat_indices_coarse_onehot(onehotnum=8, dataset=self.cfg.dataset_name)(scat)
                        scatonehot = torch.from_numpy(scatonehot)
                        scatonehot = scatonehot.permute(2, 0, 1)
                        return{
                            'image':self.tsf(image),
                            'scatrgb':scatrgb,
                            'scatonehot':scatonehot.float(),
                            'pauli_scat_concat':imagea.astype('float32'),
                            'label':label,
                            'name':name,
                            'name1':name1}
                    else:
                        if self.bs == 4:
                            return{
                                'image':self.tsf(image),
                                'scatrgb':scatrgb,
                                'pauli_scat_concat':imagea.astype('float32'),
                                'label':label,
                                'name':name}
                            # return self.tsf(image), label, scat
                        else:
                            return imagea.astype('float32'), label

    def __len__(self):
        return len(self.images)


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def custom_random_affine(image, label, scat, havescat=True, degrees=38):

    width, height = image.size
    
    shear_x = 0  # random.uniform(-0.38, 0)
    shear_y = 0  # random.uniform(-0.38, 0)
    scale_x = random.uniform(0.83, 1.38)
    scale_y = random.uniform(0.83, 1.38)
    tran_x = random.uniform(-83, 83)
    tran_y = random.uniform(-83, 83)
    image = image.transform((width, height), Image.AFFINE, (scale_x, shear_y, tran_x, shear_x, scale_y, tran_y), Image.BICUBIC)
    label = label.transform((width, height), Image.AFFINE, (scale_x, shear_y, tran_x, shear_x, scale_y, tran_y), Image.NEAREST)
    if havescat:
        if not isinstance(scat, Image.Image):
            scat = Image.fromarray(scat.astype('uint8'))
            
        scat = scat.transform((width, height), Image.AFFINE, (scale_x, shear_y, tran_x, shear_x, scale_y, tran_y), Image.NEAREST)

    ppx = np.random.rand()
    ppf = np.random.rand()
    if ppx > 0.5:
        angle = random.uniform(-degrees, degrees)
        image = image.rotate(angle)
        label = label.rotate(angle)
        if havescat:
            scat = scat.rotate(angle)
            
    if ppf > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        if havescat:
            scat = scat.transpose(Image.FLIP_LEFT_RIGHT)
    return image, label, scat


class label_indices():
    def __init__(self,dataset_SF='SF'):
        if dataset_SF == 'SF':
            colormap = [(0, 0, 0), (0, 128, 0), (0, 255, 0),  (0, 255, 255),
                    (255, 0, 255), (255, 0, 0), (255, 255, 0), (0, 0, 255)]
        elif dataset_SF == 'coarse':
            colormap = [[0, 0, 0],[100, 149, 237], [0, 30, 100],
                    [255, 215, 0], [50, 205, 50],[0, 128, 0]]
        self.colormap = colormap
        colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
        for i, colormap in enumerate(self.colormap):
            colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
        self.colormap2label = colormap2label

    def __call__(self, label):
        label = np.array(label).astype('int32')
        idx = (label[:, :, 0] * 256 + label[:, :, 1]) * 256 + label[:, :, 2]  # 512,512
        labels = np.array(self.colormap2label[idx]).astype('int32')
        return labels

def get_color_index(colormap):
    index = (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]
    return index


class scat_indices_coarse_onehot():
    def __init__(self,dataset,onehotnum=8):
        if dataset == 'SF':
            colormap = [[30,0,100],[180,110,150],[64,86,64],[0,128,0],[120,150,150]]
        elif dataset == 'coarse':
            if onehotnum == 4:
                colormap = [[30,0,100],[20,100,110],[0,128,0],[255,215,0],[173,255,47]]
            elif onehotnum == 8:
                colormap = [[30,0,100],[0,30,20],[0,0,0],[255,215,0],[50,205,50],
                        [20,100,110],[100,149,237],[0,128,0],[173,255,47]]
        self.colormap = colormap
        colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
        for i, colormap in enumerate(self.colormap):
            colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
        self.colormap2label = colormap2label
        if dataset == 'SF':
            self.num_classes = len(self.colormap)
        elif onehotnum == 4:
            self.colormap2label[get_color_index((173, 255, 47))] = 3
            self.num_classes = len(self.colormap) - 1
        elif onehotnum == 8:
            self.colormap2label[get_color_index((173, 255, 47))] = 7
            self.num_classes = len(self.colormap) - 1
    def __call__(self, scat):
        scat = np.array(scat).astype('int32')
        colormap_indices = (scat[:, :, 0] * 256 + scat[:, :, 1]) * 256 + scat[:, :, 2]
        scatonehot = np.eye(self.num_classes)[self.colormap2label[colormap_indices]]
        return scatonehot

csv_pathtest = r'datasets/XLCS_mydata/pauli_512test.csv'
csv_pathtest_scat = r'datasets/XLCS_mydata/scat_512test.csv'
csv_pathtest_scat_npy = r'datasets/XLCS_mydata/scatnpy_512test.csv'
path_test_m6 = r'datasets/SF/pauli_scat_label_val copy.csv'
path_test = r'datasets/SF/pauli_scat_label_val.csv'
path_test_coarse_m6 = r'datasets/XLCS_mydata/XLCS_mydata_test copy.csv'
path_test_coarse = r'datasets/XLCS_mydata/XLCS_mydata_test.csv'
path_test_coarse16 = r'datasets/XLCS_mydata/XLCS_mydata_test16.csv'


class Test(Dataset):
    def __init__(self, SF):
        self.SF = SF
        if self.SF == 'SF':
            images, scats, labels = csv_list_multimodel(path_test)
            self.tsf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.361, 0.258, 0.563], [0.144, 0.129, 0.162])])
            self.tsf_scat = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.245, 0.153, 0.395], [0.145, 0.111, 0.081])
            ])
        elif self.SF == 'coarse':
            images, scats, labels = csv_list_multimodel(path_test_coarse)
            self.tsf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.440, 0.372, 0.442], [0.488, 0.424, 0.484])
            ])
            self.tsf_scat = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.289, 0.282, 0.154], [0.375, 0.460, 0.350])
            ])
        self.images = images
        self.scats = scats
        self.labels = labels
        print('Read ' + str(len(self.images)) + ' valid examples')

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        scat = self.scats[idx]
        image = Image.open(image).convert('RGB')
        scat = Image.open(scat).convert('RGB')
        label = Image.open(label).convert('RGB')
        label = np.array(label).astype('int32')
        scatonehot = scat_indices_coarse_onehot(onehotnum=8, dataset=self.SF)(scat)
        scatonehot = torch.from_numpy(scatonehot)
        scatonehot = scatonehot.permute(2, 0, 1)
        label = label_indices(dataset_SF=self.SF)(label)
        label = torch.from_numpy(label).long()
        imagea = np.zeros((6,512,512))
        imagea[0:3, :, :] = self.tsf(image)
        imagea[3:6, :, :] = self.tsf_scat(scat)
        return self.tsf(image), self.tsf_scat(scat), scatonehot.float(), imagea.astype('float32'), label
    def __len__(self):
        return len(self.images)
