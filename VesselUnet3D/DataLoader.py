import tifffile, os
import numpy as np
import torch
from os.path import join
from skimage import exposure
from torch import nn as nn
import torch

def MaxProject(img):
    imgs = []
    for z in range(2):
        for y in range(2):
            for x in range(2):
                imgs.append(img[z::2, y::2, x::2])
    img = np.max(imgs, axis=0)
    return img

class GetMultiTypeMemoryDataSetAndCropQxz:
    def __init__(self, path, txtName, imgSize, imgName, maskName):
        # self.backThre = 0.3
        self.imgSize = imgSize
        self.pSum = self.imgSize[0] * self.imgSize[1] * self.imgSize[2]
        self.imgPath = join(path, str(imgName))
        # self.maskPath = join(path, 'mask')
        self.maskPath = join(path, str(maskName))
        with open(join(path, txtName), 'r') as f:
            self.nameLs = f.read().strip().split('\n')

    def __len__(self): return len(self.nameLs)

    def __getitem__(self, ind):
        img = tifffile.imread(join(self.imgPath, self.nameLs[ind]))[:self.imgSize[2], :self.imgSize[1], :self.imgSize[0]].astype(np.float32)
        img = np.expand_dims(img, axis=0).astype(np.float32)
        img = torch.from_numpy(img)  # Convert ToPyTorch张量
        # img = (img - img.min()) / (img.max() - img.min())
        img = (img - img.mean()) / img.std()
        mask = tifffile.imread(join(self.maskPath, self.nameLs[ind]))[:self.imgSize[2], :self.imgSize[1], :self.imgSize[0]] / 255.0
        mask = np.expand_dims(mask, axis=0).astype(np.float32)
        return img, mask, self.nameLs[ind]


class GetMultiTypeMemoryDataSetAndCropQxz2:
    def __init__(self, imagesDir, nameLs):
        self.imagesDir = imagesDir
        nameLs = [l for l in nameLs if ".tif" in l]
        if len(nameLs) == 0:
            self.nameLs = [l for l in os.listdir(self.imagesDir) if ".tif" in l]
        else:
            self.nameLs = nameLs
        if len(self.nameLs):
            nameLs = self.nameLs
            for name in nameLs:
                with tifffile.TiffFile(join(self.imagesDir, name)) as tif:
                    img0 = tif.pages[0].asarray()
                if len(img0.shape) != 2:
                    self.nameLs.remove(name)
                    continue
                if len(img0.shape) == 2:
                    img = tifffile.imread(join(self.imagesDir, name))
                    if len(img.shape) != 3:
                        self.nameLs.remove(name)
            if len(self.nameLs):
                self.imgSize = tifffile.imread(join(self.imagesDir, self.nameLs[0])).shape[::-1]  # xyz
                self.imgSize = np.array(self.imgSize, dtype=np.int32)
                self.pSum = self.imgSize[0] * self.imgSize[1] * self.imgSize[2]

    def __len__(self): return len(self.nameLs)

    def __getitem__(self, ind):
        imgPath = join(str(self.imagesDir), self.nameLs[ind])
        img = tifffile.imread(imgPath)[:self.imgSize[2], :self.imgSize[1], :self.imgSize[0]].astype(np.float32)
        # img = np.zeros([self.imgSize[2], self.imgSize[1], self.imgSize[0]], dtype=np.uint16)
        img = np.expand_dims(img, axis=0).astype(np.float32)
        img = torch.from_numpy(img)
        # img = (img - img.min()) / (img.max() - img.min())
        img = (img - img.mean()) / img.std()
        return img, self.nameLs[ind]
