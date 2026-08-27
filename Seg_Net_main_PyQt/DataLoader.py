import tifffile, os
import torch
from os.path import join
from skimage import exposure
import numpy as np
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
    def __init__(self, path, txtName, imgSize, imageName, maskName, distName=None):
        # self.backThre = 0.3
        self.imgSize = imgSize
        self.pSum = self.imgSize[0] * self.imgSize[1] * self.imgSize[2]
        self.imgPath = join(path, str(imageName))
        self.maskPath = join(path, str(maskName))
        if distName is not None:
            self.distPath = join(path, str(distName))
        else:
            self.distPath = None
        with open(join(path, txtName), 'r') as f:
            self.nameLs = f.read().strip().split('\n')

    def __len__(self): return len(self.nameLs)

    def __getitem__(self, ind):
        img = tifffile.imread(join(self.imgPath, self.nameLs[ind])
                              )[:self.imgSize[2], :self.imgSize[1], :self.imgSize[0]].astype(np.float32)
        mask = tifffile.imread(join(self.maskPath, self.nameLs[ind])
                               )[:self.imgSize[2], :self.imgSize[1], :self.imgSize[0]]
        # mask[mask < 100] = 0
        mask = mask / 255.0
        img = np.expand_dims(img, axis=0).astype(np.float32)
        img = torch.from_numpy(img)
        # img = (img - img.min()) / (img.max() - img.min())
        mask = np.expand_dims(mask, axis=0).astype(np.float32)
        if self.distPath is not None:
            dist = tifffile.imread(join(self.distPath, self.nameLs[ind])
                                   )[:self.imgSize[2], :self.imgSize[1], :self.imgSize[0]] / 255.0
            dist = np.expand_dims(dist, axis=0).astype(np.float32)
            return img, mask, dist, self.nameLs[ind]
        else:
            return img, mask, None, self.nameLs[ind]

    # def __getitem__(self, ind):
    #     tyx_i = (128, 128, 128)
    #     img = tifffile.imread(join(self.imgPath, self.nameLs[ind])
    #                           )[:self.imgSize[2], :self.imgSize[1], :self.imgSize[0]].astype(np.float32)
    #     mask = tifffile.imread(join(self.maskPath, self.nameLs[ind])
    #                            )[:self.imgSize[2], :self.imgSize[1], :self.imgSize[0]] / 255.0
    #     cut_label = [0]
    #     cut_data = [0]
    #     while np.max(cut_label) <= 0 or np.max(cut_data) <= 0:
    #         start_z = np.random.randint(0, mask.shape[0] - tyx_i[0])
    #         start_y = np.random.randint(0, mask.shape[1] - tyx_i[0])
    #         start_x = np.random.randint(0, mask.shape[2] - tyx_i[0])
    #         cut_data = img[
    #                    start_z:start_z + tyx_i[0],
    #                    start_y:start_y + tyx_i[1],
    #                    start_x:start_x + tyx_i[2]
    #                    ]
    #         cut_label = mask[
    #                     start_z:start_z + tyx_i[0],
    #                     start_y:start_y + tyx_i[1],
    #                     start_x:start_x + tyx_i[2]
    #                     ]
    #     img = np.expand_dims(cut_data, axis=0).astype(np.float32)
    #     img = torch.from_numpy(img)
    #     # img = (img - img.min()) / (img.max() - img.min())
    #     mask = np.expand_dims(cut_label, axis=0).astype(np.float32)
    #     return img, mask, self.nameLs[ind]
