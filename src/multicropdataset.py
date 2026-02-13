# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from logging import getLogger

import numpy as np
from torch.utils.data import Dataset
import torch
import rasterio
from rasterio.windows import Window
import kornia
from scipy import ndimage

logger = getLogger()



class DataloaderEval(Dataset):
    def __init__(self, 
                 img, 
                 coords, 
                 psize = 128):

        super(DataloaderEval, self).__init__()
        self.img = img
        self.coord = coords
        self.psize = psize
        
        
    def __len__(self):
        return self.coord.shape[0]
        
    def __getitem__(self, idx):
        
        image = self.img[:,self.coord[idx,0]-self.psize//2:self.coord[idx,0]+self.psize//2+self.psize%2,
                              self.coord[idx,1]-self.psize//2:self.coord[idx,1]+self.psize//2+self.psize%2]
        

        image = torch.from_numpy(image).float()/255
        

        return image


    
class RasterTileDataset(Dataset):
    """
    Raster-based tile dataset using rasterio windowed reads.
    """

    def __init__(self, 
                 img, 
                 lab, 
                 coords, 
                 psize, 
                 samples=None,
                 p_flip=0.5,
                 p_noise=0.3,
                 noise_amount=0.01):
        
        self.img = img
        self.lab = lab
        self.coords = coords
        self.tile_size = psize + (psize % 2)
        self.samples = samples
        self.p_flip = p_flip

    def __len__(self):
        return self.samples


    @staticmethod
    def _random_flip(img, gt):
        if torch.rand(1) < 0.5:
            img = torch.flip(img, dims=[1])   # vertical
            gt = torch.flip(gt, dims=[1])

        if torch.rand(1) < 0.5:
            img = torch.flip(img, dims=[2])   # horizontal
            gt = torch.flip(gt, dims=[2])

        return img, gt


    def __getitem__(self, idx):
        coords_idx = np.random.randint(0, self.coords.shape[0])
        row, col = self.coords[coords_idx]

        half = self.tile_size // 2
        row_start = max(0, row - half)
        col_start = max(0, col - half)
        row_end = row_start + self.tile_size
        col_end = col_start + self.tile_size


        inp = self.img[:,row_start:row_end, col_start:col_end].copy()
        gt = self.lab[row_start:row_end, col_start:col_end].copy()

        inp = torch.from_numpy(inp).float()/255
        gt = torch.from_numpy(gt[None, :, :].astype(np.float32))

        if torch.rand(1) < self.p_flip:
            inp, gt = self._random_flip(inp, gt)


        return inp, gt[0]
    
    
