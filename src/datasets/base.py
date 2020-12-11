import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
import torch.utils.data as data

__all__ = ['BaseDataset']


class BaseDataset(data.Dataset):
    def __init__(self, root, split, mode=None, transform=None,
                 gt_transform=None, inst_map_transform=None,
                 logger=None, scale=True, opt=None):
        self.root = root
        self.transform = transform
        self.gt_transform = gt_transform
        self.inst_map_transform = inst_map_transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.logger = logger
        self.scale = scale
        self.opt = opt

    def __getitem__(self, index):
        raise NotImplemented

    @property
    def num_class(self):
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        raise NotImplemented



