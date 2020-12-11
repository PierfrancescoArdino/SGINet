import os
import sys
import numpy as np
import random
import math
from PIL import Image, ImageOps, ImageFilter

import torch
import torch.utils.data as data
import torchvision.transforms as transform
import torchvision.transforms.functional as TF
import re
from tqdm import tqdm
from .base import BaseDataset
import utils.util as util
import json


class CityscapesSegmentation(BaseDataset):
    BASE_DIR = 'cityscapes'
    NUM_CLASS = 28


    def __init__(self, root='../datasets', split='train',
                 mode=None, transform=None, crop_type = "random",
                 **kwargs):
        super(CityscapesSegmentation, self).__init__(
            root, split, mode, transform, **kwargs)
        # assert exists
        root = os.path.join(root, self.BASE_DIR)
        if self.opt.label_nc == 8:
            self.ch_to_inst_id = {"5":"24", "6":"26"}
            self.param_obj = {"24": [0.25, 50, 10, 4, 0.4],
                              "26": [0.4, 50, 20, 6, 0.5]}
        elif self.opt.label_nc == 17:
            self.ch_to_inst_id = {"8": "24", "9": "25", "10": "26","11": "27", "12": "28", "13": "31", "14":"32", "15":"33"}
            self.param_obj = {"24": [0.25, 50, 10, 4, 0.3],
                              "25": [0.1, 50, 10, 10, 0.10],
                              "26": [0.6, 40, 10, 1.5, 0.70],
                              "27": [0.5, 30, 10, 3, 0.4],
                              "28": [0.5, 50, 20, 6, 0.5],
                              "31": [0.5, 50, 20, 6, 0.5],
                              "32": [0.20, 50, 10, 6, 0.25],
                              "33": [0.20, 50, 10, 6, 0.25],
                              }
        self.use_bbox = False
        self.idx_valid_ins = -1
        assert os.path.exists(root), "Please download the dataset!!"
        self.classes_of_interest_ids = self.opt.classes_of_interest_ids
        self.use_bbox = self.opt.use_bbox
        self.crop_type = crop_type
        self.images, self.gts, self.inst_maps, self.inst_data_json =\
            _get_cityscapes_tuple(root, split, self.use_bbox, self.opt.fineTuning)
        if split != 'vis':
            assert (len(self.images) == len(self.gts))
        if len(self.images) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: \
                " + root + "\n"))

    def __getitem__(self, index):
        if self.mode == "test":
            outputs = self.get_test_item(index)
            return outputs
        outputs = {}
        flip = False
        crop = True if self.opt.size_crop_height is not None and self.opt.size_crop_width is not None else False
        """resize_img = transform.Resize((self.opt.image_height,
                                       self.opt.image_width),
                                      Image.BICUBIC)
        resize_gt_inst = transform.Resize((self.opt.image_height,
                                           self.opt.image_width),
                                          Image.NEAREST)"""
        if (not self.opt.no_flip) and np.random.uniform() < 0.5 and self.split == "train":
            flip = True
        img = Image.open(self.images[index]).convert('RGB')
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        #img = resize_img(img)
        if self.use_bbox:
            inst_map = Image.open(self.inst_maps[index])
            json_data = json.load(open(self.inst_data_json[index]))
            if flip:
                inst_map = inst_map.transpose(Image.FLIP_LEFT_RIGHT)
            #inst_map = resize_gt_inst(inst_map)

        gt = Image.open(self.gts[index])
        if flip:
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)

        #gt = resize_gt_inst(gt)
        if crop:

            i, j, h, w = self.get_crop_params()
            if self.use_bbox:
                self.idx_valid_ins = self.check_instance(inst_map, gt, False, [i, j, h, w], json_data, flip=flip)
            img = TF.crop(img, i, j, h, w)
            gt = TF.crop(gt, i, j, h, w)
            if self.use_bbox:
                inst_map = TF.crop(inst_map, i, j, h, w)
        elif self.use_bbox:
            self.idx_valid_ins = self.check_instance(inst_map, gt, False, json_data=json_data, flip=flip)

        self.idx_valid_ins = self.idx_valid_ins if np.random.uniform() >= self.opt.prob_bg else -1
        if self.transform is not None:
            img = self.transform(img)
        if self.gt_transform is not None:
            gt = self.gt_transform(gt) * 255
        if self.inst_map_transform is not None and self.use_bbox:
            inst_map = inst_map_complete =self.inst_map_transform(inst_map)
        if self.idx_valid_ins is not -1:
            masks, indexes = util._random_mask_with_instance_cond(
                img.shape[1:], self.opt.min_hole_size,
                self.opt.max_hole_size, (inst_map == self.idx_valid_ins) * 1.0, None)
            insta_maps_bbox, inst_map_compact, theta = util.inst_map2bbox(
                (inst_map == self.idx_valid_ins) * 1.0, self.opt)

        else:
            masks, indexes = util._random_mask(img.shape[1:],
                                               self.opt.min_hole_size,
                                               self.opt.max_hole_size)
            inst_map = insta_maps_bbox = torch.zeros(1, img.shape[1], img.shape[2])
            inst_map_compact = torch.zeros(1, self.opt.compact_sizey, self.opt.compact_sizex)
            theta = torch.Tensor([0 for _ in range(6)])
        mask_transform = transform.Compose([
            transform.ToTensor()])
        outputs["gt_images"] = img
        outputs["gt_seg_maps"] = gt
        outputs["inst_map"] = (inst_map == self.idx_valid_ins) * 1.0 if self.use_bbox else inst_map
        outputs["inst_map_complete"] = inst_map_complete
        outputs["inst_map_valid_idx"] = self.idx_valid_ins
        outputs["images_path"] = self.images[index]
        outputs["masks"] = mask_transform(masks)
        outputs["indexes"] = torch.Tensor(indexes)
        outputs["insta_maps_bbox"] = insta_maps_bbox
        outputs["inst_map_compact"] = inst_map_compact
        outputs["theta"] = theta
        outputs["compute_instance"] = torch.Tensor([1]) if self.idx_valid_ins is not -1 else torch.Tensor([0])


        return outputs

    def get_test_item(self, index):
        outputs = {}
        flip = False
        crop = True if self.opt.size_crop_height is not None and self.opt.size_crop_width is not None else False

        img = Image.open(self.images[index]).convert('RGB')
        if self.use_bbox:
            inst_map = Image.open(self.inst_maps[index])
            json_data = json.load(open(self.inst_data_json[index]))

        gt = Image.open(self.gts[index])
        i, j, h, w = 0, 0, self.opt.image_height, self.opt.image_width
        if crop:
            i, j, h, w = self.get_crop_params()
            if self.opt.use_load_mask:
                masks, indexes = util.load_masks_data(self.opt)
            if self.use_bbox:
                self.idx_valid_ins = self.check_instance_with_masks_and_crop(
                    inst_map, gt, True, (
                        indexes[index] if self.opt.use_load_mask else [0, self.opt.size_crop_height,
                                                                       0, self.opt.size_crop_width]),
                    [i, j, h, w], json_data, False)
            img = TF.crop(img, i, j, h, w)
            gt = TF.crop(gt, i, j, h, w)
            if self.use_bbox:
                inst_map = TF.crop(inst_map, i, j, h, w)
        else:
            if self.opt.use_load_mask:
                masks, indexes = util.load_masks_data(self.opt)
            if self.use_bbox:
                self.idx_valid_ins = self.check_instance_with_masks_and_crop(
                    inst_map, gt, True, (
                        indexes[index] if self.opt.use_load_mask else [0, 256,
                                                                       0, 256]),
                    [i, j, h, w], json_data, False)
        if self.gt_transform is not None:
            gt = self.gt_transform(gt) * 255
        if self.inst_map_transform is not None and self.use_bbox:
            inst_map = inst_map_complete =self.inst_map_transform(inst_map)
        self.idx_valid_ins = self.idx_valid_ins if np.random.uniform() >= self.opt.prob_bg else -1
        if self.transform is not None:
            img = self.transform(img)
        if self.opt.use_load_mask:
            if self.idx_valid_ins is not -1:
                insta_maps_bbox, inst_map_compact, theta = util.inst_map2bbox(
                    (inst_map == self.idx_valid_ins) * 1.0, self.opt)
            else:
                inst_map = insta_maps_bbox = torch.zeros(1, img.shape[1],
                                                         img.shape[2])
                inst_map_compact = torch.zeros(1, self.opt.compact_sizey,
                                               self.opt.compact_sizex)
                theta = torch.Tensor([0 for _ in range(6)])
        else:
            if self.idx_valid_ins is not -1:
                masks, indexes = util._random_mask_with_instance_cond(
                    img.shape[1:], self.opt.min_hole_size,
                    self.opt.max_hole_size,
                    (inst_map == self.idx_valid_ins) * 1.0, index)
                insta_maps_bbox, inst_map_compact, theta = util.inst_map2bbox(
                    (inst_map == self.idx_valid_ins) * 1.0, self.opt)

            else:
                masks, indexes = util._random_mask(img.shape[1:],
                                                   self.opt.min_hole_size,
                                                   self.opt.max_hole_size, True)
                inst_map = insta_maps_bbox = torch.zeros(1, img.shape[1],
                                                         img.shape[2])
                inst_map_compact = torch.zeros(1, self.opt.compact_sizey,
                                               self.opt.compact_sizex)
                theta = torch.Tensor([0 for _ in range(6)])
        mask_transform = transform.Compose([
            transform.ToTensor()])
        outputs["gt_images"] = img
        outputs["gt_seg_maps"] = gt
        outputs["inst_map"] = (
                                      inst_map == self.idx_valid_ins) * 1.0 if self.use_bbox else inst_map
        outputs["inst_map_complete"] = inst_map_complete
        outputs["inst_map_valid_idx"] = self.idx_valid_ins
        outputs["images_path"] = self.images[index]
        outputs["masks"] = insta_maps_bbox
        outputs["indexes"] = torch.Tensor(
            (indexes[index] if self.opt.use_load_mask else indexes))
        outputs["insta_maps_bbox"] = insta_maps_bbox
        outputs["inst_map_compact"] = inst_map_compact
        outputs["theta"] = theta
        outputs["compute_instance"] = torch.Tensor(
            [1]) if self.idx_valid_ins is not -1 else torch.Tensor([0])

        return outputs


    def get_crop_params(self):
        crop_type = np.random.choice(["left","center","right"])
        if self.crop_type != "random":
            crop_type = self.crop_type
        if crop_type == "left":
            return 0, 0, self.opt.size_crop_width, self.opt.size_crop_width
        elif crop_type == "right":
            return 0, self.opt.size_crop_width, self.opt.size_crop_height, self.opt.size_crop_width
        else:
            return 0, self.opt.size_crop_width /2, self.opt.size_crop_height, self.opt.size_crop_width

    def check_instance(self, inst, gt, test = False, crop_indexes =[0, 128, 256, 256], json_load=[], flip=False):
        instance_idx = []
        idx_valid_ins = []
        inst = self.inst_map_transform(inst)
        gt = self.gt_transform(gt) * 255
        contained_idx = []
        contained_idx = [int(obj_id) for obj_id, obj in json_load.items() if obj["train_id"] in self.classes_of_interest_ids and str(obj["class_id"]) in list(self.ch_to_inst_id.values())]
        if len(contained_idx) == 0:
            return -1
        for obj_id, obj in json_load.items():
            if int(obj_id) not in contained_idx:
                continue
            min_size_percent = self.param_obj[str(obj_id)[:2]][0]
            max_width_ratio  = self.param_obj[str(obj_id)[:2]][1]
            max_height_ratio  = self.param_obj[str(obj_id)[:2]][2]
            max_aspect_ratio  = self.param_obj[str(obj_id)[:2]][3]
            min_area_ratio  = self.param_obj[str(obj_id)[:2]][4]
            if obj["area"] > (
                    inst.size()[1] * inst.size()[2] * min_size_percent / 100):
                height = obj["height"]
                width = obj["width"]
                area_ratio = obj["area_ratio"]

                width_ratio = obj["width_ratio"]
                height_ratio = obj["height_ratio"]
                aspect_ratio = obj["aspect_ratio"]
                bbox = obj["bbox"]
                if flip == True:
                    bbox = [bbox[0], self.opt.image_width - 1 - bbox[3], bbox[2],
                             self.opt.image_width - 1 - bbox[1]]

                if obj["occlusion"] is False and width_ratio < max_width_ratio and height_ratio < max_height_ratio and \
                        area_ratio > min_area_ratio and aspect_ratio < max_aspect_ratio and \
                        height <= self.opt.max_hole_size and width <= self.opt.max_hole_size and util.instInCrop(bbox, crop_indexes, 0)\
                        and util.instInCrop(bbox, [0, 0, self.opt.image_height, self.opt.image_width], 20):
                    idx_valid_ins.append(np.float(obj_id))
        if test == False:
            return np.random.choice(idx_valid_ins) if len(idx_valid_ins) > 0 else -1
        else:
            return idx_valid_ins if len(
                idx_valid_ins) > 0 else -1

    def check_instance_with_masks_and_crop(self, inst, gt, test=False,
                                           indexes=[], crop_indexes=[], json_load=[], flip=False):
        instance_idx = []
        idx_valid_ins = []
        inst = self.inst_map_transform(inst)
        gt = self.gt_transform(gt) * 255
        contained_idx = []
        areas = []
        contained_idx = [int(obj_id) for obj_id, obj in json_load.items() if obj["train_id"] in self.classes_of_interest_ids and str(obj["class_id"]) in list(self.ch_to_inst_id.values())]
        if len(contained_idx) == 0:
            return -1
        for obj_id, obj in json_load.items():
            if int(obj_id) not in contained_idx:
                continue
            min_size_percent = self.param_obj[str(obj_id)[:2]][0]
            max_width_ratio = self.param_obj[str(obj_id)[:2]][1]
            max_height_ratio = self.param_obj[str(obj_id)[:2]][2]
            max_aspect_ratio = self.param_obj[str(obj_id)[:2]][3]
            min_area_ratio = self.param_obj[str(obj_id)[:2]][4]
            if obj["area"] > (
                    inst.size()[1] * inst.size()[2] * min_size_percent / 100):
                height = obj["height"]
                width = obj["width"]
                area_ratio = obj["area_ratio"]

                width_ratio = obj["width_ratio"]
                height_ratio = obj["height_ratio"]
                aspect_ratio = obj["aspect_ratio"]
                bbox = obj["bbox"]
                if flip == True:
                    bbox = [bbox[0], self.opt.image_width - 1 - bbox[3],
                            bbox[2],
                            self.opt.image_width - 1 - bbox[1]]

                if obj["occlusion"] is False and width_ratio < max_width_ratio and height_ratio < max_height_ratio and \
                        area_ratio > min_area_ratio and aspect_ratio < max_aspect_ratio and \
                        height <= self.opt.max_hole_size and width <= self.opt.max_hole_size and util.instInMap(
                    bbox, indexes, crop_indexes)  and util.instInCrop(bbox, crop_indexes, 0)\
                        and util.instInCrop(bbox, [0, 0, self.opt.image_height, self.opt.image_width], 20):
                    idx_valid_ins.append(np.float(obj_id))
                    areas.append(obj["area"])
        if test == False:
            return np.random.choice(idx_valid_ins) if len(
                idx_valid_ins) > 0 else -1
        else:
            return idx_valid_ins[np.argmax(areas)] if len(
                idx_valid_ins) > 0 else -1


    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0







def _get_cityscapes_tuple(folder, split='train', use_bbox=False, fine_tutning=False):
    def get_path_tuples(folder, split_f, use_bbox, split):
        img_paths = []
        gt_paths = []
        inst_maps = []
        insts_data_json = []
        with open(split_f, 'r') as lines:
            for line in lines.read().splitlines():
                imgpath = os.path.join(folder, f"{split}_img", line + "_leftImg8bit.png")
                gtpath = os.path.join(folder, f"{split}_label", line + "_gtFine_trainIds.png")
                if use_bbox:
                    inst_map = os.path.join(folder, f"{split}_inst", line + "_gtFine_instanceIds.png")
                    inst_data_json = os.path.join(folder, f"{split}_inst", line + "_gtFine_data.json")
                else:
                    inst_bbox = inst_map = inst_data_json = ""
                if os.path.isfile(gtpath):
                    img_paths.append(imgpath)
                    gt_paths.append(gtpath)
                    inst_maps.append(inst_map)
                    insts_data_json.append(inst_data_json)
                else:
                    print('cannot find the mask:', gtpath)
        return img_paths, gt_paths, inst_maps, insts_data_json

    if split == 'train':
        split_f = os.path.join(folder, 'train.txt')
        img_paths, mask_paths, inst_maps, inst_data_json = get_path_tuples(folder, split_f, use_bbox, "train")
    elif split == 'test':
        split_f = os.path.join(folder, 'val.txt')
        img_paths, mask_paths, inst_maps, inst_data_json = get_path_tuples(folder, split_f, use_bbox, "val")
    else:
        split_f = os.path.join(folder, 'trainval.txt')
        img_paths, mask_paths, inst_maps, inst_data_json = get_path_tuples(folder, split_f, use_bbox)

    return img_paths, mask_paths, inst_maps, inst_data_json