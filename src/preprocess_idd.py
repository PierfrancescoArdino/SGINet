import os
import glob
from shutil import copy2
from PIL import Image
import json
import numpy as np
import argparse
import json
from publiccode.helpers.annotation import Annotation
from publiccode.helpers.anue_labels import labels, name2label
from collections import namedtuple
Rectangle = namedtuple('Rectangle', 'ymin xmin ymax xmax')

ra = Rectangle(3., 3., 5., 5.)
rb = Rectangle(1., 1., 4., 3.5)

def check_bbox_occlusion(bbox1, bbox2):

    def x_touch(bbox1, bbox2):
        return bbox1[2] == bbox2[0] and ((bbox1[1] <= bbox2[1] <= bbox1[3]) or (bbox1[1] <= bbox2[3] <= bbox1[3]))

    def y_touch(bbox1, bbox2):
        return bbox1[1] == bbox2[3] and ((bbox1[0] <= bbox2[0] <= bbox1[2]) or (bbox1[0] <= bbox2[2] <= bbox1[2]))

    def touch(bbox1, bbox2):
        return x_touch(bbox1, bbox2) or x_touch(bbox2, bbox1) or y_touch(bbox1, bbox2) or y_touch(bbox2, bbox1)

    def overlap(a, b):  # returns None if rectangles don't intersect
        height = float(a[2] - a[0] + 1)
        width = float(a[3] - a[1] + 1)
        a = Rectangle(a[0],a[1], a[2], a[3])
        b = Rectangle(b[0], b[1], b[2], b[3])
        dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
        dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
        if (dx>=0) and (dy>=0):
            if dx * dy > 0.05 * height * width:
                return True
            else:
                return False
        else:
            return False
    return touch(bbox1, bbox2) or overlap(bbox1, bbox2)


def check_occlusion(json_data, img):
    occlusions = {}
    for obj_id in json_data:
        occlusion = list((map(lambda x: json_data[x]["global_id"] > json_data[obj_id]["global_id"] and check_bbox_occlusion(json_data[obj_id]["bbox"], json_data[x]["bbox"]), json_data)))
        occlusions[obj_id] = any(occlusion)
        json_data[obj_id]["occlusion"] = any(occlusion)

    return json_data



def generate_json_data_inst(fname, img, src):
    json_name = "_".join(fname.split('_')[:-1]) + "_polygons.json"
    annotation = Annotation()
    annotation.fromJsonFile(json_name)
    nbInstances = {}
    instances_ids = {}
    json_data = {}
    global_id = 0
    for labelTuple in labels:
        if labelTuple.hasInstances:
            nbInstances[labelTuple.name] = 0
    for obj in annotation.objects:
        label   = obj.label
        polygon = obj.polygon

        # If the object is deleted, skip it
        if obj.deleted:
            continue

        # if the label is not known, but ends with a 'group' (e.g. cargroup)
        # try to remove the s and see if that works
        # also we know that this polygon describes a group
        isGroup = False
        if ( not label in name2label ) and label.endswith('group'):
            label = label[:-len('group')]
            isGroup = True

        if not label in name2label:
            print( "Label '{}' not known.".format(label) )
            continue

        # the label tuple
        labelTuple = name2label[label]
        id = labelTuple.train_id
        if labelTuple.hasInstances and not isGroup and id != 255:
            if id not in instances_ids:
                instances_ids[id] = []
            id = id * 1000 + nbInstances[label]
            instances_ids[labelTuple.train_id].append(id)
            nbInstances[label] += 1
            object_id = global_id
            global_id += 1
            ins_map = (np.squeeze(img) == np.squeeze(id)).astype(float)
            ins_area = np.sum(ins_map)
            if ins_area > 0:
                yy, xx = np.where(ins_map > 0)
                height = float(yy.max() - yy.min() + 1)
                width = float(xx.max() - xx.min() + 1)
                box_area = height * width
                area_ratio = float(ins_area) / box_area
                width_ratio = float(512) / width
                height_ratio = float(256) / height
                aspect_ratio = np.max((height / width, width / height))
                json_data[str(id)] = {"class_id": int(labelTuple.train_id),
                                      "global_id": int(object_id),
                                      "train_id": int(labelTuple.train_id),
                                      "area": float(ins_area),
                                      "height": height,
                                      "width": width,
                                      "box_area": box_area,
                                      "area_ratio": area_ratio,
                                      "width_ratio": width_ratio,
                                      "height_ratio": height_ratio,
                                      "aspect_ratio": float(aspect_ratio),
                                      "bbox": [float(yy.min()), float(xx.min()), float(yy.max()), float(xx.max())]
                                      }
    json_data = check_occlusion(json_data, img)
    return json_data



def copy_file(src, src_ext, dst, type):
    # find all files ends up with ext
    flist = sorted(glob.glob(os.path.join(src, '*', src_ext)))
    for fname in flist:
        src_path = fname
        img = Image.open(src_path)
        if img.size != (1920, 1080):
            continue
        size = 512, 512
        if type == "img":
            img = img.resize((512,288), Image.BICUBIC)
            img = img.crop((0,0,512,256))
        elif type in ["inst", "label"]:
            img = img.resize((512,288), Image.NEAREST)
            img = img.crop((0, 0, 512, 256))
            if type == "inst":
                json_data = generate_json_data_inst(fname, img, src)
                json.dump(json_data, open(os.path.join(dst,"_".join(src_path.replace("_instanceIds.png", "").split("/")[-2:])  + "_data.json"),
                                          "w"), indent=4)
        dst_path = ".".join([".".join(os.path.join(dst,"_".join(fname.split("/")[-2:])).split(".")[:-1]), "png"])
        img.save(dst_path)
        print('copied %s to %s' % (src_path, os.path.join(dst,"_".join(fname.split("/")[-2:]))))


# organize image
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str,
                        default='../datasets')

    opt = parser.parse_args()
    folder_name = opt.dataroot
    train_img_dst = os.path.join(folder_name, 'train_img')
    train_label_dst = os.path.join(folder_name, 'train_label')
    train_inst_dst = os.path.join(folder_name, 'train_inst')
    train_bbox_dst = os.path.join(folder_name, 'train_bbox')
    val_img_dst = os.path.join(folder_name, 'val_img')
    val_label_dst = os.path.join(folder_name, 'val_label')
    val_inst_dst = os.path.join(folder_name, 'val_inst')
    val_bbox_dst = os.path.join(folder_name, 'val_bbox')
    if not os.path.exists(train_img_dst):
        os.makedirs(train_img_dst)
    if not os.path.exists(train_label_dst):
        os.makedirs(train_label_dst)
    if not os.path.exists(train_inst_dst):
        os.makedirs(train_inst_dst)
    if not os.path.exists(val_img_dst):
        os.makedirs(val_img_dst)
    if not os.path.exists(val_label_dst):
        os.makedirs(val_label_dst)
    if not os.path.exists(val_inst_dst):
        os.makedirs(val_inst_dst)
    # train_image
    copy_file(os.path.join(opt.dataroot,'leftImg8bit/train'),
              '*_leftImg8bit.*', train_img_dst,"img")
    # train_label
    copy_file(os.path.join(opt.dataroot,'gtFine/train'),
              '*_trainIds.*', train_label_dst, "label")
    # train_inst
    copy_file(os.path.join(opt.dataroot,'gtFine/train'),
            '*_instanceIds.*', train_inst_dst, "inst")
    # val_image
    copy_file(os.path.join(opt.dataroot,'leftImg8bit/val'),
            '*_leftImg8bit.*', val_img_dst, "img")
    # val_label
    copy_file(os.path.join(opt.dataroot,'gtFine/val'),
            '*_trainIds.*', val_label_dst, "label")
    # val_inst
    copy_file(os.path.join(opt.dataroot,'gtFine/val'),
            '*_instanceIds.*', val_inst_dst, "inst")