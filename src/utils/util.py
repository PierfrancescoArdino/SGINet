from __future__ import print_function
import torch
torch.set_printoptions(precision=10)

import numpy as np
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import Line2D
import torchvision.transforms as transform
import cv2
from torchvision.transforms.functional import to_pil_image
def normalize_tensor(in_feat,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
    return in_feat/(norm_factor+eps)


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if image_tensor is None:
        return [None]
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)

# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8):
    if label_tensor is None:
        return [None]
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def generate_masks(opt):
    masks = []
    indexes = []
    image_height = opt.size_crop_height if opt.size_crop_height else opt.image_height
    image_width = opt.size_crop_width if opt.size_crop_width else opt.image_width
    for _ in range(opt.how_many):
        mask, index = _random_mask([image_height, image_width], opt.min_hole_size, opt.max_hole_size)
        masks.append(mask)
        indexes.append(index)
    return np.asarray(masks), indexes


def generare_masks_data(opt):
    import pickle
    masks, indexes = generate_masks(opt)
    opt_dict = {}
    image_height = opt.size_crop_height if opt.size_crop_height else opt.image_height
    image_width = opt.size_crop_width if opt.size_crop_width else opt.image_width
    opt_dict["masks"] = masks
    opt_dict["indexes"] = indexes
    opt_dict["min_hole_size"] = opt.min_hole_size
    opt_dict["max_hole_size"] = opt.max_hole_size
    opt_dict["image_height"] = image_height
    opt_dict["image_width"] = image_width
    if opt.dataset.lower() == "indiandrivingdataset":
        folder = "../masks_data_idd"
    else:
        folder = "../masks_data"
    mkdir(folder)
    pickle.dump(opt_dict, open(f"{folder}/masks_dict_H{image_height}"
                               f"_W{image_width}_Min{opt.min_hole_size}"
                               f"_Max{opt.max_hole_size}.p", "wb"))
    return masks, indexes


def generate_latent_data(opt):
    import pickle
    z_appr = torch.FloatTensor(opt.how_many, opt.z_len, 1, 1).normal_(0, 1)
    opt_dict = {}
    opt_dict["z_latent_appr"] = z_appr
    if opt.dataset.lower() == "indiandrivingdataset":
        folder = "../latent_data_idd"
    else:
        folder = "../latent_data"
    mkdir(folder)
    pickle.dump(opt_dict, open(f"{folder}/latent_data_Z{opt.z_len}_MANY_{opt.how_many}", "wb"))
    return z_appr


def load_masks_data(opt):
    import pickle
    image_height = opt.size_crop_height if opt.size_crop_height else opt.image_height
    image_width = opt.size_crop_width if opt.size_crop_width else opt.image_width
    if opt.dataset.lower() == "indiandrivingdataset":
        path = f"../masks_data_idd/masks_dict_H{image_height}" \
               f"_W{image_width}_Min{opt.min_hole_size}" \
               f"_Max{opt.max_hole_size}.p"
    else:
        path = f"../masks_data/masks_dict_H{image_height}" \
           f"_W{image_width}_Min{opt.min_hole_size}" \
           f"_Max{opt.max_hole_size}.p"
    if os.path.exists(path):
        opt_dict = pickle.load(open(path, "rb"))
        return opt_dict["masks"], opt_dict["indexes"]
    else:
        print("Mask data does not exists, generate new ones.")
        return generare_masks_data(opt)


def load_latent_vector_appr(opt):
    import pickle
    if opt.dataset.lower() == "indiandrivingdataset":
        path = f"../latent_data_idd/latent_data_Z{opt.z_len}_MANY_{opt.how_many}"
    else:
        path = f"../latent_data/latent_data_Z{opt.z_len}_MANY_{opt.how_many}"
    if os.path.exists(path):
        opt_dict = pickle.load(open(path, "rb"))
        return opt_dict["z_latent_appr"]
    else:
        print("Latent data does not exists, generate new ones.")
        return generate_latent_data(opt)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _random_mask(img_shape, min_hole, max_hole, fix_seed=False):
    cur_rows, cur_cols = np.random.randint(min_hole, max_hole, size=2)
    cur_y = np.random.randint(0, img_shape[0] - cur_rows)
    cur_x = np.random.randint(0, img_shape[1] - cur_cols)
    cur_mask = np.full((img_shape[0], img_shape[1]), 0, dtype='uint8')
    #cur_mask = np.zeros([img.shape[1], img.shape[2]], dtype='uint8')
    cur_mask[cur_y:cur_y + cur_rows, cur_x:cur_x + cur_cols] = 255
    return cur_mask, [cur_y, cur_y + cur_rows, cur_x, cur_x + cur_cols]


def _center_mask(img_shape, min_hole, max_hole, fix_seed=False):
    cur_rows, cur_cols = min_hole, max_hole
    cur_y = int((img_shape[0] / 2) - (cur_rows / 2))
    cur_x = int((img_shape[1] / 2) - (cur_cols / 2))
    cur_mask = np.full((img_shape[0], img_shape[1]), 0, dtype='uint8')
    #cur_mask = np.zeros([img.shape[1], img.shape[2]], dtype='uint8')
    cur_mask[cur_y:cur_y + cur_rows, cur_x:cur_x + cur_cols] = 255
    return cur_mask, [cur_y, cur_y + cur_rows, cur_x, cur_x + cur_cols]


def _random_mask_with_instance_cond(img_shape, min_hole, max_hole, inst_maps, seed):
    if seed is not None:
        np.random.seed(seed)
    cur_rows_random, cur_cols_random = np.random.randint(min_hole, max_hole, size=2)
    yy, xx = np.where(inst_maps.view(img_shape[0],img_shape[1]) > 0)
    height = int(yy.max() - yy.min() + 1)
    width = int(xx.max() - xx.min() + 1)
    cur_rows = max(cur_rows_random, min(height + min_hole, max_hole))
    cur_cols = max(cur_cols_random, min(width + min_hole, max_hole))
    cur_y = np.random.randint(max(0,yy.max() - cur_rows), min(yy.min(), img_shape[0] - cur_rows) + 1)
    cur_x = np.random.randint(max(0,xx.max() - cur_cols), min(xx.min(), img_shape[1] - cur_cols) + 1)

    cur_mask = np.full((img_shape[0], img_shape[1]), 0, dtype='uint8')
    #cur_mask = np.zeros([img.shape[1], img.shape[2]], dtype='uint8')
    cur_mask[cur_y:cur_y + cur_rows, cur_x:cur_x + cur_cols] = 255
    assert cur_y <= yy.min() <= cur_y + cur_rows
    assert cur_x <= xx.min() <= cur_x + cur_cols
    assert cur_y <= yy.max() <= cur_y + cur_rows
    assert cur_x <= xx.max() <= cur_x + cur_cols
    assert cur_y + cur_rows <= img_shape[0]
    assert cur_x + cur_cols <= img_shape[1]
    return cur_mask, (cur_y, cur_y + cur_rows, cur_x, cur_x + cur_cols)



def instInCrop(bbox, indexes, offset = 0):
    if not indexes:
        return True
    return bbox[0] >= (indexes[0] + offset) and bbox[2] <= (indexes[0] + indexes[2] - offset) and\
           bbox[1] >= (indexes[1] + offset) and bbox[3] <= (indexes[1] + indexes[3] - offset)


def instInMap(bbox, indexes, crop_indexes):
    if not indexes:
        return True
    return bbox[0] >= (indexes[0] + crop_indexes[0]) and bbox[2] <= indexes[1] + (crop_indexes[0]) and\
           bbox[1] >= (indexes[2] + crop_indexes[1]) and bbox[3] <= (indexes[3] + (crop_indexes[1]))

def local_patch(x, bbox):
    assert len(x.size()) == 4
    patches = []
    for i in range(x.size(0)):
        h, h_final, w, w_final = bbox
        patches.append(x[i, :, h:h_final, w:w_final])
    return torch.stack(patches, dim=0)

def inst_map2bbox(maps, opt):
    x = maps.squeeze(0)
    bbox = torch.zeros(x.shape)
    yy, xx = np.where(x.view(x.shape[0], x.shape[1]) > 0)
    yt, yb = float(yy.min()) / x.shape[0] * 2 - 1, float(
            yy.max() + 1) / x.shape[0] * 2 - 1
    xl, xr = float(xx.min()) / x.shape[1] * 2 - 1, float(
            xx.max() + 1) / x.shape[1] * 2 - 1
    theta11 = 2 / (xr - xl)
    theta13 = 1 - 2 * xr / (xr - xl)
    theta22 = 2 / (yb - yt)
    theta23 = 1 - 2 * yb / (yb - yt)
    bbox[yy.min():yy.max()+1, xx.min():xx.max()+1] = 1
    compact = x.view(x.shape[0], x.shape[1])[yy.min():yy.max()+1, xx.min():xx.max()+1]
    resize_img = transform.Resize((opt.compact_sizex,
                                       opt.compact_sizey),
                                      Image.NEAREST)
    to_pil = transform.ToPILImage()
    to_tensor = transform.ToTensor()
    b_theta = torch.from_numpy(np.array([theta11, 0., theta13, 0., theta22, theta23])).float()
    inst_compact = to_tensor(resize_img(to_pil(compact)))
    bboxes = (bbox.unsqueeze(0))
    return bboxes, inst_compact, b_theta


def extract_coords(maps_bboxes):
    bboxes = []
    for x in maps_bboxes.squeeze(1):
        yy, xx = torch.where(x.view(x.shape[0], x.shape[1]) > 0)
        bboxes.append(torch.cuda.FloatTensor([yy.min(), yy.max() + 1, xx.min(), xx.max() + 1]))
    return torch.stack(bboxes, dim=0)


def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()

    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks


def spatial_discounting_mask(opt, indexes, gpu_ids):
    """Generate spatial discounting mask constant.

    Spatial discounting mask is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.

    Args:
        config: Config should have configuration including HEIGHT, WIDTH,
            DISCOUNTED_MASK.

    Returns:
        tf.Tensor: spatial discounting mask

    """
    gamma = opt.spatial_discounting_gamma
    height, width = indexes[1] - indexes[0], indexes[3] - indexes[2]
    shape = [1, 1, height, width]
    if opt.use_discounted_mask:
        mask_values = np.ones((height, width))
        for i in range(height):
            for j in range(width):
                mask_values[i, j] = max(
                    gamma ** min(i, height - i),
                    gamma ** min(j, width - j))
        mask_values = np.expand_dims(mask_values, 0)
        mask_values = np.expand_dims(mask_values, 0)
    else:
        mask_values = np.ones(shape)
    spatial_discounting_mask_tensor = torch.tensor(mask_values, dtype=torch.float32)
    if len(gpu_ids) > 0:
        spatial_discounting_mask_tensor = spatial_discounting_mask_tensor.cuda()
    return spatial_discounting_mask_tensor


def normalize(tensors, mean, std):
    """
    Normalize tensor with std and mean
    :param tensors: input tensor
    :param mean: mean of normalization
    :param std: std of normalization
    :return: normalized tensor
    """
    if not torch.is_tensor(tensors):
        raise TypeError('tensor is not a torch image.')
    for tensor in tensors:
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
    return tensors


def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x


def reduce_std(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.std(x, dim=i, keepdim=keepdim)
    return x


def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x


def reduce_tensor(tensor, world_size):
    import torch.distributed as dist
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt


def flow_to_image(flow):
    """Transfer flow map to image.
    Part of code forked from flownet.
    """
    out = []
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxrad = -1
    for i in range(flow.shape[0]):
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))
        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(maxrad, np.max(rad))
        u = u / (maxrad + np.finfo(float).eps)
        v = v / (maxrad + np.finfo(float).eps)
        img = compute_color(u, v)
        out.append(img)
    return np.float32(np.uint8(out))


def pt_flow_to_image(flow):
    """Transfer flow map to image.
    Part of code forked from flownet.
    """
    out = []
    maxu = torch.tensor(-999)
    maxv = torch.tensor(-999)
    minu = torch.tensor(999)
    minv = torch.tensor(999)
    maxrad = torch.tensor(-1)
    if torch.cuda.is_available():
        maxu = maxu.cuda()
        maxv = maxv.cuda()
        minu = minu.cuda()
        minv = minv.cuda()
        maxrad = maxrad.cuda()
    for i in range(flow.shape[0]):
        u = flow[i, 0, :, :]
        v = flow[i, 1, :, :]
        idxunknow = (torch.abs(u) > 1e7) + (torch.abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = torch.max(maxu, torch.max(u))
        minu = torch.min(minu, torch.min(u))
        maxv = torch.max(maxv, torch.max(v))
        minv = torch.min(minv, torch.min(v))
        rad = torch.sqrt((u ** 2 + v ** 2).float()).to(torch.int64)
        maxrad = torch.max(maxrad, torch.max(rad))
        u = u / (maxrad + torch.finfo(torch.float32).eps)
        v = v / (maxrad + torch.finfo(torch.float32).eps)
        img = pt_compute_color(u, v)
        out.append(img)

    return torch.stack(out, dim=0)


def highlight_flow(flow):
    """Convert flow into middlebury color code image.
    """
    out = []
    s = flow.shape
    for i in range(flow.shape[0]):
        img = np.ones((s[1], s[2], 3)) * 144.
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        for h in range(s[1]):
            for w in range(s[1]):
                ui = u[h, w]
                vi = v[h, w]
                img[ui, vi, :] = 255.
        out.append(img)
    return np.float32(np.uint8(out))


def pt_highlight_flow(flow):
    """Convert flow into middlebury color code image.
        """
    out = []
    s = flow.shape
    for i in range(flow.shape[0]):
        img = np.ones((s[1], s[2], 3)) * 144.
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        for h in range(s[1]):
            for w in range(s[1]):
                ui = u[h, w]
                vi = v[h, w]
                img[ui, vi, :] = 255.
        out.append(img)
    return np.float32(np.uint8(out))


def compute_color(u, v):
    h, w = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    # colorwheel = COLORWHEEL
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0
    for i in range(np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))
    return img


def pt_compute_color(u, v):
    h, w = u.shape
    img = torch.zeros([3, h, w])
    if torch.cuda.is_available():
        img = img.cuda()
    nanIdx = (torch.isnan(u) + torch.isnan(v)) != 0
    u[nanIdx] = 0.
    v[nanIdx] = 0.
    # colorwheel = COLORWHEEL
    colorwheel = pt_make_color_wheel()
    if torch.cuda.is_available():
        colorwheel = colorwheel.cuda()
    ncols = colorwheel.size()[0]
    rad = torch.sqrt((u ** 2 + v ** 2).to(torch.float32))
    a = torch.atan2(-v.to(torch.float32), -u.to(torch.float32)) / np.pi
    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = torch.floor(fk).to(torch.int64)
    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0.to(torch.float32)
    for i in range(colorwheel.size()[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1]
        col1 = tmp[k1 - 1]
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1. / 255.
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = (idx != 0)
        col[notidx] *= 0.75
        img[i, :, :] = col * (1 - nanIdx).to(torch.float32)
    return img


def make_color_wheel():
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255
    return colorwheel


def pt_make_color_wheel():
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = torch.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 1.
    colorwheel[0:RY, 1] = torch.arange(0, RY, dtype=torch.float32) / RY
    col += RY
    # YG
    colorwheel[col:col + YG, 0] = 1. - (torch.arange(0, YG, dtype=torch.float32) / YG)
    colorwheel[col:col + YG, 1] = 1.
    col += YG
    # GC
    colorwheel[col:col + GC, 1] = 1.
    colorwheel[col:col + GC, 2] = torch.arange(0, GC, dtype=torch.float32) / GC
    col += GC
    # CB
    colorwheel[col:col + CB, 1] = 1. - (torch.arange(0, CB, dtype=torch.float32) / CB)
    colorwheel[col:col + CB, 2] = 1.
    col += CB
    # BM
    colorwheel[col:col + BM, 2] = 1.
    colorwheel[col:col + BM, 0] = torch.arange(0, BM, dtype=torch.float32) / BM
    col += BM
    # MR
    colorwheel[col:col + MR, 2] = 1. - (torch.arange(0, MR, dtype=torch.float32) / MR)
    colorwheel[col:col + MR, 0] = 1.
    return colorwheel

###############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    if N == 35: # cityscape
        cmap = np.array([(  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (111, 74,  0), ( 81,  0, 81),
                     (128, 64,128), (244, 35,232), (250,170,160), (230,150,140), ( 70, 70, 70), (102,102,156), (190,153,153),
                     (180,165,180), (150,100,100), (150,120, 90), (153,153,153), (153,153,153), (250,170, 30), (220,220,  0),
                     (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
                     (  0, 60,100), (  0,  0, 90), (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,142)],
                     dtype=np.uint8)
    elif N == 29:
        cmap = np.array([(128, 64,128), (244, 35,232), (250,170,160), (230,150,140), ( 70, 70, 70), (102,102,156), (190,153,153),
                     (180,165,180), (150,100,100), (150,120, 90), (153,153,153), (153,153,153), (250,170, 30), (220,220,  0),
                     (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
                     (  0, 60,100), (  0,  0, 90), (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32), (0,  0, 0), (255,255,255)],
                     dtype=np.uint8)
    elif N == 18:
        cmap = np.array([(128, 64,128), (244, 35,232), ( 70, 70, 70), (102,102,156), (220,220,  0),
                     (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
                     (  0, 60,100), (  0, 80,100), (  0,  0,230), (119, 11, 32), (0,  0, 0), (255,255,255)],
                     dtype=np.uint8)
    elif N == 9:
        cmap = np.array(
            [(128, 64, 128), (70, 70, 70), (220, 220, 0), (107, 142, 35),
             (70, 130, 180), (220, 20, 60), (0, 0, 142), (0, 0, 0), (255,255,255)],
            dtype=np.uint8)
    elif N == 17:
        cmap = np.array(
            [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
             (220, 220, 0),
             (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
             (255, 0, 0), (0, 0, 142), (0, 0, 70),
             (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 0)],
            dtype=np.uint8)
    elif N == 8:
        cmap = np.array(
            [(128, 64, 128), (70, 70, 70), (220, 220, 0), (107, 142, 35),
             (70, 130, 180), (220, 20, 60), (0, 0, 142), (0, 0, 0)],
            dtype=np.uint8)
    elif N == 21:
        cmap = np.array(
            [(128, 64, 128), ( 81,  0, 81), (244, 35, 232), (152,251,152), (220, 20, 60),
             (246, 198, 145), (255, 0, 0), (0, 0, 230), (119, 11, 32), (255, 204, 54),
            (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (136, 143, 153),
             (102, 102, 156), ( 70, 70, 70), (220,220,  0),
             (107, 142, 35), (70, 130, 180), (0, 0, 0)],
            dtype=np.uint8)
    elif N == 22:
        cmap = np.array(
            [(128, 64, 128), (81, 0, 81), (244, 35, 232), (152, 251, 152),
             (220, 20, 60),
             (246, 198, 145), (255, 0, 0), (0, 0, 230), (119, 11, 32),
             (255, 204, 54),
             (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100),
             (136, 143, 153),
             (102, 102, 156), (70, 70, 70), (220, 220, 0),
             (107, 142, 35), (70, 130, 180), (0, 0, 0), (255,255,255)],
            dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.show()




def plot_grad_flow_v2(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.4)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)],
               ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()


class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


