import os
from torch.utils import data
from tqdm import tqdm
from datasets import get_segmentation_dataset
import torchvision.transforms as transform
import torch
from options.train_options import TrainOptions
import numpy as np
import math
from models import SPNetModel, SGNetModel, SPNetInferenceModel, SPGNetModel,\
    SPGNetInferenceModel, SLGNetModel, SLGNetInferenceModel, SLGNetModelFineTune, SLGNetInferenceModelFineTune, SPNetModelPaper, SGNetModelPaper
import time
import utils.util as util
from utils import html
from utils.visualizer import Visualizer
from collections import OrderedDict
from utils.psnr import psnr
from pytorch_msssim import ssim
import models.functions.lpips
import models.functions.yolo.models as yolo_model
import models.functions.yolo.utils.utils as yolo_utils
from skimage.measure import compare_psnr
from utils.fid import calculate_fid
torch.set_printoptions(precision=10)
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
import itertools
from sklearn.metrics import f1_score
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

def lcm(a, b): return abs(a * b) / math.gcd(a, b) if a and b else 0



mask_transform = transform.Compose([
    transform.ToTensor()])

input_transform = transform.Compose([
    transform.ToTensor(),
    transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
gt_transform = transform.Compose([
    transform.ToTensor()])
inst_map_transform = transform.Compose([
    transform.ToTensor()])
# dataset
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.backends.cudnn.benchmark = True

device = \
    [torch.device(f"cuda:0" if use_cuda else "cpu")]

def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)


def overlap(a, b):  # returns None if rectangles don't intersect
    from collections import namedtuple
    Rectangle = namedtuple('Rectangle', 'ymin xmin ymax xmax')
    height = float(a[2] - a[0] + 1)
    width = float(a[3] - a[1] + 1)
    a = Rectangle(a[0],a[1], a[2], a[3])
    b = Rectangle(b[0], b[1], b[2], b[3])
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx>=0) and (dy>=0):
        if dx * dy > 0.30 * height * width:
            return True
        else:
            return False
    else:
        return False



def compute_detection(output_images, model_yolo,
                      images, model_input, classes,
                      count_detection_cond, count_detection_no_cond, colors, web_dir, images_path, crop_type,
                      inst_maps_valid_idx, cond):
    completed_images = output_images.to(device[0])
    image = F.interpolate(completed_images, size=416,
                          mode="nearest")
    image = (image + 1) * 0.5
    with torch.no_grad():
        detections = model_yolo(image)
        detections = \
            yolo_model.non_max_suppression(detections, 0.90,
                                           0.4)[0]
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(util.tensor2im(completed_images[0].cpu()))
    images.append(1)
    # Draw bounding boxes and labels of detections
    if detections is not None:
        # Rescale boxes to original image
        detections = yolo_utils.rescale_boxes(detections, 416,
                                              completed_images.shape[
                                              2:])
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        import random
        bbox_colors = random.sample(colors, n_cls_preds)
        if opt.label_nc == 17:
            id_to_text = {"24": "person", "26": "car"}
        else:
            id_to_text = {"4": "person", "10": "car"}
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            if cond:
                yy, xx = np.where(
                    util.tensor2im(model_input["insta_maps_bbox"][0],
                                   normalize=False) > 0)
                bbox_y_min = yy.min()
                bbox_x_min = xx.min()
                bbox_y_max = yy.max()
                bbox_x_max = xx.max()
            else:
                bbox_y_min = model_input["indexes"][0][0]
                bbox_x_min = model_input["indexes"][0][2]
                bbox_y_max = model_input["indexes"][0][1]
                bbox_x_max = model_input["indexes"][0][3]
            if not overlap([bbox_y_min, bbox_x_min, bbox_y_max, bbox_x_max],
                           [int(y1), int(x1), int(y2), int(x2)]):
                continue
            if cond:
                if classes[int(cls_pred)] == id_to_text[
                    str(int(inst_maps_valid_idx[0].item()))[:-3]]:
                    count_detection_cond.append(1)
                else:
                    continue
            else:
                if classes[int(cls_pred)] in ["car", "person"]:
                    count_detection_no_cond.append(1)
                else:
                    continue
            print("\t+ Label: %s, Conf: %.5f" % (
                classes[int(cls_pred)], cls_conf.item()))

            box_w = x2 - x1
            box_h = y2 - y1

            color = bbox_colors[
                int(np.where(unique_labels == int(cls_pred))[0])]
            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), box_w, box_h,
                                     linewidth=2, edgecolor=color,
                                     facecolor="none")
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            plt.text(
                x1,
                y1,
                s=classes[int(cls_pred)],
                color="white",
                verticalalignment="top",
                bbox={"color": color, "pad": 0},
            )

    # Save generated image with detections
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    img_dir = os.path.join(web_dir, 'images')
    if cond:
        filename = f"{images_path[0][:-4].split('/')[-1]}_{crop_type}_detection_cond"
    else:
        filename = f"{images_path[0][:-4].split('/')[-1]}_{crop_type}_detection_no_cond"
    plt.savefig(f"{img_dir}/{filename}.png", bbox_inches="tight",
                pad_inches=0.0)
    plt.close()
    from PIL import Image
    detection_image = input_transform(
        Image.open(f"{img_dir}/{filename}.png").convert('RGB'))
    os.remove(f"{img_dir}/{filename}.png")
    return detection_image


def test(opt):
    import glob
    epochs = [f"epoch_{x.split('/')[-1].split('_')[0]}" for x in glob.glob(os.path.join(opt.test_model_sg) + "/latest*SG*_G.pth")]
    #test_z_appr = util.load_latent_vector_appr(opt)
    model_yolo = yolo_model.Darknet("models/functions/yolo/config/yolov3.cfg",
                                    img_size=256).to(device[0])
    model_yolo.load_darknet_weights(
        "models/functions/yolo/weights/yolov3.weights")
    model_yolo.eval()
    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    classes = yolo_utils.load_classes("models/functions/yolo/data/coco.names")
    for epoch in epochs:
        old_epoch_folder = epoch.split("_")[1]
        opt.which_epoch = epoch
        web_dir = os.path.join(opt.checkpoints_dir, "SPG-NET", opt.name, "test", opt.which_epoch, "results")
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (
        opt.name, opt.phase, opt.which_epoch))

        visualizer = Visualizer(opt)
        opt.which_epoch = old_epoch_folder
        model = SPGNetInferenceModel(opt)
        if use_cuda:
            model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
        opt.which_epoch = epoch
        real_images = []
        fake_images = []
        fake_images_no_cond = []
        l1 = 0
        l2 = 0
        psnr_all = []
        ssim_score = 0
        ssim_score_no_cond = 0
        psnr_score = 0
        psnr_score_no_cond = 0
        print(f"epoch: {epoch} \n")
        count = 0
        images_cond = []
        images_no_cond = []
        count_detection_cond = []
        count_detection_no_cond = []
        for crop_type in ["left", "center", "right"]:
            test_loader_kwargs = {'transform': input_transform,
                                  "gt_transform": gt_transform,
                                  "inst_map_transform": inst_map_transform,
                                  "root": opt.dataroot, "opt": opt,
                                  "crop_type": crop_type}
            testset = get_segmentation_dataset(opt.dataset, split='test',
                                               mode='test',
                                               **test_loader_kwargs)

            testloader = data.DataLoader(testset, batch_size=1,
                                         drop_last=False, shuffle=False,
                                         num_workers=1)
            tbar = tqdm(testloader, leave=True)
            for i, input_dict in enumerate(tbar):
                if -1 in input_dict["inst_map_valid_idx"]:
                    continue
                count += 1
                gt_images = input_dict["gt_images"]
                gt_seg_maps = input_dict["gt_seg_maps"]
                inst_maps = input_dict["inst_map"]
                inst_maps_valid_idx = input_dict["inst_map_valid_idx"]
                images_path = input_dict["images_path"]
                masks = input_dict["masks"]
                indexes = input_dict["indexes"]
                insta_maps_bbox = input_dict["insta_maps_bbox"]
                inst_map_compact = input_dict["inst_map_compact"]
                theta = input_dict["theta"]
                compute_instance = input_dict["compute_instance"]
                images_masked = (
                        gt_images.to(device[0]) - gt_images.to(device[0]) * masks.to(
                    device[0]))
                one_hot_seg_map = torch.FloatTensor(opt.batchSize, opt.semantic_nc,
                                                    gt_seg_maps.shape[2],
                                                    gt_seg_maps.shape[
                                                        3]).zero_().to(device[0])
                one_hot_seg_map = one_hot_seg_map.scatter(1, gt_seg_maps.type(
                    torch.LongTensor).to(device[0]), 1)

                gt_seg_maps_masked = gt_seg_maps.clone().type(torch.LongTensor).to(
                    device[0])
                gt_seg_maps_masked[masks == 1.0] = opt.label_nc - 1 if opt.no_contain_dontcare_label else opt.label_nc
                one_hot_gt_seg_maps_masked = torch.FloatTensor(1, opt.semantic_nc,
                                                               gt_seg_maps.shape[2],
                                                               gt_seg_maps.shape[
                                                                   3]).zero_().to(
                    device[0])
                one_hot_gt_seg_maps_masked = \
                    one_hot_gt_seg_maps_masked.scatter_(1, gt_seg_maps_masked, 1.0)
                model_input = {
                    "one_hot_gt_seg_maps_masked": one_hot_gt_seg_maps_masked,
                    "gt_images_masked": images_masked,
                    "masks": masks.to(device[0]),
                    "insta_maps_bbox": insta_maps_bbox.to(device[0]),
                    "inst_maps": inst_maps.to(device[0]),
                    "inst_maps_compact": inst_map_compact.to(device[0]),
                    "theta_transform": theta.to(device[0]),
                    "inst_maps_valid_idx": inst_maps_valid_idx,
                    "compute_instance": compute_instance,
                    "use_gt_instance_encoder": opt.use_gt_instance_encoder,
                    "indexes": indexes,
                    "test_z_appr": torch.FloatTensor(1, opt.z_len, 1, 1).normal_(0, 1)}
                generated_images, generated_images_no_cond, generated_instances_pad, generated_instances, generated_seg_maps, generated_seg_maps_no_cond = model.forward(
                    model_input)
                output_images = generated_images.detach().cpu()
                output_images_no_cond = generated_images_no_cond.detach().cpu()
                output_seg_maps = generated_seg_maps.detach().cpu()
                output_seg_maps_no_cond = generated_seg_maps_no_cond.detach().cpu()
                if generated_instances[0] is not None:
                    output_gen_instances = generated_instances_pad.detach().cpu()
                    inst_map_bbox = insta_maps_bbox
                    inst_map = inst_maps
                detection_cond = compute_detection(output_images, model_yolo,
                                                   images_cond, model_input, classes, count_detection_cond, count_detection_no_cond, colors, web_dir, images_path, crop_type, inst_maps_valid_idx, True)
                detection_no_cond = compute_detection(output_images_no_cond,model_yolo,
                                                   images_no_cond, model_input, classes, count_detection_cond, count_detection_no_cond, colors, web_dir, images_path, crop_type, inst_maps_valid_idx, False)
                visuals = OrderedDict([('gt_seg_map',
                                        util.tensor2label(gt_seg_maps[0],
                                                          opt.label_nc)),
                                       ('gt_seg_map_masked',
                                        util.tensor2label(
                                            one_hot_gt_seg_maps_masked[0],
                                            opt.semantic_nc)),
                                       ('real_image',
                                        util.tensor2im(gt_images[0])),
                                       ('inst_map',
                                        util.tensor2im(inst_map[0])),
                                       ('inst_bbox',
                                        util.tensor2im(inst_map_bbox[0])),
                                       ('masked_image',
                                        util.tensor2im(images_masked.cpu()[0])),
                                       ('fake_image',
                                        util.tensor2im(output_images[0])),
                                       ("fake_instance",
                                        util.tensor2label(
                                            output_gen_instances[0],
                                            opt.semantic_nc)),
                                       ('fake_seg_maps',
                                        util.tensor2label(output_seg_maps[0],
                                                          opt.semantic_nc)),
                                       ('fake_seg_maps_no_cond',
                                        util.tensor2label(output_seg_maps_no_cond[0],
                                                          opt.semantic_nc)),
                                       ('reconstructed_image_cond',
                                        util.tensor2im(output_images[0])),
                                       ('reconstructed_image_no_cond',
                                        util.tensor2im(output_images_no_cond[0])),
                                       ('detection_image_cond',
                                        util.tensor2im(detection_cond)),
                                       ('detection_image_no_cond',
                                        util.tensor2im(detection_no_cond)),
                                       ])
                visualizer.save_images(webpage, visuals, [
                    f"{images_path[0][:-4]}_{crop_type}_{images_path[0][-4:]}"])

            webpage.save()
        count_detection_cond.extend([0 for _ in range(len(images_cond) - len(count_detection_cond))])
        count_detection_no_cond.extend(
            [0 for _ in range(len(images_no_cond) - len(count_detection_no_cond))])
        results = {"F1_score_cond": f1_score(images_cond, count_detection_cond),
                   "F1_score_no_cond": f1_score(images_cond, count_detection_no_cond)}
        print(results)
        visualizer.print_results(results)

def test_slg(opt):
    folder_name = "SPG-NET" if any(
        x in opt.name for x in ["SG-NET", "SP-NET", "SPG-NET"]) else "SLG-NET"
    import glob
    epochs = [f"epoch_{x.split('/')[-1].split('_')[0]}" for x in glob.glob(os.path.join(opt.checkpoints_dir, folder_name, opt.name) + "/latest*SLG*_G.pth")]
    #test_z_appr = util.load_latent_vector_appr(opt)
    best_psnr = []
    model_yolo = yolo_model.Darknet("models/functions/yolo/config/yolov3.cfg", img_size=256).to(device[0])
    model_yolo.load_darknet_weights("models/functions/yolo/weights/yolov3.weights")
    model_yolo.eval()
    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    classes = yolo_utils.load_classes("models/functions/yolo/data/coco.names")
    for epoch in epochs:
        old_epoch_folder = epoch.split("_")[1]
        opt.which_epoch = epoch
        web_dir = os.path.join(opt.checkpoints_dir, folder_name, opt.name, "test", opt.which_epoch, "results")
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (
        opt.name, opt.phase, opt.which_epoch))

        visualizer = Visualizer(opt)
        opt.which_epoch = old_epoch_folder
        if opt.fineTuning:
            model = SLGNetInferenceModelFineTune(opt)
        else:
            model = SLGNetInferenceModel(opt)
        if use_cuda:
            model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
        opt.which_epoch = epoch
        count = 0
        lpips_images = []
        images_cond = []
        images_no_cond = []
        count_detection_cond = []
        count_detection_no_cond = []
        for crop_type in ["left", "center", "right"]:
            print(f"epoch: {epoch} crop_type:{crop_type}\n")
            test_loader_kwargs = {'transform': input_transform,
                                  "gt_transform": gt_transform,
                                  "inst_map_transform": inst_map_transform,
                                  "root": opt.dataroot, "opt": opt,
                                  "crop_type": crop_type}
            testset = get_segmentation_dataset(opt.dataset, split='test',
                                               mode='test',
                                               **test_loader_kwargs)

            testloader = data.DataLoader(testset, batch_size=1,
                                         drop_last=False, shuffle=False,
                                         num_workers=1)
            tbar = tqdm(testloader, leave=True)
            for i, input_dict in enumerate(tbar):
                if -1 in input_dict["inst_map_valid_idx"]:
                    continue
                count += 1
                gt_images = input_dict["gt_images"]
                gt_seg_maps = input_dict["gt_seg_maps"]
                inst_maps = input_dict["inst_map"]
                inst_maps_valid_idx = input_dict["inst_map_valid_idx"]
                images_path = input_dict["images_path"]
                masks = input_dict["masks"]
                indexes = input_dict["indexes"]
                insta_maps_bbox = input_dict["insta_maps_bbox"]
                inst_map_compact = input_dict["inst_map_compact"]
                theta = input_dict["theta"]
                compute_instance = input_dict["compute_instance"]
                images_masked = (
                        gt_images.to(device[0]) - gt_images.to(device[0]) * masks.to(
                    device[0]))
                one_hot_seg_map = torch.FloatTensor(opt.batchSize, opt.semantic_nc,
                                                    gt_seg_maps.shape[2],
                                                    gt_seg_maps.shape[
                                                        3]).zero_().to(device[0])

                gt_seg_maps_masked = gt_seg_maps.clone().type(torch.LongTensor).to(
                    device[0])
                gt_seg_maps_masked[masks == 1.0] = opt.label_nc - 1 if opt.no_contain_dontcare_label else opt.label_nc
                one_hot_gt_seg_maps_masked = torch.FloatTensor(1, opt.semantic_nc,
                                                               gt_seg_maps.shape[2],
                                                               gt_seg_maps.shape[
                                                                   3]).zero_().to(
                    device[0])
                one_hot_gt_seg_maps_masked = \
                    one_hot_gt_seg_maps_masked.scatter_(1, gt_seg_maps_masked, 1.0)
                model_input = {
                    "one_hot_gt_seg_maps_masked": one_hot_gt_seg_maps_masked,
                    "gt_images_masked": images_masked,
                    "masks": masks.to(device[0]),
                    "insta_maps_bbox": insta_maps_bbox.to(device[0]),
                    "inst_maps": inst_maps.to(device[0]),
                    "inst_maps_compact": inst_map_compact.to(device[0]),
                    "theta_transform": theta.to(device[0]),
                    "inst_maps_valid_idx": inst_maps_valid_idx,
                    "compute_instance": compute_instance,
                    "indexes": indexes,
                    "test_z_appr": torch.FloatTensor(1, opt.z_len, 1, 1).normal_(0, 1),
                    "use_gt_instance_encoder": opt.use_gt_instance_encoder}
                generated_images, generated_instances, generated_seg_maps, offset_flow, one_hot_gt_seg_maps_masked_inst, res_no_cond = model.forward(
                    model_input)
                output_images = generated_images.detach().cpu()
                output_images_no_cond = res_no_cond[0].detach().cpu()
                output_seg_maps = generated_seg_maps.detach().cpu()
                output_seg_maps_no_cond = res_no_cond[1].detach().cpu()
                one_hot_gt_seg_maps_masked_inst = one_hot_gt_seg_maps_masked_inst.detach().cpu()
                one_hot_gt_seg_maps_masked_inst_no_cond = res_no_cond[-1].detach().cpu()
                if generated_instances[0] is not None:
                    output_gen_instances = generated_instances.detach().cpu()
                    inst_map_bbox = insta_maps_bbox
                    inst_map = inst_maps
                detection_cond = compute_detection(output_images, model_yolo,
                                                   images_cond, model_input, classes,
                                                   count_detection_cond,
                                                   count_detection_no_cond,
                                                   colors, web_dir, images_path,
                                                   crop_type,
                                                   inst_maps_valid_idx, True)
                detection_no_cond = compute_detection(output_images_no_cond,
                                                      model_yolo,
                                                      images_no_cond, model_input,
                                                      classes,
                                                      count_detection_cond,
                                                      count_detection_no_cond,
                                                      colors, web_dir,
                                                      images_path, crop_type,
                                                      inst_maps_valid_idx,
                                                      False)
                visuals = OrderedDict([('gt_seg_map',
                                        util.tensor2label(gt_seg_maps[0],
                                                          opt.label_nc)),
                                       ('gt_seg_map_masked',
                                        util.tensor2label(
                                            one_hot_gt_seg_maps_masked_inst[0],
                                            opt.semantic_nc)),
                                       ('real_image',
                                        util.tensor2im(gt_images[0])),
                                       ('inst_map',
                                        util.tensor2im(inst_map[0])),
                                       ('inst_bbox',
                                        util.tensor2im(inst_map_bbox[0])),
                                       ('masked_image',
                                        util.tensor2im(images_masked.cpu()[0])),
                                       ("expected_flow",
                                        util.tensor2im(offset_flow[0])),
                                       ('fake_image',
                                        util.tensor2im(output_images[0])),
                                       ("fake_instance",
                                        util.tensor2label(
                                            output_gen_instances[0],
                                            opt.semantic_nc)),
                                       ('fake_seg_maps',
                                        util.tensor2label(output_seg_maps[0],
                                                          opt.semantic_nc)),
                                       ('fake_seg_maps_no_cond',
                                        util.tensor2label(output_seg_maps_no_cond[0],
                                                          opt.semantic_nc)),
                                       ('reconstructed_image_cond',
                                        util.tensor2im(output_images[0])),
                                       ('reconstructed_image_no_cond',
                                        util.tensor2im(output_images_no_cond[0])),
                                       ('detection_image_cond',
                                        util.tensor2im(detection_cond)),
                                       ('detection_image_no_cond',
                                        util.tensor2im(detection_no_cond)),
                                       ])
                visualizer.save_images(webpage, visuals, [f"{images_path[0][:-4]}_{crop_type}_{images_path[0][-4:]}"])


            webpage.save()
        count_detection_cond.extend([0 for _ in range(len(images_cond) - len(count_detection_cond))])
        count_detection_no_cond.extend(
            [0 for _ in range(len(images_no_cond) - len(count_detection_no_cond))])
        results = {"F1_score_cond": f1_score(images_cond, count_detection_cond),
                   "F1_score_no_cond": f1_score(images_cond, count_detection_no_cond)}
        print(results)
        visualizer.print_results(results)



if __name__ == "__main__":
    opt = TrainOptions().parse()
    opt.no_flip = True
    opt.isTrain = False
    if opt.model_phase == "SPG-NET":
        test(opt)
    else:
        test_slg(opt)
