import os
from torch.utils import data
from tqdm import tqdm
from datasets import get_segmentation_dataset
import torchvision.transforms as transform
import torch
from options.train_options import TrainOptions
import numpy as np
import math
from models import SGINetModel, SGINetInferenceModel
import time
import utils.util as util
from utils import html
from utils.visualizer import Visualizer
from collections import OrderedDict
from utils.psnr import psnr
from pytorch_msssim import ssim
import torch.distributed as dist
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel
from skimage.measure import compare_psnr
from utils.fid import calculate_fid
torch.set_printoptions(precision=10)
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

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


def train(gpu, opt):
    device = torch.device(f"cuda:{opt.gpu_ids[0]}" if use_cuda else "cpu")
    opt.print_freq = lcm(opt.print_freq, opt.batchSize)

    folder_name = "SGI-NET"

    iter_path = os.path.join(opt.checkpoints_dir, folder_name, opt.name,
                             'iter.txt')
    if opt.continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',',
                                                 dtype=int)
        except FileNotFoundError as e:
            start_epoch, epoch_iter = 1, 0
        print('Resuming from epoch %d at iteration %d' % (
        start_epoch, epoch_iter))
    else:
        start_epoch, epoch_iter = 1, 0
    train_loader_kwargs = {'transform': input_transform,
                           "gt_transform": gt_transform,
                           "inst_map_transform": inst_map_transform,
                           "root": opt.dataroot, "opt": opt}
    trainset = get_segmentation_dataset(opt.dataset, split='train', mode='train',
                                        **train_loader_kwargs)
    if len(opt.gpu_ids) > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
                trainset)
    else:
        train_sampler = None
    trainloader = data.DataLoader(trainset, batch_size=opt.batchSize,
                                  drop_last=True, shuffle=(train_sampler is None),
                                  num_workers=opt.nThreads, worker_init_fn=worker_init_fn, pin_memory=True, sampler=train_sampler)


    dataset_size = len(trainloader) * opt.batchSize
    visualizer = Visualizer(opt)
    total_steps = (start_epoch - 1) * dataset_size + epoch_iter
    display_delta = total_steps % opt.display_freq
    print_delta = total_steps % opt.print_freq
    save_delta = total_steps % opt.save_latest_freq

    if len(opt.gpu_ids) > 1:
        torch.cuda.set_device(opt.local_rank)
    model = SGINetModel(opt)

    if use_cuda:
        if len(opt.gpu_ids) > 1:
            n = torch.cuda.device_count() // opt.local_world_size
            device = list(
                range(opt.local_rank * n, (opt.local_rank + 1) * n))
            print(f"[{os.getpid()}] rank = {dist.get_rank()}, "
                  + f"world_size = {dist.get_world_size()},"
                    f" n = {n}, device_ids = {device}")
            model.cuda(device[0])
            if opt.fp16:
                from apex import amp
                model, [optimizer_G, optimizer_D, optimizer_G_instance,
                        optimizer_D_obj] = amp.initialize(model, [
                    model.optimizer_G, model.optimizer_D,
                    model.optimizer_G_instance, model.optimizer_D_obj],
                                                          opt_level='O1')
                model = DistributedDataParallel(model, device_ids=device,
                                                find_unused_parameters=True)
            else:
                model = DistributedDataParallel(model, device_ids=device,
                                                find_unused_parameters=True)
                optimizer_G, optimizer_D, optimizer_G_instance, optimizer_D_obj = model.module.optimizer_G, \
                                                                                  model.module.optimizer_D, \
                                                                                  model.module.optimizer_G_instance, \
                                                                                  model.module.optimizer_D_obj
        else:
            device =\
                [torch.device(f"cuda:{opt.gpu_ids[0]}" if use_cuda else "cpu")]
            if opt.fp16:
                from apex import amp
                if opt.use_bbox:
                    model, [optimizer_G, optimizer_D, optimizer_G_instance,
                            optimizer_D_obj] = amp.initialize(model, [
                        model.optimizer_G, model.optimizer_D,
                        model.optimizer_G_instance, model.optimizer_D_obj],
                                                              opt_level='O1')
                else:
                    model, [optimizer_G, optimizer_D] = amp.initialize(
                        model, [
                            model.optimizer_G, model.optimizer_D],
                        opt_level='O1')
                model = DataParallel(model, device_ids=opt.gpu_ids)
            else:
                model = DataParallel(model, device_ids=opt.gpu_ids)
                if opt.use_bbox:
                    optimizer_G, optimizer_D, optimizer_G_instance, optimizer_D_obj = model.module.optimizer_G, \
                                                                              model.module.optimizer_D, \
                                                                              model.module.optimizer_G_instance, \
                                                                              model.module.optimizer_D_obj
                else:
                    optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D

    else:
        optimizer_G, optimizer_D, optimizer_G_instance, optimizer_D_obj = model.optimizer_G, model.optimizer_D, model.optimizer_G_instance, model.optimizer_D_obj

    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size
        if len(opt.gpu_ids) > 1:
            train_sampler.set_epoch(epoch)
        for i, input_dict in enumerate(iter(trainloader)):

            gt_images = input_dict["gt_images"].to(device[0])
            gt_seg_maps = input_dict["gt_seg_maps"].type(torch.LongTensor).to(device[0])
            inst_maps = input_dict["inst_map"].to(device[0])
            inst_maps_valid_idx = input_dict["inst_map_valid_idx"]
            masks = input_dict["masks"].to(device[0])
            indexes = input_dict["indexes"]
            insta_maps_bbox = input_dict["insta_maps_bbox"].to(device[0])
            inst_map_compact = input_dict["inst_map_compact"].to(device[0])
            theta = input_dict["theta"].to(device[0])
            compute_instance = input_dict["compute_instance"]

            images_masked = (gt_images - gt_images * masks)
            if total_steps % opt.print_freq == print_delta:
                iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            save_fake = total_steps % opt.display_freq == display_delta

            one_hot_seg_map = torch.FloatTensor(opt.batchSize, opt.semantic_nc, gt_seg_maps.shape[2],
                                                   gt_seg_maps.shape[3]).zero_().to(device[0])
            one_hot_seg_map.scatter_(1, gt_seg_maps, 1)

            gt_seg_maps_masked = gt_seg_maps.clone()
            gt_seg_maps_masked[masks == 1.0] = opt.label_nc - 1 if opt.no_contain_dontcare_label else opt.label_nc
            one_hot_gt_seg_maps_masked = torch.FloatTensor(opt.batchSize, opt.semantic_nc,
                                                           gt_seg_maps.shape[
                                                               2],
                                                           gt_seg_maps.shape[
                                                               3]).zero_().to(
                device[0])
            one_hot_gt_seg_maps_masked.scatter_(1, gt_seg_maps_masked,
                                                    1.0)
            model_input = {
                "one_hot_gt_seg_maps_masked" : one_hot_gt_seg_maps_masked,
                "one_hot_seg_map" : one_hot_seg_map,
                "gt_images_masked" : images_masked,
                "gt_images" : gt_images,
                "masks" : masks,
                "insta_maps_bbox" : insta_maps_bbox,
                "inst_maps" : inst_maps,
                "inst_maps_compact" : inst_map_compact,
                "theta_transform" : theta,
                "indexes" : indexes,
                "compute_loss_g" : True,
                "inst_maps_valid_idx": inst_maps_valid_idx,
                "compute_instance": compute_instance}
            losses, generated_images, generated_instances_pad, generated_instances, generated_seg_maps, offset_flow, one_hot_gt_seg_maps_masked_inst = \
                model.forward(model_input)
            output_images = generated_images.detach()
            output_seg_maps = generated_seg_maps.detach()
            one_hot_gt_seg_maps_masked_inst = one_hot_gt_seg_maps_masked_inst.detach()

            if generated_instances[0] is not None:
                generated_instances_pad = generated_instances_pad.detach()
                inst_map = inst_maps
            else:
                inst_map = [None]
            blended_images = (output_images * masks + gt_images * (1 - masks)).cpu()
            # sum per device losses
            losses = [
                torch.mean(x) if not isinstance(x,
                                                int) and x is not None else x
                for x in losses]
            if use_cuda:
                loss_dict = dict(zip(model.module.loss_names, losses))
            else:
                loss_dict = dict(zip(model.loss_names, losses))
            # calculate final loss scalar
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_D_obj = (loss_dict.get("D_fake_obj", 0) + \
                         loss_dict.get("D_real_obj", 0)) * 0.5

            ############### Backward Pass ####################
            # update generator weights
            # Update G
            loss_G = loss_dict['G_GAN'] + \
                         loss_dict.get('G_GAN_Feat', 0) + \
                         loss_dict.get('G_perceptual_' +
                                       opt.which_perceptual_loss, 0) + \
                         loss_dict.get("G_Image_Rec", 0) + \
                         loss_dict.get("G_seg_map_rec", 0) + \
                         loss_dict.get("G_Style", 0) + \
                         loss_dict.get("G_TV_loss", 0)
            loss_instance = loss_dict.get('G_KL_inst', 0) + \
                         loss_dict.get('G_Inst_rec', 0) +\
                                loss_dict.get("G_GAN_obj", 0)
            optimizer_G.zero_grad()
            if opt.use_bbox:
                optimizer_G_instance.zero_grad()
            if opt.fp16:
                if loss_instance != 0:
                    loss_instance.backward()
                    optimizer_G_instance.step()
                    optimizer_G_instance.zero_grad()
                with amp.scale_loss(loss_G, optimizer_G) as scaled_loss:
                    scaled_loss.backward()
                optimizer_G.step()
                optimizer_G.zero_grad()
            else:
                loss_G.backward()
                optimizer_G.step()
                optimizer_G.zero_grad()
                loss_instance.backward()
                optimizer_G_instance.step()
                optimizer_G_instance.zero_grad()

            # update discriminator weights
            if opt.use_bbox:
                optimizer_D_obj.zero_grad()
            optimizer_D.zero_grad()
            if opt.fp16:
                if loss_D_obj != 0:
                    loss_D_obj.backward()
                    optimizer_D_obj.step()
                    optimizer_D_obj.zero_grad()
                with amp.scale_loss(loss_D, optimizer_D) as scaled_loss:
                    scaled_loss.backward()
                optimizer_D.step()
            else:
                if loss_D_obj != 0:
                    loss_D_obj.backward()
                    optimizer_D_obj.step()
                loss_D.backward()
                optimizer_D.step()

            optimizer_D.zero_grad()

            ############## Display results and errors ##########
            ### print out errors

            if total_steps % opt.print_freq == print_delta:
                if not len(opt.gpu_ids) > 1:
                    errors = {k: v.data.item() if not isinstance(v,
                                                                 int) and v is not None else v
                              for k, v
                              in loss_dict.items()}
                else:
                    errors = {k: util.reduce_tensor(v.data if torch.is_tensor(v) else torch.Tensor([v]).to(device[0]).data, dist.get_world_size()) for k, v
                              in loss_dict.items()}
                if opt.local_rank == 0:
                    t = (time.time() - iter_start_time) / opt.print_freq
                    visualizer.print_current_errors(epoch, epoch_iter, errors,
                                                    t)
                    visualizer.plot_current_errors(errors, total_steps)
                # call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])

            ### display output images
            if len(opt.gpu_ids) > 1 and opt.local_rank != 0:
                save_fake = False
            if save_fake:
                visuals = OrderedDict([('gt_seg_map',
                                        util.tensor2label(gt_seg_maps[0],
                                                          opt.semantic_nc)),
                                       ('gt_seg_map_masked',
                                        util.tensor2label(one_hot_gt_seg_maps_masked_inst[0],
                                                          opt.semantic_nc)),
                                       ('real_image',
                                        util.tensor2im(gt_images[0])),
                                       ('inst_map',
                                        util.tensor2im(inst_map[0], normalize=False)),
                                       ('masked_image',
                                        util.tensor2im(images_masked[0])),
                                       ("expected_flow",
                                        util.tensor2im(offset_flow[0])),
                                       ('fake_image',
                                        util.tensor2im(output_images[0])),
                                       ("fake_instance_pad",
                                        util.tensor2label(
                                            generated_instances_pad[0],
                                            opt.semantic_nc)),
                                       ('fake_seg_maps',
                                        util.tensor2label(output_seg_maps[0],
                                                          opt.semantic_nc)),
                                       ('reconstructed_image',
                                        util.tensor2im(blended_images[0]))
                                       ])
                visualizer.display_current_results(visuals, epoch,
                                                   total_steps)
            ### save latest model
            if len(opt.gpu_ids) > 1 and opt.local_rank != 0:
                save_model = False
            else:
                save_model = True
            if (total_steps % opt.save_latest_freq == save_delta) and save_model:

                print(
                    'saving the latest model (epoch %d, total_steps %d)' % (
                        epoch, total_steps))
                if use_cuda:
                    model.module.save('latest')
                else:
                    model.save('latest')
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',',
                           fmt='%d')
            if epoch_iter >= dataset_size:
                break
            # end of epoch
        if opt.local_rank == 0:
            iter_end_time = time.time()
            print('End of epoch %d / %d \t Time Taken: %d sec' %
                  (epoch, opt.niter + opt.niter_decay,
                   time.time() - epoch_start_time))

        ### save model for this epoch
        if len(opt.gpu_ids) > 1 and opt.local_rank != 0:
            save_model = False
        else:
            save_model = True
        if (epoch % opt.save_epoch_freq == 0) and save_model:
            print('saving the model at the end of epoch %d, iters %d' % (
                epoch, total_steps))
            if use_cuda:
                model.module.save('latest')
                model.module.save(epoch)
            else:
                model.save('latest')
                model.save(epoch)
            np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

        ### linearly decay learning rate after certain iterations
        if epoch > opt.niter:
            if use_cuda:
                model.module.update_learning_rate()
            else:
                model.update_learning_rate()

def test_slg(opt):
    folder_name = "SGI-NET"
    import glob
    epochs = [f"epoch_{x.split('/')[-1].split('_')[0]}" for x in glob.glob(os.path.join(opt.checkpoints_dir, folder_name, opt.name) + "/latest*SGI*_G.pth")]
    test_z_appr = util.load_latent_vector_appr(opt)
    for epoch in epochs:
        old_epoch_folder = epoch.split("_")[1]
        opt.which_epoch = epoch
        web_dir = os.path.join(opt.checkpoints_dir, folder_name, opt.name, "test", opt.which_epoch, "results")
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (
        opt.name, opt.phase, opt.which_epoch))

        visualizer = Visualizer(opt)
        opt.which_epoch = old_epoch_folder
        model = SGINetInferenceModel(opt)
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
        count = 0
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
                    "test_z_appr": test_z_appr.to(device[0])[i].unsqueeze(0),
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
                completed_images = output_images * masks + gt_images * (1 - masks)
                completed_images_no_cond = output_images_no_cond * masks + gt_images * (1 - masks)
                for j in range(len(gt_images)):
                    real_images.append(gt_images[j])
                    fake_images.append(completed_images[j])
                    fake_images_no_cond.append(completed_images_no_cond[j])
                    visuals = OrderedDict([('gt_seg_map',
                                            util.tensor2label(gt_seg_maps[0],
                                                              opt.label_nc)),
                                           ('gt_seg_map_masked',
                                            util.tensor2label(
                                                one_hot_gt_seg_maps_masked_inst[0],
                                                opt.semantic_nc)),
                                           ('gt_seg_map_masked_no_cond',
                                            util.tensor2label(
                                                one_hot_gt_seg_maps_masked_inst_no_cond[0],
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
                                           ('fake_image_no_cond',
                                            util.tensor2im(output_images_no_cond[0])),
                                           ("fake_instance",
                                            util.tensor2label(
                                                output_gen_instances[0],
                                                opt.semantic_nc)),
                                           ('fake_seg_maps',
                                            util.tensor2label(output_seg_maps[0],
                                                              opt.semantic_nc)),
                                           ('reconstructed_image',
                                            util.tensor2im(completed_images[0])),
                                           ('fake_seg_maps_no_cond',
                                            util.tensor2label(
                                                output_seg_maps_no_cond[0],
                                                opt.semantic_nc)),
                                           ('reconstructed_image_no_cond',
                                            util.tensor2im(
                                                completed_images_no_cond[0]))
                                           ])
                    visualizer.save_images(webpage, visuals, [f"{images_path[j][:-4]}_{crop_type}{images_path[j][-4:]}"])

            webpage.save()
        print(len(real_images))
        for i in range(len(real_images)):
            real = real_images[i].numpy()
            fake = fake_images[i].cpu().numpy()
            fake_no_cond = fake_images_no_cond[i].cpu().numpy()
            t = real - fake
            t1 = real - fake_no_cond
            l2 += np.square(t).sum((0,1,2), keepdims=True)
            l1 += np.abs(t).sum((0,1,2), keepdims=True)
            real = (real + 1) * 127.5
            fake = (fake + 1) * 127.5
            fake_no_cond = (fake_no_cond + 1) * 127.5
            ssim_score += ssim(torch.from_numpy(real).unsqueeze(0),
                               torch.from_numpy(fake).unsqueeze(0),
                               data_range=255, size_average=True)
            ssim_score_no_cond += ssim(torch.from_numpy(real).unsqueeze(0),
                                       torch.from_numpy(fake_no_cond).unsqueeze(0),
                                       data_range=255, size_average=True)
            psnr_score += compare_psnr(fake.transpose(1, 2, 0) / 255,
                                 real.transpose(1, 2, 0) / 255, data_range=1)
            psnr_score_no_cond += compare_psnr(fake_no_cond.transpose(1, 2, 0) / 255,
                                         real.transpose(1, 2, 0) / 255,
                                         data_range=1)
        real_images = torch.stack([((real_image + 1) * 0.5).permute(1,2,0) for real_image in real_images], dim=0).numpy()
        fake_images = torch.stack([((fake_image + 1) * 0.5).permute(1,2,0) for fake_image in fake_images], dim=0).numpy()
        fake_images_no_cond = torch.stack([((fake_image_no_cond + 1) * 0.5).permute(1,2,0) for fake_image_no_cond in fake_images_no_cond], dim=0).numpy()
        fid_cond = calculate_fid(real_images, fake_images, True, 8)
        fid_no_cond = calculate_fid(real_images, fake_images_no_cond, True, 8)
        final_psnr = psnr_score / count
        final_psnr_no_cond = psnr_score_no_cond / count
        final_ssim = ssim_score / count
        final_ssim_no_cond = ssim_score_no_cond / count
        print(f"epoch: {epoch} count: {count}")
        results = {"l1": l1/count,
                   "l2": l2/count,
                   "psnr": final_psnr,
                   "ssim": final_ssim,
                   "psnr_no_cond": final_psnr_no_cond,
                   "ssim_no_cond": final_ssim_no_cond,
                   "fid_cond" : fid_cond,
                   "fid_no_cond": fid_no_cond
                   }
        visualizer.print_results(results)



if __name__ == "__main__":
    opt = TrainOptions().parse()
    if opt.phase == "train":
        if len(opt.gpu_ids) > 1:
            env_dict = {
                key: os.environ[key]
                for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
            }
            print(
                f"[{os.getpid()}] Initializing process group with: {env_dict}")
            dist.init_process_group(backend="nccl")
            print(
                f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
                + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
            )
            train(None, opt)
            dist.destroy_process_group()
        else:
            train(None, opt)
    else:

        opt.no_flip = True
        opt.isTrain = False
        test_slg(opt)
