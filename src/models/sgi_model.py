import numpy as np
import torch
import os
from torch.autograd import Variable
from torch import autograd
from .base_model import BaseModel
from .generators import sgi_net
from .functions.base_function import STN_fixTheta
from .discriminators.multiscale_discriminator import MultiscaleDiscriminator
from .discriminators.discriminator_obj import DiscriminatorObj
import functools
from torch import nn
from losses import losses
from torch.nn import functional as F
from utils import util


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_activation_layer(activation_type="elu"):
    if activation_type == 'elu':
        activation_layer = functools.partial(nn.ELU, inplace=True)
    elif activation_type == 'relu':
        activation_layer = functools.partial(nn.ReLU, inplace=True)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % activation_type)
    return activation_layer


class SGINetModel(BaseModel):
    def name(self):
        return 'SGINetModel'

    def init_loss_filter(self, use_gan_feat_loss, use_perceptual_loss, use_style_loss, gan_mode):
        flags = (True, True, True, use_gan_feat_loss, True, True, use_perceptual_loss, use_style_loss, True, True, True, True, True)
        if gan_mode == "wgangp":
            def loss_filter(g_gan, g_gan_feat, g_image_rec, g_seg_map_rec, g_perceptual, g_style, d_loss,
                            wgan_gp, d_real_obj, d_fake_obj, g_gan_obj):
                return [l for (l, f) in
                        zip((g_gan, g_gan_feat, g_image_rec, g_seg_map_rec, g_perceptual, g_style, d_loss,
                             wgan_gp, d_real_obj, d_fake_obj, g_gan_obj), flags) if f]
        else:
            def loss_filter(g_gan, g_kl_inst, g_inst_rec, g_gan_feat, g_image_rec, g_seg_map_rec, g_perceptual, g_style, d_real, d_fake, d_real_obj, d_fake_obj, g_gan_obj):
                return [l for (l, f) in
                        zip((g_gan, g_kl_inst, g_inst_rec, g_gan_feat, g_image_rec, g_seg_map_rec, g_perceptual, g_style, d_real, d_fake, d_real_obj, d_fake_obj, g_gan_obj), flags) if f]
        return loss_filter

    def __init__(self, opt):
        super(SGINetModel, self).__init__(opt)
        if opt.isTrain:  # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.gen_features = False
        self.gan_mode = opt.gan_mode
        self.save_dir = os.path.join(opt.checkpoints_dir, "SGI-NET", opt.name)
        ##### define networks
        norm_layer = get_norm_layer(norm_type=opt.norm)
        activation_layer = get_activation_layer(activation_type=opt.activation)
        # Generator network
        netG_input_nc = opt.semantic_nc
        self.netG = sgi_net.SGINet(netG_input_nc, opt.input_nc, norm_layer,
                                   not opt.no_deconv, opt.use_attention,
                                   not opt.no_dilated_conv, opt.use_sn_generator,
                                   opt.ngf, opt.n_downsample_global, opt.use_skip,
                                   activation_layer, opt.use_spade,
                                   opt.which_encoder, opt.use_pixel_shuffle, opt.use_bbox, opt.use_multi_scale_loss, self.gpu_ids)
        print(self.netG)
        crop = True if self.opt.size_crop_height is not None and self.opt.size_crop_width is not None else False
        if not crop:
            self.image_width = self.opt.image_width
            self.image_height = self.opt.image_height
        else:
            self.image_width = self.opt.size_crop_width
            self.image_height = self.opt.size_crop_height
        if self.opt.use_bbox:
            self.netG_shape_encoder = sgi_net.Shape_Encoder(32, opt.z_len, opt.compact_sizex * opt.compact_sizey, len(self.opt.classes_of_interest_ids), 6)
            self.netG_shape_decoder = sgi_net.Shape_Decoder(opt.z_len, len(self.opt.classes_of_interest_ids), 6)
            self.stn_fix = STN_fixTheta(self.opt)
        if len(self.gpu_ids) > 0:
            assert (torch.cuda.is_available())
            self.netG.cuda("cuda")
            if self.opt.use_bbox:
                self.netG_shape_decoder.cuda("cuda")
                self.netG_shape_encoder.cuda("cuda")
                self.stn_fix.cuda("cuda")
        self.netG.apply(weights_init)
        if self.opt.use_bbox:
            self.netG_shape_decoder.apply(weights_init)
            self.netG_shape_encoder.apply(weights_init)
            self.pad = Variable(torch.zeros(self.opt.semantic_nc, self.image_height, self.image_width)).cuda()
            if self.opt.label_nc == 21:
                self.ch_to_inst_id = {"4": "4", "5": "5", "6": "6", "7": "7",
                                      "8": "8", "9": "9", "10": "10",
                                      "11": "11", "12": "12", "13": "13",
                                      "14": "14"}
            elif self.opt.label_nc == 17:
                self.ch_to_inst_id = {"8": "24", "9": "25", "10": "26",
                                      "11": "27", "12": "28", "13": "31",
                                      "14": "32", "15": "33"}
            self.ch_to_index = {ch: i for i, ch in enumerate(self.opt.classes_of_interest_ids)}
            self.index_to_ch = {v: k for k, v in self.ch_to_index.items()}
            self.inst_id_to_ch = {v: k for k, v in self.ch_to_inst_id.items()}
            self.one_hot_instance =\
                torch.FloatTensor(self.opt.semantic_nc, self.image_height, self.image_width).zero_().to("cuda")
            self.bb_white = Variable(
                self.Tensor(self.opt.batchSize, 1, self.opt.compact_sizey,
                            self.opt.compact_sizex).fill_(1.))
            self.rotFix = Variable(
                torch.from_numpy(np.array([1., 0., 1., 0., 1., 1.])).float()).cuda()

        # Discriminator network
        if self.isTrain:
            use_sigmoid = True if opt.gan_mode == "vanilla" else False
            netD_input_nc = opt.input_nc + opt.semantic_nc

            self.netD_global = \
                MultiscaleDiscriminator(netD_input_nc,
                                        ndf=64,
                                        n_layers=opt.n_layers_D,
                                        norm_layer=norm_layer,
                                        use_sigmoid=use_sigmoid,
                                        num_D=opt.num_D_global,
                                        getIntermFeat=not opt.no_ganFeat_loss,
                                        use_sn_discriminator= opt.use_sn_discriminator)
            if self.opt.use_bbox:
                self.netD_obj = DiscriminatorObj(ndf=64)
            if len(self.gpu_ids) > 0:
                assert (torch.cuda.is_available())
                self.netD_global.cuda("cuda")
                if self.opt.use_bbox:
                    self.netD_obj.cuda("cuda")
            self.netD_global.apply(weights_init)
            if self.opt.use_bbox:
                self.netD_obj.apply(weights_init)
        if self.opt.verbose:
            print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain_sg:
            pretrained_path = opt.test_model_sg if not self.isTrain else opt.load_pretrain_sg

            self.load_network(self.netG, 'SGI_NET_G', opt.which_epoch,
                              pretrained_path)
            if self.opt.use_bbox:
                self.load_network(self.netG_shape_encoder, "SGI_NET_INST_ENC",opt.which_epoch,
                              pretrained_path)
                self.load_network(self.netG_shape_decoder, "SGI_NET_INST_DEC",opt.which_epoch,
                              pretrained_path)
            if self.isTrain:
                self.load_network(self.netD_global,
                                  'SGI_NET_D_GLOBAL',
                                  opt.which_epoch,
                                  pretrained_path)
                if self.opt.use_bbox:
                    self.load_network(self.netD_obj,
                                  'SGI_NET_D_OBJ',
                                  opt.which_epoch,
                                  pretrained_path)
        # set loss functions and optimizers
        if self.isTrain:
            self.old_lr = opt.lr
            self.use_perceptual_loss = True if opt.which_perceptual_loss in [
                "alex", "vgg"] else False
            # define loss functions
            self.loss_filter = self.init_loss_filter(
                not opt.no_ganFeat_loss,
                self.use_perceptual_loss, not self.opt.no_ganStyle_loss,
                not self.opt.no_ganTV_loss,
                self.opt.gan_mode)
            if len(self.gpu_ids) > 0:
                self.criterionGAN = losses.GANLoss(gan_mode=self.gan_mode,
                                                   tensor=self.Tensor,
                                                   device="cuda")
                self.criterionGAN_obj = losses.GANLoss(gan_mode="lsgan",
                                                   tensor=self.Tensor,
                                                   device="cuda")

            else:
                self.criterionGAN = losses.GANLoss(gan_mode=self.gan_mode,
                                                   tensor=self.Tensor,
                                                   device="cpu")
                self.criterionGAN_obj = losses.GANLoss(gan_mode="lsgan",
                                                   tensor=self.Tensor,
                                                   device="cpu")

            self.criterionRec = torch.nn.L1Loss(reduction="sum")
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionMSE = torch.nn.MSELoss()
            self.criterionCE = torch.nn.NLLLoss2d()
            self.criterionInst = torch.nn.L1Loss(reduction="none")
            if self.use_perceptual_loss or not self.opt.no_ganStyle_loss:
                if opt.which_perceptual_loss == "vgg":
                    self.criterion_perceptual_style = \
                        losses.Style_Perceptual_Loss(gpu_ids=self.gpu_ids, fp16=self.opt.fp16)


            # Names so we can breakout loss
            self.loss_names = \
                    self.loss_filter('G_GAN', "G_KL_inst", "G_Inst_rec", 'G_GAN_Feat', "G_Image_Rec",
                                     "G_seg_map_rec",
                                     'G_perceptual_' + opt.which_perceptual_loss,
                                     "G_Style", 'D_real', 'D_fake', 'D_real_obj', 'D_fake_obj', "G_GAN_obj")

            # initialize optimizers
            # optimizer G
            if self.opt.use_bbox:
                params_instance = list(self.netG_shape_encoder.parameters()) + \
                                  list(self.netG_shape_decoder.parameters())
                self.optimizer_G_instance = torch.optim.Adam(params_instance, lr=opt.lr,
                                                             betas=(opt.beta1, opt.beta2))
            params = list(self.netG.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr,
                                                betas=(opt.beta1, opt.beta2))
            # optimizer D
            params = list(self.netD_global.parameters())

            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr,
                                                betas=(opt.beta1, opt.beta2))
            if self.opt.use_bbox:
                params = list(self.netD_obj.parameters())
                self.optimizer_D_obj = torch.optim.Adam(params, lr=opt.lr,
                                                betas=(opt.beta1, opt.beta2))

    def pad_to_nClass(self, x, masks, compute, inst_maps_valid_idx):
        seg_maps = []
        for i in range(x.shape[0]):
            seg_map = x[i].clone().type(torch.LongTensor).to("cuda")
            comp_instance = compute[i]
            mask = masks[i]
            idx = int(inst_maps_valid_idx[i].item())
            if 1 in comp_instance:
                ch = int(self.inst_id_to_ch[str(idx)[:-3]])
                seg_map[seg_map==1] = ch
                seg_map[seg_map==0] = self.opt.label_nc - 1 if self.opt.no_contain_dontcare_label else self.opt.label_nc
                self.one_hot_instance.zero_()
                one_hot_instance = self.one_hot_instance.scatter(0, seg_map, 1.0)
                seg_maps.append(one_hot_instance)
            else:
                seg_map[seg_map == 0] = self.opt.label_nc - 1 if self.opt.no_contain_dontcare_label else self.opt.label_nc
                self.one_hot_instance.zero_()
                one_hot_instance = self.one_hot_instance.scatter(0, seg_map, 1.0)
                seg_maps.append(one_hot_instance)
        return torch.stack(seg_maps, 0)


    def ch_to_one_hot(self, inst_maps_valid_idx, comp_instances):
        vae_cond_decoder = []
        self.ch = torch.ones(1, 1, 1).cuda()
        for i in range(inst_maps_valid_idx.shape[0]):
            idx = int(inst_maps_valid_idx[i].item())
            comp_instance = comp_instances[i]
            if 1 in comp_instance:
                ch = int(self.inst_id_to_ch[str(idx)[:-3]])
                index = torch.LongTensor([self.ch_to_index[ch]]).cuda()
                index_one_hot = torch.FloatTensor(len(self.opt.classes_of_interest_ids)).zero_().cuda()
                index_one_hot.scatter(0, index, 1)
                vae_cond_decoder.append(index_one_hot.unsqueeze(1).unsqueeze(2))
            else:
                pad_ch = Variable(
                    torch.zeros(len(self.opt.classes_of_interest_ids), 1, 1)).cuda()
                vae_cond_decoder.append(pad_ch)
        return torch.stack(vae_cond_decoder, 0)



    def discriminate(self, netD, gt_seg_maps, fake_seg_maps, fake_image, real_image, mask):
        input_concat_fake = \
            torch.cat((fake_image.detach(), fake_seg_maps.detach()), dim=1)
        input_concat_real = \
            torch.cat((real_image, gt_seg_maps),
                      dim=1)
        return netD.forward(input_concat_fake), \
               netD.forward(input_concat_real)

    def reparameterize(self, mu, logvar, mode):
        from apex import amp
        mu = mu.float()
        logvar = logvar.float() if logvar is not None else None
        if self.opt.fp16:
            with amp.disable_casts():
                if mode == 'train':
                    std = torch.exp(0.5*logvar)
                    eps = Variable(std.data.new(std.size()).normal_())
                    return mu + eps*std
                else:
                    return mu
        else:
            if mode == 'train':
                std = torch.exp(0.5 * logvar)
                eps = Variable(std.data.new(std.size()).normal_())
                return mu + eps * std
            else:
                return mu

    def forward(self, model_input):
        # Encode Inputs
        # input_concat = torch.cat((image_masked, predicted_label), dim=1)
        loss_D_real_obj = loss_D_fake_obj = 0
        if self.opt.use_bbox:
            self.class_conditioning = self.ch_to_one_hot(
                model_input["inst_maps_valid_idx"],
                model_input["compute_instance"])
            self.mu, self.logvar =\
                self.netG_shape_encoder(model_input["inst_maps_compact"], self.class_conditioning, model_input["theta_transform"])
            self.z_rep = self.reparameterize(self.mu, self.logvar, "train")
            self.instance = self.netG_shape_decoder(self.z_rep,
                                                    self.class_conditioning, model_input["theta_transform"]) * model_input["compute_instance"].unsqueeze(2).unsqueeze(2)
            self.instance_transformed = self.stn_fix(self.instance, model_input["theta_transform"], self.image_width, self.image_height)
            disc_input_fake = self.instance
            disc_input_real = model_input["inst_maps_compact"]
            obj_pred_fake = self.netD_obj.forward(disc_input_fake.detach())
            obj_pred_real = self.netD_obj.forward(disc_input_real)
            # update D obj

            obj_pred_fake = obj_pred_fake * model_input["compute_instance"]
            obj_pred_real = obj_pred_real * model_input["compute_instance"] + (1 - model_input["compute_instance"])
            loss_D_fake_obj = self.criterionGAN.forward(obj_pred_fake, False)
            loss_D_real_obj = self.criterionGAN.forward(obj_pred_real, True)
            self.instance_masked = (self.instance_transformed.detach() * model_input["insta_maps_bbox"]).float()
            self.instance_pad = self.pad_to_nClass((self.instance_masked > 0.5).float(), model_input["masks"], model_input['compute_instance'], model_input["inst_maps_valid_idx"])
            disc_input_fake = self.instance
            obj_pred_fake = self.netD_obj.forward(disc_input_fake)
            obj_pred_fake = obj_pred_fake * model_input["compute_instance"] + (1 - model_input["compute_instance"])
            loss_G_fake_obj = self.criterionGAN.forward(obj_pred_fake, True)
            loss_G_GAN_obj = loss_G_fake_obj
            from apex import amp
            self.mu = self.mu.float()[
                model_input["compute_instance"].bool().squeeze(1)]
            self.logvar = self.logvar.float()[
                model_input["compute_instance"].bool().squeeze(1)]
            if self.opt.fp16:
                with amp.disable_casts():
                    denominator = max((torch.sum(
                        model_input["compute_instance"]) * self.opt.z_len), 1)
                    instance_KL_loss = (- 0.5 * (torch.sum(
                        1 + self.logvar - self.mu.pow(
                            2) - self.logvar.exp()) / denominator)) * self.opt.lambda_KL_rec
                    fake_inst = self.instance[
                        model_input["compute_instance"].bool().squeeze(1)]
                    real_inst = model_input["inst_maps_compact"][
                        model_input["compute_instance"].bool().squeeze(1)]
                    denominator = max(
                        torch.prod(torch.FloatTensor(list(real_inst.shape))), 1)
                    instance_rec_loss = torch.sum(
                        torch.abs(fake_inst - real_inst)) / denominator
                    instance_rec_loss = instance_rec_loss * self.opt.lambda_inst_rec
            else:
                denominator = max((torch.sum(model_input["compute_instance"]) * self.opt.z_len), 1)
                instance_KL_loss = (- 0.5 * (torch.sum(
                    1 + self.logvar - self.mu.pow(2) - self.logvar.exp()) / denominator)) * self.opt.lambda_KL_rec
                fake_inst = self.instance[model_input["compute_instance"].squeeze().nonzero()]
                real_inst = model_input["inst_maps_compact"][model_input["compute_instance"].squeeze().nonzero()]
                denominator = max(torch.prod(torch.FloatTensor(list(real_inst.shape))), 1)
                instance_rec_loss = torch.sum(torch.abs(fake_inst - real_inst)) / denominator
                instance_rec_loss = instance_rec_loss * self.opt.lambda_inst_rec
        else:
            self.instance_masked = None
            instance_KL_loss = 0
            instance_rec_loss = 0
            loss_G_GAN_obj = 0
            self.instance_pad = self.instance_masked = [None]
        multi_scale_fake_seg_maps, fake_images, offset_flow = self.netG.forward(model_input["one_hot_gt_seg_maps_masked"],
                                                                                model_input["gt_images_masked"],
                                                                                model_input["masks"],
                                                                                self.instance_masked)
        # Fake Detection and Loss

        global_pred_fake, global_pred_real = self.discriminate(
            self.netD_global,
            model_input["one_hot_seg_map"],
            model_input["one_hot_seg_map"],
            fake_images,
            model_input["gt_images"], model_input["masks"])
        loss_D_fake_global = self.criterionGAN.forward(global_pred_fake, False)

        # Real Loss

        loss_D_real_global = self.criterionGAN(global_pred_real, True)

        loss_D_penalty = 0
        loss_D = 0
        loss_G_GAN = 0
        loss_D_fake = loss_D_fake_global
        loss_D_real = loss_D_real_global

        # GAN feature matching loss
        ## G part
        loss_G_GAN_Feat = loss_image_rec = loss_seg_map_rec = loss_G_perceptual = loss_G_style = 0
        # GAN loss (Fake Passability Loss)
        fake_cond = torch.cat((fake_images, model_input["one_hot_seg_map"]), dim=1)
        global_pred_fake = self.netD_global.forward(fake_cond)
        loss_G_GAN_global = self.criterionGAN(global_pred_fake, True)
        loss_G_GAN = loss_G_GAN_global

        if not self.opt.no_ganFeat_loss:
            numD = len(global_pred_fake)
            for i in range(self.opt.num_D_global):
                for j in range(len(global_pred_real[i]) - 1):
                    fake = global_pred_fake[i][j]
                    real = global_pred_real[i][j].detach()
                    criterionFeat = self.criterionFeat(fake, real)
                    loss_G_GAN_Feat += criterionFeat * self.opt.lambda_feat / numD
                # VGG feature matching loss
        ### real image

        rec_hole = self.criterionRec(fake_images * model_input["masks"], model_input["gt_images"] * model_input["masks"]) / (model_input["masks"].sum() * 3)
        rec_no_hole = self.criterionRec(fake_images * (1- model_input["masks"]), model_input["gt_images"] * (1 - model_input["masks"])) / ((1 - model_input["masks"]).sum() * 3)
        loss_image_rec = (rec_hole + rec_no_hole) * self.opt.lambda_rec
        loss_seg_map_rec = 0
        for i, scale_fake_seg_maps in enumerate(multi_scale_fake_seg_maps):
            if i < len(multi_scale_fake_seg_maps) - 1:
                scale_fake_seg_maps = torch.nn.functional.interpolate(scale_fake_seg_maps, size=(fake_images.shape[2:]), mode="nearest")
            loss_seg_map_rec +=\
                self.criterionCE(scale_fake_seg_maps,
                                 torch.argmax(model_input["one_hot_seg_map"].long(),
                                              dim=1)) * self.opt.lambda_seg_map * 1 / len(multi_scale_fake_seg_maps)
        loss_G_style_out, loss_G_perceptual_out =\
            self.criterion_perceptual_style(fake_images,
                                            model_input["gt_images"],
                                            self.use_perceptual_loss,
                                            not self.opt.no_ganStyle_loss)
        loss_G_perceptual = loss_G_perceptual_out
        loss_G_style = loss_G_style_out
        loss_G_perceptual *= self.opt.lambda_perceptual
        loss_G_style *= self.opt.lambda_style

        # Only return the fake_B image if necessary to save BW
        return [self.loss_filter(loss_G_GAN, instance_KL_loss, instance_rec_loss, loss_G_GAN_Feat, loss_image_rec,
                                 loss_seg_map_rec, loss_G_perceptual,
                                 loss_G_style, loss_D_real, loss_D_fake,
                                 loss_D_real_obj, loss_D_fake_obj, loss_G_GAN_obj),
                fake_images, self.instance_pad, self.instance_masked, multi_scale_fake_seg_maps[-1], offset_flow, model_input["one_hot_gt_seg_maps_masked"]]


    def inference(self, model_input):
        with torch.no_grad():
            if torch.any(model_input['inst_maps_valid_idx'].unique() != -1).item():
                if model_input["use_gt_instance_encoder"]:
                    self.class_conditioning = self.ch_to_one_hot(
                        model_input["inst_maps_valid_idx"],
                        model_input["compute_instance"])
                    self.mu, self.logvar = \
                        self.netG_shape_encoder(
                            model_input["inst_maps_compact"],
                            self.class_conditioning, model_input["theta_transform"])
                    self.z_rep = self.reparameterize(self.mu, self.logvar,
                                                     "train")
                else:
                    test_z_appr = model_input["test_z_appr"]
                    self.z_rep = self.reparameterize(test_z_appr, None, "test")
                self.class_conditioning = self.ch_to_one_hot(
                    model_input["inst_maps_valid_idx"],
                    model_input["compute_instance"])
                self.instance = self.netG_shape_decoder(self.z_rep,
                                                        self.class_conditioning, model_input["theta_transform"]) * \
                                model_input["compute_instance"].unsqueeze(
                                    2).unsqueeze(2)
                self.instance_transformed = self.stn_fix(self.instance,
                                                         model_input[
                                                             "theta_transform"],
                                                         self.image_width,
                                                         self.image_height)
                self.instance_masked = (
                            self.instance_transformed.detach() * model_input[
                        "insta_maps_bbox"]).float()
                self.instance_pad = self.pad_to_nClass((self.instance_masked > 0.5).float(),
                                                       model_input["masks"],
                                                       model_input['compute_instance'],
                                                       model_input["inst_maps_valid_idx"])
                one_hot_gt_seg_maps_masked_pre = model_input["one_hot_gt_seg_maps_masked"]
                input_concat_no_cond = True
            else:
                input_concat_no_cond = False
                self.instance_pad = self.pad_to_nClass(model_input["inst_maps"],
                                                       model_input["masks"],
                                                       model_input[
                                                           'compute_instance'],
                                                       model_input[
                                                           "inst_maps_valid_idx"])
                self.instance_masked = model_input["inst_maps"]
            multi_scale_fake_seg_maps, fake_images, offset_flow = self.netG.forward(model_input["one_hot_gt_seg_maps_masked"],
                     model_input["gt_images_masked"],
                                      model_input["masks"], self.instance_masked)
            if input_concat_no_cond:
                fake_seg_maps_no, fake_images_no, offset_flow_no = self.netG.forward(model_input["one_hot_gt_seg_maps_masked"],
                     model_input["gt_images_masked"],
                     model_input["masks"],
                     self.instance_masked * 0)
            else:
                fake_seg_maps_no, fake_images_no, offset_flow_no, one_hot_gt_seg_maps_masked_pre = multi_scale_fake_seg_maps, fake_images, offset_flow, model_input["one_hot_gt_seg_maps_masked"]
            return fake_images, self.instance_pad, multi_scale_fake_seg_maps[-1], offset_flow, model_input["one_hot_gt_seg_maps_masked"], \
                   [fake_images_no, fake_seg_maps_no[-1], offset_flow_no, one_hot_gt_seg_maps_masked_pre]

    def save(self, which_epoch):
        self.save_network(self.netG, 'SLG_NET_G', which_epoch, self.gpu_ids)
        if self.opt.use_bbox:
            self.save_network(self.netG_shape_encoder, "SLG_NET_INST_ENC", which_epoch, self.gpu_ids)
            self.save_network(self.netG_shape_decoder, "SLG_NET_INST_DEC",
                              which_epoch, self.gpu_ids)
            self.save_network(self.netD_obj, 'SLG_NET_D_OBJ', which_epoch,
                              self.gpu_ids)
        self.save_network(self.netD_global, 'SLG_NET_D_GLOBAL', which_epoch,
                          self.gpu_ids)
        if self.gen_features:
            self.save_network(self.netE, 'SLG_NET_E', which_epoch,
                              self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr,
                                            betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print(
                '------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr


class SGINetInferenceModel(SGINetModel):
    def __init__(self, opt):
        super(SGINetInferenceModel, self).__init__(opt)

    def forward(self, model_input):
        return self.inference(model_input)
