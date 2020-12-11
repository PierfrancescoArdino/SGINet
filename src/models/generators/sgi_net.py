from torch import nn
from collections import namedtuple
import torch
from utils import util
import torch.nn.functional as F
import sys
sys.path.append("..")
from models.functions.base_function import spectral_norm, BasicBlock_v2, BasicBlock_v3, conv3x3, ContextualAttention, Upsample,\
    SPADEResnetBlock, get_downsampler, get_input, FeatureFusionBlock, Identity
BatchNorm = nn.BatchNorm2d
Elu = nn.ELU
ReLU = nn.ReLU


class SGINet(nn.Module):
    def __init__(self, label_categories, image_channels, norm_layer, use_deconv=True, use_attention=False,
                 use_dilated_conv = False, use_sn_generator=False, ngf=64, n_downsample_global=3, use_skip = True, activation = Elu, use_spade = False,
                 which_encoder = "ctx_label", use_pixel_shuffle = False, use_bbox=False, use_multi_scale_loss = False, gpu_ids=[0]):
        super(SGINet, self).__init__()
        self.label_categories = label_categories
        self.image_channels = image_channels
        self.use_deconv= use_deconv if not use_pixel_shuffle else False
        self.use_pixel_shuffle = use_pixel_shuffle
        self.use_sn_generator = use_sn_generator
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.use_attention = use_attention
        self.use_dilated_conv = use_dilated_conv
        self.n_downsample_global = n_downsample_global
        self.activation = activation
        self.use_spade = use_spade
        self.ngf = ngf
        self.use_bbox = use_bbox
        self.use_multi_scale_loss = use_multi_scale_loss
        feat_dim = self.ngf * 2 ** self.n_downsample_global
        self.feat_dim = feat_dim
        self.use_skip = use_skip
        self.input_channel = (self.label_categories + self.image_channels + 1 + 1) if self.use_bbox else self.label_categories + self.image_channels + 1
        self.which_encoder = which_encoder
        self.gpu_ids= gpu_ids
        if 'ctx' in which_encoder:
            self.ctx_inputEmbedder = get_input(self.image_channels + 1, norm_layer, self.use_sn_generator, ngf // 2, self.activation)
            self.ctx_downsampler = get_downsampler(n_downsample_global, ngf // 2, use_sn_generator, norm_layer, activation)
        if 'label' in which_encoder:
            self.obj_inputEmbedder = get_input((self.label_categories + 2) if self.use_bbox else self.label_categories + 1, norm_layer, self.use_sn_generator, ngf // 2, self.activation)
            self.obj_downsampler = get_downsampler(n_downsample_global, ngf // 2, use_sn_generator, norm_layer, activation)

        if which_encoder == "concat":
            self.input_embedder = get_input(self.input_channel, norm_layer, self.use_sn_generator, ngf, self.activation)
            self.downsampler = get_downsampler(n_downsample_global, ngf, use_sn_generator, norm_layer, activation)



        self.res_blocks = []
        if self.n_downsample_global < 4:
            dilation_factors = [2, 5, 9, 16, 2, 5, 9, 16, 16]
        else:
            dilation_factors = [2, 2, 2, 4, 4, 4, 8, 8, 8]

        for factor in dilation_factors:
            if self.use_dilated_conv:
                self.res_blocks += \
                    [BasicBlock_v2(inplanes=feat_dim, planes=feat_dim, stride=1,
                                   dilation=(factor, factor), norm=norm_layer,
                                   use_sn=self.use_sn_generator, activation=self.activation)]
            else:
                self.res_blocks += \
                    [BasicBlock_v2(inplanes=feat_dim, planes=feat_dim, stride=1,
                                   dilation=(1, 1), norm=norm_layer,
                                   use_sn=self.use_sn_generator, activation=self.activation)]

        self.res_blocks = nn.Sequential(*self.res_blocks)
        self.upconv = []
        self.get_inter_seg_map = []
        if self.use_deconv:
            for i in range(self.n_downsample_global):
                mult = 2 ** (self.n_downsample_global - i)
                dim_in = self.ngf * mult
                dim_out = int(self.ngf * mult / 2)
                if self.use_skip and i > 0:
                    dim_in = dim_in * 2
                if not self.use_spade:
                    self.upconv += [
                        nn.ConvTranspose2d(dim_in, dim_out,
                                           kernel_size=3, stride=2, padding=1,
                                           output_padding=1),
                        norm_layer(dim_out),
                        self.activation()]
                else:
                    self.upconv += [nn.ConvTranspose2d(dim_in, dim_out,
                                           kernel_size=3, stride=2, padding=1,
                                           output_padding=1),
                                    SPADEResnetBlock(dim_out, dim_out, label_categories - 1, gpu_ids = self.gpu_ids)]
                if self.use_multi_scale_loss:
                    self.get_inter_seg_map += [nn.Sequential(
                        nn.ReflectionPad2d(1),
                        spectral_norm(
                            nn.Conv2d(dim_out, label_categories - 1,
                                      kernel_size=3, padding=0,
                                      stride=1), self.use_sn_generator),
                    nn.LogSoftmax(dim=1))]
            dim_in = int(dim_out)
            dim_out = dim_in // 2
            self.up_conv_image = []
            if not self.use_spade:
                self.up_conv_image += [
                    nn.ReflectionPad2d(1),
                    spectral_norm(
                        nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=0,
                                  stride=1), self.use_sn_generator),
                    norm_layer(dim_out),
                    self.activation()]
            else:
                self.up_conv_image += [
                    SPADEResnetBlock(dim_in, dim_out, label_categories - 1, gpu_ids = self.gpu_ids)]
            dim_in = dim_out
            dim_out = dim_out
            self.up_conv_image += [nn.ReflectionPad2d(3),
                                   spectral_norm(
                                       nn.Conv2d(dim_out, 3, kernel_size=7,
                                                 padding=0,
                                                 stride=1),
                                       self.use_sn_generator),
                                   nn.Tanh()]
        elif self.use_pixel_shuffle:
            self.upconv = []
            self.get_inter_seg_map = []
            self.get_inter_image = []
            for i in range(self.n_downsample_global):
                mult = 2 ** (self.n_downsample_global - i)
                dim_in = self.ngf * mult
                dim_out = int(self.ngf * mult / 2)
                if self.use_skip and i > 0:
                    if self.which_encoder == "concat":
                        dim_in = dim_in * 2
                    else:
                        dim_in = dim_in + int((dim_in // 2))
                if not self.use_spade:
                    self.upconv += [
                        nn.ReflectionPad2d(1),
                        spectral_norm(
                            nn.Conv2d(dim_in, dim_out * 4, kernel_size=3, padding=0,
                                      stride=1), self.use_sn_generator),
                        nn.PixelShuffle(2),
                        norm_layer(dim_out),
                        self.activation()]
                else:
                    self.upconv += [nn.ReflectionPad2d(1),
                                    spectral_norm(nn.Conv2d(dim_in, dim_out * 4, kernel_size=3, padding=0,
                                      stride=1), self.use_sn_generator),
                                    nn.PixelShuffle(2),
                                    SPADEResnetBlock(dim_out, dim_out,
                                                     label_categories - 1, seg_probs=True, gpu_ids= self.gpu_ids)
                                    ]
                if self.use_multi_scale_loss:
                    self.get_inter_seg_map += [nn.Sequential(
                        Upsample(scale_factor=2.0, mode="nearest"),
                        nn.ReflectionPad2d(1),
                        spectral_norm(
                            nn.Conv2d(dim_out, label_categories - 1,
                                      kernel_size=3, padding=0,
                                      stride=1), self.use_sn_generator),
                    nn.LogSoftmax(dim=1))]
            dim_in = int(dim_out)
            dim_out = dim_in // 2
            self.up_conv_image = []
            if not self.use_spade:
                self.up_conv_image += [
                    nn.ReflectionPad2d(1),
                    spectral_norm(
                        nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=0,
                                  stride=1), self.use_sn_generator),
                    norm_layer(dim_out),
                    self.activation()]
            else:
                self.up_conv_image += [
                    SPADEResnetBlock(dim_in, dim_out, label_categories - 1, gpu_ids= self.gpu_ids)]
            dim_in = dim_out
            dim_out = dim_out
            self.up_conv_image += [nn.ReflectionPad2d(3),
                                   spectral_norm(
                                       nn.Conv2d(dim_out, 3, kernel_size=7,
                                                 padding=0,
                                                 stride=1),
                                       self.use_sn_generator),
                                   nn.Tanh()]
        else:
            self.upconv = []
            self.get_inter_seg_map = []
            self.get_inter_image = []
            for i in range(self.n_downsample_global):
                mult = 2 ** (self.n_downsample_global - i)
                dim_in = self.ngf * mult
                dim_out = int(self.ngf * mult / 2)
                if self.use_skip and i > 0:
                    if self.which_encoder == "concat":
                        dim_in = dim_in * 2
                    else:
                        dim_in = dim_in + int((dim_in // 2))
                if not self.use_spade:
                    self.upconv += [
                        Upsample(scale_factor=2.0),
                        nn.ReflectionPad2d(1),
                        spectral_norm(
                            nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=0,
                                      stride=1), self.use_sn_generator),
                        norm_layer(dim_out),
                        self.activation(),
                        nn.ReflectionPad2d(1),
                        spectral_norm(
                            nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=0,
                                      stride=1), self.use_sn_generator),
                        norm_layer(dim_out),
                        self.activation()]
                else:
                    self.upconv += [Upsample(scale_factor=2.0),
                                    SPADEResnetBlock(dim_in, dim_out, label_categories - 1, gpu_ids = self.gpu_ids)]
                if self.use_multi_scale_loss:
                    self.get_inter_seg_map += [nn.Sequential(
                        nn.ReflectionPad2d(1),
                        spectral_norm(
                            nn.Conv2d(dim_in, label_categories - 1,
                                      kernel_size=3, padding=0,
                                      stride=1), self.use_sn_generator),
                    nn.LogSoftmax(dim=1))]
            dim_in = int(dim_out)
            dim_out = dim_in // 2
            self.up_conv_image = []
            if not self.use_spade:
                self.up_conv_image += [
                    nn.ReflectionPad2d(1),
                    spectral_norm(
                        nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=0,
                                  stride=1), self.use_sn_generator),
                    norm_layer(dim_out),
                    self.activation()]
            else:
                self.up_conv_image += [
                    SPADEResnetBlock(dim_in, dim_out, label_categories - 1, gpu_ids = self.gpu_ids)]
            dim_in = dim_out
            dim_out = dim_out
            self.up_conv_image += [nn.ReflectionPad2d(3),
                                   spectral_norm(
                                       nn.Conv2d(dim_out, 3, kernel_size=7,
                                                 padding=0,
                                                 stride=1),
                                       self.use_sn_generator),
                                   nn.Tanh()]
        self.up_conv_image = nn.Sequential(*self.up_conv_image)
        if self.use_multi_scale_loss:
            self.get_inter_seg_map = nn.Sequential(*self.get_inter_seg_map)
        else:
            self.get_inter_seg_map = None
        self.upconv = nn.Sequential(*self.upconv)


    def forward_encoder(self, encoder, downsampler, input, use_skip):
        enc_feats = []
        enc_feat = encoder(input)
        for i, layer in enumerate(downsampler):
            enc_feat = layer(enc_feat)
            if use_skip and ((i < self.n_downsample_global*4-1) and (i % 4 == 3)): # super-duper hard-coded
                enc_feats.append(enc_feat)
        return enc_feat, enc_feats


    def forward_decoder_ps(self, decoder, dec_feat, enc_feats, get_inter_seg_map):
        embed_layer = 5
        j = 0
        multi_scale_seg_map = []
        for i, layer in enumerate(decoder):
            if "SPADE" in layer.__str__() or "Instance" in layer.__str__():
                scale_seg_map = get_inter_seg_map[j](dec_feat)
                multi_scale_seg_map.append(scale_seg_map)
                j += 1
                if "SPADE" in layer.__str__():
                    dec_feat = layer(dec_feat, scale_seg_map)
                else:
                    dec_feat = layer(dec_feat)
            else:
                dec_feat = layer(dec_feat)
        return dec_feat, multi_scale_seg_map

    def forward_decoder_deconv(self,decoder, dec_feat, enc_feats, get_inter_seg_map):
        embed_layer = 3
        j=0
        multi_scale_seg_map = []
        for i, layer in enumerate(decoder):
            if "SPADE" in layer.__str__():
                scale_seg_map = get_inter_seg_map[j](dec_feat)
                multi_scale_seg_map.append(scale_seg_map)
                j += 1
                dec_feat = layer(dec_feat, scale_seg_map)
            else:
                dec_feat = layer(dec_feat)
        return dec_feat, multi_scale_seg_map

    def forward_decoder_upsample(self,decoder, dec_feat, enc_feats, get_inter_seg_map):
        embed_layer = 9
        j = 0
        multi_scale_seg_map = []
        for i, layer in enumerate(decoder):
            if "SPADE" in layer.__str__():
                scale_seg_map = get_inter_seg_map[j](dec_feat)
                multi_scale_seg_map.append(scale_seg_map)
                j += 1
                dec_feat = layer(dec_feat, scale_seg_map)
            else:
                dec_feat = layer(dec_feat)
        return dec_feat, multi_scale_seg_map

    def forward_decoder(self, decoder, dec_feat, enc_feats, get_inter_seg_map):
        if self.use_deconv:
            return self.forward_decoder_deconv(decoder, dec_feat, enc_feats, get_inter_seg_map)
        elif self.use_pixel_shuffle:
            return self.forward_decoder_ps(decoder, dec_feat, enc_feats, get_inter_seg_map)
        else:
            return self.forward_decoder_upsample(decoder, dec_feat, enc_feats, get_inter_seg_map)


    def forward_image(self, decoder_image, dec_feat, seg_map):
        for i, layer in enumerate(decoder_image):
            if i == 0:
                dec_feat = layer(dec_feat, seg_map)
            else:
                dec_feat = layer(dec_feat)
        return dec_feat

    def forward(self, label, img, mask, instance):
        ctx_feat = obj_feat = mask_feat = 0
        ctx_feats = []
        if 'ctx' in self.which_encoder:
            img_masked = torch.cat((img, mask), dim=1)
            ctx_feat, feat = self.forward_encoder(self.ctx_inputEmbedder,
                                                       self.ctx_downsampler,
                                                       img_masked, self.use_skip)
        if 'label' in self.which_encoder:
            if self.use_bbox:
                input_concat = torch.cat([label, instance, mask], dim=1)
            else:
                input_concat = torch.cat([label, mask], dim=1)
            obj_feat, _ = self.forward_encoder(self.obj_inputEmbedder,
                                               self.obj_downsampler, input_concat,
                                               False)
            x = torch.cat((ctx_feat, obj_feat), dim=1)

        if self.which_encoder == "concat":
            if self.use_bbox:
                input_concat = torch.cat([img, label, mask, instance], dim=1)
            else:
                input_concat = torch.cat([img, label, mask], dim=1)
            x, feat = self.forward_encoder(self.input_embedder, self.downsampler, input_concat, self.use_skip)
        x = self.res_blocks(x)
        offset_flow = [None]
        x, multi_scale_seg_map = self.forward_decoder(self.upconv, x, feat, self.get_inter_seg_map)
        if not self.use_spade:
            image_inp = self.up_conv_image(x)
        else:
            image_inp = self.forward_image(self.up_conv_image, x, multi_scale_seg_map[-1])
        return multi_scale_seg_map, image_inp, offset_flow



class Shape_Encoder(nn.Module):
    def __init__(self, ngf=64, z_len=1024, input_dim=4096, classes= 31, theta_len = 6):
        super(Shape_Encoder, self).__init__()

        self.f_dim = ngf
        self.z_len = z_len
        self.input_dim = input_dim
        self.classes = classes
        self.theta_len = theta_len
        self.lin0 = nn.Sequential(
            nn.Linear(self.input_dim + self.classes + theta_len, 32 * 32 * 8),
            nn.LeakyReLU(0.2),
        )
        self.conv0 = nn.Sequential(
            nn.Conv2d(8, self.f_dim,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.f_dim, self.f_dim * 2,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.f_dim * 2, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.f_dim * 2, self.f_dim * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.f_dim * 4, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.f_dim * 4, self.f_dim * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.f_dim * 8, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Sequential(
            nn.Conv2d(self.f_dim * 8, self.z_len,
                      kernel_size=4, stride=1, padding=0),
        )
        self.fc_logvar = nn.Sequential(
            nn.Conv2d(self.f_dim * 8, self.z_len,
                      kernel_size=4, stride=1, padding=0),
        )

    def forward(self, inst_shape, conditioning, theta):
        inst_shape_flatten = inst_shape.view(inst_shape.shape[0], -1)
        l0 = self.lin0(torch.cat([inst_shape_flatten, conditioning.squeeze(2).squeeze(2), theta], dim=1))
        l0_reshaped = l0.view(l0.shape[0], 8, 32, 32)
        e0 = self.conv0(l0_reshaped)
        e1 = self.conv1(e0)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        mu = self.fc_mu(e3)
        logvar = self.fc_logvar(e3)

        return mu, logvar


class Shape_Decoder(nn.Module):
    def __init__(self, z_len, label_nc, theta_len = 6, ngf=32):
        super(Shape_Decoder, self).__init__()
        self.z_dim = z_len
        self.f_dim = ngf
        self.label_nc = label_nc
        self.theta_len = theta_len
        self.conv0 = nn.Sequential(
            nn.ConvTranspose2d(self.z_dim + label_nc + theta_len, self.f_dim * 8,
                      kernel_size=4, stride=1, padding=0),
            nn.InstanceNorm2d(self.f_dim, affine=False),
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(self.f_dim * 8, self.f_dim * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.f_dim * 4, affine=False),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(self.f_dim * 4, self.f_dim * 2,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.f_dim * 2, affine=False),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(self.f_dim * 2, self.f_dim,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.f_dim, affine=False),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(self.f_dim, 1,
                      kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z, conditioning, theta):

        e1 = self.conv0(torch.cat([z, conditioning, theta.unsqueeze(2).unsqueeze(2)], dim=1))
        e2 = self.conv1(e1)
        e3 = self.conv2(e2)
        e4 = self.conv3(e3)
        instance = self.conv4(e4)
        return instance