
# ---------------------------------------------------------------------------- #
# An implementation of https://arxiv.org/pdf/1805.03356.pdf                    #
# See section 3.2 for the model architecture of SG-NET                         #
# Some part of the code was referenced from below                              #
#                                                                              #
# ---------------------------------------------------------------------------- #


from torch import nn
from collections import namedtuple
import torch
from utils import util
import torch.nn.functional as F
import re
from models.functions.sync_batchnorm import SynchronizedBatchNorm2d
import torch.nn.utils.spectral_norm as spectral_norm
import numpy as np
from torch.autograd import Variable
BatchNorm = nn.BatchNorm2d
Elu = nn.ELU
ReLU = nn.ReLU


class Upsample(nn.Module):
    def __init__(self,  scale_factor, mode="bilinear"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners= True if self.mode=="bilinear" else None)


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, use_sn=True):
    return nn.Sequential(nn.ReflectionPad2d(padding),
                         spectral_norm(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=0, bias=not use_sn, dilation=dilation), use_sn))



def get_input(input_nc, norm_layer, use_sn_generator, ngf, activation):
    encoder = [nn.ReflectionPad2d(2),
                    spectral_norm(
                        nn.Conv2d(input_nc, ngf, kernel_size=5,
                                  padding=0,
                                  stride=1), use_sn_generator),
                    norm_layer(ngf),
                    activation()]
    return nn.Sequential(*encoder)


def get_downsampler(n_downsample_global, ngf, use_sn_generator, norm_layer, activation):
    downsample = []
    for i in range(n_downsample_global):
        mult = 2 ** i
        downsample += [nn.ReflectionPad2d(1),
                            spectral_norm(
                                nn.Conv2d(ngf * mult, ngf * 2 * mult,
                                          kernel_size=4,
                                          padding=0, stride=2),
                                use_sn_generator),
                            norm_layer(ngf * 2 * mult),
                            activation()]
    return nn.Sequential(*downsample)


class Identity(nn.Module):
    def __init__(self, *values):
        super(Identity, self).__init__()

    def forward(self, input_var):
        return input_var


class FeatureFusionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, fusion_type,
                 norm_fn, activation_fn, main_module):
        super(FeatureFusionBlock, self).__init__()
        #assert in_planes == out_planes
        assert fusion_type in ['add', 'concat'] # 'deep']
        self.fusion_type = fusion_type
        self.main_module = main_module
        self.norm_fn = norm_fn
        self.activation_fn = activation_fn
        if fusion_type == 'concat':
            self.fuse_module = self.initialize_concat_layer(in_planes)
        else:
            self.fuse_module = self.initialize_deep_layer(in_planes)

    def initialize_concat_layer(self, in_planes):
        self.nonlinear1 = self.activation_fn
        self.conv1 = nn.Conv2d(in_planes * 2, in_planes,
            kernel_size=1, stride=1, padding=0)
        self.norm1 = self.norm_fn(in_planes)

    #def initialize_deep_layer(self, in_planes, out_planes):
    def initialize_deep_layer(self, in_planes):
        pass

    def forward(self, x, y):
        #out = self.main_module(x)
        if self.fusion_type == 'add':
            out = x + y
        elif self.fusion_type == 'concat':
            out = torch.cat([x, y], 1)
            out = self.nonlinear1(out)
            out = self.conv1(out)
            out = self.norm1(out)
        out = self.main_module(out)
        return out


def weights_init_horizontal1(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        filter = torch.FloatTensor(1, 1, 1, 2)
        filter[0, 0, 0, 0] = -1
        filter[0, 0, 0, 1] = 1
        m.weight.data = filter


def weights_init_horizontal2(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        filter = torch.FloatTensor(1, 1, 1, 2)
        filter[0, 0, 0, 0] = 1
        filter[0, 0, 0, 1] = -1
        m.weight.data = filter


def weights_init_vertical1(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        filter = torch.FloatTensor(1, 1, 2, 1)
        filter[0, 0, 0, 0] = -1
        filter[0, 0, 1, 0] = 1
        m.weight.data = filter


def weights_init_vertical2(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        filter = torch.FloatTensor(1, 1, 2, 1)
        filter[0, 0, 0, 0] = 1
        filter[0, 0, 1, 0] = -1
        m.weight.data = filter


class NetEdgeHorizontal1(nn.Module):
    def __init__(self):
        super(NetEdgeHorizontal1, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=(1, 2), stride=1, padding=0, bias=False)
        self.conv.apply(weights_init_horizontal1)
        self.conv.weight.requires_grad = False

        self.pad = nn.ZeroPad2d((1, 0, 0, 0))
        self.relu = nn.ReLU()

    def forward(self, mask):
        edge = self.conv(mask)
        edge = self.pad(edge)
        edge = self.relu(edge)
        return edge


class NetEdgeHorizontal2(nn.Module):
    def __init__(self):
        super(NetEdgeHorizontal2, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=(1, 2), stride=1, padding=0, bias=False)
        self.conv.apply(weights_init_horizontal2)
        self.conv.weight.requires_grad = False

        self.pad = nn.ZeroPad2d((0, 1, 0, 0))
        self.relu = nn.ReLU()

    def forward(self, mask):
        edge = self.conv(mask)
        edge = self.pad(edge)
        edge = self.relu(edge)
        return edge


class NetEdgeVertical1(nn.Module):
    def __init__(self):
        super(NetEdgeVertical1, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=(2, 1), stride=1, padding=0, bias=False)
        self.conv.apply(weights_init_vertical1)
        self.conv.weight.requires_grad = False

        self.pad = nn.ZeroPad2d((0, 0, 1, 0))
        self.relu = nn.ReLU()

    def forward(self, mask):
        edge = self.conv(mask)
        edge = self.pad(edge)
        edge = self.relu(edge)
        return edge


class NetEdgeVertical2(nn.Module):
    def __init__(self):
        super(NetEdgeVertical2, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=(2, 1), stride=1, padding=0, bias=False)
        self.conv.apply(weights_init_vertical2)
        self.conv.weight.requires_grad = False

        self.pad = nn.ZeroPad2d((0, 0, 0, 1))
        self.relu = nn.ReLU()

    def forward(self, mask):
        edge = self.conv(mask)
        edge = self.pad(edge)
        edge = self.relu(edge)
        return edge


class STN_fixTheta(nn.Module):
    def __init__(self, opt):
        super(STN_fixTheta, self).__init__()
        self.batch_size = opt.batchSize

    def forward(self, gt, theta, output_sizex, output_sizey):

        theta = theta.view(-1, 2, 3)

        gt_up = nn.Upsample(scale_factor=8)(gt)
        grid = F.affine_grid(theta, torch.Size([self.batch_size, 1, output_sizey, output_sizex]))
        box = F.grid_sample(gt_up, grid)
        return box


# ResNet block that uses SPADE.
# It differs from the ResNet block of pix2pixHD in that
# it takes in the segmentation map as input, learns the skip connection if necessary,
# and applies normalization first and then convolution.
# This architecture seemed like a standard architecture for unconditional or
# class-conditional GAN architecture using residual block.
# The code was inspired from https://github.com/LMescheder/GAN_stability.


class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, semantic_nc, seg_probs = True, gpu_ids = [0]):

        super().__init__()
        norm_G = 'spadeinstance3x3'
        # Attributes
        self.seg_probs = seg_probs
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = norm_G.replace('spectral', '')
        self.norm_0 = SPADE(spade_config_str, fin, semantic_nc)
        self.norm_1 = SPADE(spade_config_str, fmiddle, semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, semantic_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg_map=None):
        max_idx = torch.argmax(seg_map, 1, keepdim=True)
        seg = torch.FloatTensor(seg_map.shape).cuda()
        seg.zero_()
        seg.scatter_(1, max_idx, 1)
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)




class BasicBlock_v3(nn.Module):
    """
     A modified implementation of residual blocks with dilatation factor
     from https://arxiv.org/abs/1705.09914
     code: https://github.com/fyu/drn/blob/master/drn.py
     v3 from: https://arxiv.org/pdf/1603.05027.pdf
     """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True, norm=BatchNorm, use_sn=True):
        super(BasicBlock_v3, self).__init__()
        self.conv_block = [norm(planes),
                           nn.ReLU(inplace=True),
                           conv3x3(inplanes, planes, stride,
                                   padding=dilation[0], dilation=dilation[0],
                                   use_sn=use_sn)
                           ]

        self.conv_block += [norm(planes),
                            nn.ReLU(inplace=True),
                            conv3x3(planes, planes,
                                    padding=dilation[0], dilation=dilation[0],
                                    use_sn=use_sn)
                            ]
        self.conv_block = nn.Sequential(*self.conv_block)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class BasicBlock_v2(nn.Module):
    """
     A modified implementation of residual blocks with dilatation factor
     from https://arxiv.org/abs/1705.09914
     code: https://github.com/fyu/drn/blob/master/drn.py
     """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True, norm=BatchNorm, use_sn=True, final_activation=False, activation=ReLU):
        super(BasicBlock_v2, self).__init__()
        self.activation = activation
        self.conv_block = [conv3x3(inplanes, planes, stride,
                                   padding=dilation[0], dilation=dilation[0], use_sn=use_sn),
                           norm(planes),
                           self.activation()]

        self.conv_block += [conv3x3(planes, planes,
                                    padding=1, dilation=1, use_sn=use_sn),
                            norm(planes)]
        self.conv_block = nn.Sequential(*self.conv_block)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual
        self.final_activation = final_activation

    def forward(self, x):
        if self.final_activation:
            out = self.activation(x + self.conv_block(x))
        else:
            out = x + self.conv_block(x)
        return out


class BasicBlock(nn.Module):
    """
    An implementation of residual blocks with dilatation factor
    from https://arxiv.org/abs/1705.09914
    code: https://github.com/fyu/drn/blob/master/drn.py
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride,
                             padding=dilation[0], dilation=dilation[0])
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes,
                             padding=dilation[1], dilation=dilation[1])
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)

        return out


class ContextualAttention(nn.Module):
    def __init__(self, ksize=3, stride=1, rate=1, fuse_k=3, softmax_scale=10,
                 fuse=False, use_cuda=False, device_ids=None):
        super(ContextualAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse
        self.use_cuda = use_cuda
        self.device_ids = device_ids

    def forward(self, f, b, mask=None):
        """ Contextual attention layer implementation.
        Contextual attention is first introduced in publication:
            Generative Image Inpainting with Contextual Attention, Yu et al.
        Args:
            f: Input feature to match (foreground).
            b: Input feature for match (background).
            mask: Input mask for b, indicating patches not available.
            ksize: Kernel size for contextual attention.
            stride: Stride for extracting patches from b.
            rate: Dilation for matching.
            softmax_scale: Scaled softmax for attention.
        Returns:
            torch.tensor: output
        """
        # get shapes
        raw_int_fs = list(f.size())   # b*c*h*w
        raw_int_bs = list(b.size())   # b*c*h*w

        # extract patches from background with stride and rate
        kernel = 2 * self.rate
        # raw_w is extracted for reconstruction
        raw_w = util.extract_image_patches(b, ksizes=[kernel, kernel],
                                      strides=[self.rate*self.stride,
                                               self.rate*self.stride],
                                      rates=[1, 1],
                                      padding='same') # [N, C*k*k, L]
        # raw_shape: [N, C, k, k, L]
        raw_w = raw_w.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        raw_w = raw_w.permute(0, 4, 1, 2, 3)    # raw_shape: [N, L, C, k, k]
        raw_w_groups = torch.split(raw_w, 1, dim=0)

        # downscaling foreground option: downscaling both foreground and
        # background for matching and use original background for reconstruction.
        f = F.interpolate(f, scale_factor=1./self.rate, mode='nearest')
        b = F.interpolate(b, scale_factor=1./self.rate, mode='nearest')
        int_fs = list(f.size())     # b*c*h*w
        int_bs = list(b.size())
        f_groups = torch.split(f, 1, dim=0)  # split tensors along the batch dimension
        # w shape: [N, C*k*k, L]
        w = util.extract_image_patches(b, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
        # w shape: [N, C, k, k, L]
        w = w.view(int_bs[0], int_bs[1], self.ksize, self.ksize, -1)
        w = w.permute(0, 4, 1, 2, 3)    # w shape: [N, L, C, k, k]
        w_groups = torch.split(w, 1, dim=0)

        # process mask
        if mask is None:
            mask = torch.zeros([int_bs[0], 1, int_bs[2], int_bs[3]])
            if self.use_cuda:
                mask = mask.cuda()
        else:
            mask = F.interpolate(mask, scale_factor=1./(4*self.rate), mode='nearest')
        int_ms = list(mask.size())
        # m shape: [N, C*k*k, L]
        m = util.extract_image_patches(mask, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
        # m shape: [N, C, k, k, L]
        m = m.view(int_ms[0], int_ms[1], self.ksize, self.ksize, -1)
        m = m.permute(0, 4, 1, 2, 3)    # m shape: [N, L, C, k, k]
        m = m[0]    # m shape: [L, C, k, k]
        # mm shape: [L, 1, 1, 1]
        mm = (util.reduce_mean(m, axis=[1, 2, 3], keepdim=True)==0.).to(torch.float32)
        mm = mm.permute(1, 0, 2, 3) # mm shape: [1, L, 1, 1]

        y = []
        offsets = []
        k = self.fuse_k
        scale = self.softmax_scale    # to fit the PyTorch tensor image value range
        fuse_weight = torch.eye(k).view(1, 1, k, k)  # 1*1*k*k
        if self.use_cuda:
            fuse_weight = fuse_weight.cuda()

        for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
            '''
            O => output channel as a conv filter
            I => input channel as a conv filter
            xi : separated tensor along batch dimension of front; (B=1, C=128, H=32, W=32)
            wi : separated patch tensor along batch dimension of back; (B=1, O=32*32, I=128, KH=3, KW=3)
            raw_wi : separated tensor along batch dimension of back; (B=1, I=32*32, O=128, KH=4, KW=4)
            '''
            # conv for compare
            escape_NaN = torch.FloatTensor([1e-4])
            if self.use_cuda:
                escape_NaN = escape_NaN.cuda()
            wi = wi[0]  # [L, C, k, k]
            max_wi = torch.max(torch.sqrt(util.reduce_sum(torch.pow(wi, 2),
                                                     axis=[1, 2, 3],
                                                     keepdim=True)),
                               escape_NaN)
            wi_normed = wi / max_wi
            # xi shape: [1, C, H, W], yi shape: [1, L, H, W]
            xi = util.same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])  # xi: 1*c*H*W
            yi = F.conv2d(xi, wi_normed, stride=1)   # [1, L, H, W]
            # conv implementation for fuse scores to encourage large patches
            if self.fuse:
                # make all of depth to spatial resolution
                yi = yi.view(1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])  # (B=1, I=1, H=32*32, W=32*32)
                yi = util.same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)  # (B=1, C=1, H=32*32, W=32*32)
                yi = yi.contiguous().view(1, int_bs[2], int_bs[3], int_fs[2], int_fs[3])  # (B=1, 32, 32, 32, 32)
                yi = yi.permute(0, 2, 1, 4, 3)
                yi = yi.contiguous().view(1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])
                yi = util.same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)
                yi = yi.contiguous().view(1, int_bs[3], int_bs[2], int_fs[3], int_fs[2])
                yi = yi.permute(0, 2, 1, 4, 3).contiguous()
            yi = yi.view(1, int_bs[2] * int_bs[3], int_fs[2], int_fs[3])  # (B=1, C=32*32, H=32, W=32)
            # softmax to match
            yi = yi * mm
            yi = F.softmax(yi*scale, dim=1)
            yi = yi * mm  # [1, L, H, W]

            offset = torch.argmax(yi, dim=1, keepdim=True)  # 1*1*H*W

            if int_bs != int_fs:
                # Normalize the offset value to match foreground dimension
                times = float(int_fs[2] * int_fs[3]) / float(int_bs[2] * int_bs[3])
                offset = ((offset + 1).float() * times - 1).to(torch.int64)
            offset = torch.cat([offset//int_fs[3], offset%int_fs[3]], dim=1)  # 1*2*H*W

            # deconv for patch pasting
            wi_center = raw_wi[0]
            # yi = F.pad(yi, [0, 1, 0, 1])    # here may need conv_transpose same padding
            yi = F.conv_transpose2d(yi, wi_center, stride=self.rate, padding=1) / 4.  # (B=1, C=128, H=64, W=64)
            y.append(yi)
            offsets.append(offset)

        y = torch.cat(y, dim=0)  # back to the mini-batch
        y.contiguous().view(raw_int_fs)

        offsets = torch.cat(offsets, dim=0)
        offsets = offsets.view(int_fs[0], 2, *int_fs[2:])

        # case1: visualize optical flow: minus current position
        h_add = torch.arange(int_fs[2]).view([1, 1, int_fs[2], 1]).expand(int_fs[0], -1, -1, int_fs[3])
        w_add = torch.arange(int_fs[3]).view([1, 1, 1, int_fs[3]]).expand(int_fs[0], -1, int_fs[2], -1)
        ref_coordinate = torch.cat([h_add, w_add], dim=1)
        if self.use_cuda:
            ref_coordinate = ref_coordinate.cuda()

        offsets = offsets - ref_coordinate
        # flow = pt_flow_to_image(offsets)

        flow = torch.from_numpy(util.flow_to_image(offsets.permute(0, 2, 3, 1).cpu().data.numpy())) / 255.
        flow = flow.permute(0, 3, 1, 2)
        if self.use_cuda:
            flow = flow.cuda()
        # case2: visualize which pixels are attended
        # flow = torch.from_numpy(highlight_flow((offsets * mask.long()).cpu().data.numpy()))

        if self.rate != 1:
            flow = F.interpolate(flow, scale_factor=self.rate*4, mode='nearest')

        return y, flow


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module