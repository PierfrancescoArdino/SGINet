import torch
from torch import nn
from collections import namedtuple
from utils import util


class GANLoss(nn.Module):
    def __init__(self, gan_mode = "lsgan", target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, device="cpu"):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        self.device = device
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp', "wganr1"]:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            self.real_label_var = self.Tensor(input.size()).fill_(self.real_label)
            self.real_label_var.requires_grad = False
            target_tensor = self.real_label_var
        else:
            self.fake_label_var = self.Tensor(input.size()).fill_(self.fake_label)
            self.fake_label_var.requires_grad = False
            target_tensor = self.fake_label_var
        return target_tensor.to(self.device)

    def forward(self, input, target_is_real):
        if not isinstance(input, list) and input.shape[0] == 0:
            return 0
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                if self.gan_mode in ['lsgan', 'vanilla']:
                    target_tensor = self.get_target_tensor(pred, target_is_real)
                    loss += self.loss(pred, target_tensor)
                else:
                    if target_is_real:
                        loss += -torch.mean(pred)
                    else:
                        loss += torch.mean(pred)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)

# Learned perceptual metric
class AlexNetLoss(nn.Module):
    def __init__(self, pnet_type='alex', pnet_rand=False, pnet_tune=False,
                 use_dropout=True, gpu_ids=None):
        super(AlexNetLoss, self).__init__()

        self.pnet_type = pnet_type
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        print(gpu_ids)
        self.gpu_ids = gpu_ids
        self.scaling_layer = ScalingLayer()
        net_type = AlexNet
        self.chns = [64, 192, 384, 256, 256]
        self.L = len(self.chns)

        if len(gpu_ids) > 0:
            self.net = net_type().to("cuda")
        else:
            self.net = net_type()
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)


    def forward(self, in0, in1, retPerLayer=False):
        # v0.0 - original release had a bug, where input was not scaled
        self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(
            in1))
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = util.normalize_tensor(
                outs0[kk]), util.normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(self.lins[kk].model(diffs[kk]),
                               keepdim=True) for kk in range(self.L)]

        val = res[0]
        for l in range(1, self.L):
            val += res[l]

        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift',
                             torch.Tensor([-.030, -.088, -.188])[None, :, None,
                             None])
        self.register_buffer('scale',
                             torch.Tensor([.458, .448, .450])[None, :, None,
                             None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [
            nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)

class AlexNetLoss1(nn.Module):
    def __init__(self, gpu_ids):
        super(AlexNetLoss1, self).__init__()
        if len(gpu_ids) > 0:
            self.alexnet = AlexNet().to("cuda")
        else:
            self.alexnet = AlexNet()
        self.criterion = nn.MSELoss(reduction="none")

    def forward(self, x, y):
        x_alexnet, y_alexnet = self.alexnet(x), self.alexnet(y)
        loss = 0
        for i in range(len(x_alexnet)):
            const_shape = 1 / (x_alexnet[i].shape[2]*x_alexnet[i].shape[3])
            loss += const_shape * self.criterion(x_alexnet[i],
                                                 y_alexnet[i].detach()).mean(1).sum(1).sum(1)
        return loss


from torchvision import models


class Style_Perceptual_Loss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0], gpu_ids=[], fp16=False):
        super(Style_Perceptual_Loss, self).__init__()
        if len(gpu_ids) > 0:
            self.add_module('vgg', VGG19().to("cuda"))
        else:
            self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights
        self.fp16 = fp16

    def compute_gram(self, x):
        if self.fp16:
            from apex import amp
            with amp.disable_casts():
                x = x.type(torch.cuda.FloatTensor)
                b, ch, h, w = x.size()
                f = x.view(b, ch, w * h)
                f_T = f.transpose(1, 2)
                G = f.bmm(f_T) / (h * w * ch)
        else:
            b, ch, h, w = x.size()
            f = x.view(b, ch, w * h)
            f_T = f.transpose(1, 2)
            G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y, compute_perceptual, compute_style, normalize = False):
        # Compute features
        if normalize:
            x_norm = (x + 1) * 0.5
            x = util.normalize(x_norm, mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            y_norm = (y + 1) * 0.5
            y = util.normalize(y_norm, mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        style_loss = 0.0
        if compute_style:
            # Compute loss
            style_loss = 0.0
            style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
            style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
            style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
            style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))
        content_loss = 0.0
        if compute_perceptual:
            content_loss = 0.0
            content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'],
                                                             y_vgg['relu1_1'])
            content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'],
                                                             y_vgg['relu2_1'])
            content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'],
                                                             y_vgg['relu3_1'])
            content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'],
                                                             y_vgg['relu4_1'])
            content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'],
                                                             y_vgg['relu5_1'])


        return style_loss, content_loss


def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
        torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss


class L1ReconLoss(torch.nn.Module):
    """
    L1 Reconstruction loss for two imgae
    """
    def __init__(self, weight=1):
        super(L1ReconLoss, self).__init__()
        self.weight = weight

    def forward(self, imgs, recon_imgs, masks=None):
        if masks is None:
            return self.weight * torch.mean(torch.abs(imgs - recon_imgs))
        else:
            #print(masks.view(masks.size(0), -1).mean(1).size(), imgs.size())
            return self.weight * torch.mean(torch.abs(imgs - recon_imgs) / masks.view(masks.size(0), -1).mean(1).view(-1,1,1,1))


class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out


class AlexNet(nn.Module):
    def __init__(self, requires_grad=False):
        super(AlexNet, self).__init__()
        alexnet_pretrained_features = models.alexnet(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        alexnet_outputs = namedtuple("AlexnetOutputs",
                                     ['relu1', 'relu2', 'relu3', 'relu4',
                                      'relu5'])
        out = alexnet_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)

        return out
