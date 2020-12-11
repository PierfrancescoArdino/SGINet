import argparse
import os
from utils import util
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--name', type=str, default='SGI_NET_debug',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--gpu_ids', type=str, default='-1',
                                 help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str,
                                 default='./checkpoints',
                                 help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='instance',
                                 help='instance normalization or batch normalization')
        self.parser.add_argument('--activation', type=str, default='elu',
                                 help='relu or elu activation')
        self.parser.add_argument('--verbose', action='store_true',
                                 default=False, help='toggles verbose')
        self.parser.add_argument('--nThreads', default=2, type=int,
                                 help='# threads for loading data')
        self.parser.add_argument('--how_many', type=int, default=50,
                                 help='how many test images to run')
        self.parser.add_argument('--no_flip', action='store_true',
                                 help='if specified, do not flip the images for data argumentation')
        self.parser.add_argument("--size_crop_height", type=int, help="height of crop")
        self.parser.add_argument("--size_crop_width", type=int, help="width of crop")
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--n_downsample_global', type=int, default=3, help='number of downsampling layers in netG')
        self.parser.add_argument('--prob_bg', type=float, default=0.0,
                                 help='probablity of sampling random background patches')


        # input/output sizes
        self.parser.add_argument('--batchSize', type=int, default=1,
                                 help='input batch size')
        self.parser.add_argument('--label_nc', type=int, default=28,
                                 help='# of label channels')
        self.parser.add_argument('--no_contain_dontcare_label', action='store_true',
                                 help="DO NOT use additional channel for masked region, use background class instead")
        self.parser.add_argument('--input_nc', type=int, default=3,
                                 help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3,
                                 help='# of output image channels')
        self.parser.add_argument('--image_height',type=int, default=256, help="image_height")
        self.parser.add_argument('--image_width',type=int, default=256, help="image_width")
        self.parser.add_argument('--fp16', action='store_true', default=False,
                                 help='train with AMP')
        self.parser.add_argument("--use_bbox", action='store_true', help="use class conditioning", default=False)
        self.parser.add_argument("--classes_of_interest", type=str, help='class conditioning')
        self.parser.add_argument("--z_len", type=int, default=1024)
        self.parser.add_argument('--compact_sizex', type=int, default=64)
        self.parser.add_argument('--compact_sizey', type=int, default=64)
        self.parser.add_argument("--use_skip", action='store_true', help="use skip connections")
        self.parser.add_argument("--use_spade", action='store_true', help="use spade normalization")
        self.parser.add_argument("--local_rank", type=int, default=0)
        self.parser.add_argument("--local_world_size", type=int, default=1)
        self.parser.add_argument('--which_encoder', type=str, default='concat', choices=["concat", "ctx_label"],
                                 help='which encoder: concatenate image and seg encoder or use two encoder for context and label')
        self.parser.add_argument("--use_load_mask", action='store_true')
        self.parser.add_argument("--use_gt_instance_encoder", action='store_true')
        self.parser.add_argument("--use_pixel_shuffle",
                                 action='store_true')
        self.parser.add_argument("--use_multi_scale_loss", action="store_true", help="multi scale cross entropy loss")


        # for setting inputs
        self.parser.add_argument('--dataroot', type=str,
                                 default='../datasets')
        self.parser.add_argument("--dataset", type=str, choices=["cityscapes", "indianDrivingDataset"], default="cityscapes")

        # for displays
        self.parser.add_argument('--display_winsize', type=int, default=512,
                                 help='display window size')
        self.parser.add_argument('--tf_log', action='store_true',
                                 help='if specified, use tensorboard logging. Requires tensorflow installed')
        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        self.opt.semantic_nc = self.opt.label_nc + \
                          (0 if self.opt.no_contain_dontcare_label else 1)
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        self.opt.classes_of_interest_ids = []
        if self.opt.use_bbox is not False:
            for str_class_id in self.opt.classes_of_interest.split(','):
                id = int(str_class_id)
                if id >= 0:
                    self.opt.classes_of_interest_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)
        if self.opt.local_rank == 0:
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')

        # save to the disk
        folder_name = "SGI-NET"
        if self.opt.phase == "test":
            expr_dir = os.path.join(self.opt.checkpoints_dir, folder_name, self.opt.name, "test")
        else:
            expr_dir = os.path.join(self.opt.checkpoints_dir, folder_name,
                                    self.opt.name)
        util.mkdirs(expr_dir)
        if save and not self.opt.continue_train:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
