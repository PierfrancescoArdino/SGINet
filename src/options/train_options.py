from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # for displays
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')

        # for training
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--fineTuning', action='store_true',
                                 help='finetune training: load the latest model')
        self.parser.add_argument('--load_pretrain_sp', type=str, default='', help='load the pretrained model for SP-NET from the specified location')
        self.parser.add_argument('--load_pretrain_sg', type=str, default='',
                                 help='load the pretrained model for SG-NET from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
        self.parser.add_argument('--beta2', type=float, default=0.9,
                                 help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--no_deconv', action='store_true', help='do *not* use deconv layer, if false, use upsampling + conv')
        self.parser.add_argument('--use_attention', action='store_true',
                                 help='do use contextual attention')
        self.parser.add_argument('--use_discounted_mask', action='store_true',
                                 help='do use discounted mask for l1 loss')
        self.parser.add_argument('--use_sn_generator', action='store_true',
                                 help='do use spectral norm in generator')
        self.parser.add_argument('--use_sn_discriminator', action='store_true',
                                 help='do use spectral norm in discriminator')
        self.parser.add_argument('--no_dilated_conv', action='store_true',
                                 help='do use resblock without dilation instead of resblock+dilation')
        self.parser.add_argument('--min_hole_size', type=int, default=32, help='min size of missing hole')
        self.parser.add_argument('--max_hole_size', type=int, default=128, help="max size of missing hole")

        # for discriminators
        self.parser.add_argument('--num_D_local', type=int, default=1, help='number of local discriminators to use')
        self.parser.add_argument('--num_D_global', type=int, default=1,
                                 help='number of global discriminators to use')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        self.parser.add_argument('--lambda_rec', type=float, default=10.0,
                                 help='weight for reconstruction loss')
        self.parser.add_argument('--lambda_KL_rec', type=float, default=5.0,
                                 help='weight for KL loss')
        self.parser.add_argument('--lambda_inst_rec', type=float, default=20.0,
                                 help='weight for instance reconstruction loss')
        self.parser.add_argument('--lambda_seg_map', type=float, default=5.0,
                                 help='weight for reconstruction loss')
        self.parser.add_argument('--lambda_perceptual', type=float, default=10.0,
                                 help='weight for perceptual loss')
        self.parser.add_argument('--lambda_style', type=float,
                                 default=10.0,
                                 help='weight for style loss')
        self.parser.add_argument('--gp_lambda', type=float, default=10.0, help="weight for gradient penalty")
        self.parser.add_argument('--spatial_discounting_gamma', type=float, default=0.9,
                                 help="gamma for discounting mask")
        self.parser.add_argument('--n_critic', type=int, default=5, help="critic number for wgan training")
        self.parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        self.parser.add_argument('--which_perceptual_loss', type=str, help='specify wich perceptuall loss to use', choices=['vgg','alex','None'], default="vgg")
        self.parser.add_argument('--no_ganStyle_loss', action='store_true',
                                 help='if specified, do *not* use Gan Style loss')
        self.parser.add_argument('--no_ganTV_loss', action='store_true',
                                 help='if specified, do *not* use TV loss')
        self.parser.add_argument('--gan_mode', type=str, default='lsgan', choices=['vanilla','lsgan','wgangp','wganr1'], help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')


        self.isTrain = True
