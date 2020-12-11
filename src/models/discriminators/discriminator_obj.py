from torch import nn
import sys

class DiscriminatorObj(nn.Module):
    def __init__(self, ndf=64):
        super(DiscriminatorObj, self).__init__()

        f_dim = ndf

        self.conv0 = nn.Sequential(
            nn.Conv2d(1, f_dim * 1,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(f_dim * 1, f_dim * 2,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 2, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(f_dim * 2, f_dim * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 4, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(f_dim * 4, f_dim * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 8, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(f_dim * 8, 1,
                      kernel_size=4, stride=1, padding=0)
        )

    def forward(self, x):
        d0 = self.conv0(x)
        d1 = self.conv1(d0)
        d2 = self.conv2(d1)
        d3 = self.conv3(d2)
        disc = self.conv4(d3)
        return disc