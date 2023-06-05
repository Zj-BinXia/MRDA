import torch
from torch import nn
import model.common as common
import torch.nn.functional as F


def make_model(args):
    return BlindSR(args)

class RDA_conv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, reduction):
        super(RDA_conv, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size

        self.kernel = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Linear(64, 64 * self.kernel_size * self.kernel_size, bias=False)
        )
        self.conv = common.default_conv(channels_in, 1, 1)
        self.relu = nn.LeakyReLU(0.1, True)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        b, c, h, w = x[0].size()

        # branch 1
        kernel = self.kernel(x[1]).view(-1, 1, self.kernel_size, self.kernel_size)
        out = self.relu(F.conv2d(x[0].view(1, -1, h, w), kernel, groups=b*c, padding=(self.kernel_size-1)//2))
        out = out.view(b, -1, h, w)
        M = self.sig(self.conv(x[0]))
        # branch 2
        out = out*M + x[0]

        return out


class RDAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction):
        super(RDAB, self).__init__()

        self.da_conv1 = RDA_conv(n_feat, n_feat, kernel_size, reduction)
        self.da_conv2 = RDA_conv(n_feat, n_feat, kernel_size, reduction)
        self.conv1 = conv(n_feat, n_feat, kernel_size)
        self.conv2 = conv(n_feat, n_feat, kernel_size)

        self.relu =  nn.LeakyReLU(0.1, True)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''

        out = self.relu(self.da_conv1(x))
        out = self.relu(self.conv1(out))
        out = self.relu(self.da_conv2([out, x[1]]))
        out = self.conv2(out) + x[0]

        return out


class RDAG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_blocks):
        super(RDAG, self).__init__()
        self.n_blocks = n_blocks
        modules_body = [
            RDAB(conv, n_feat, kernel_size, reduction) \
            for _ in range(n_blocks)
        ]
        #modules_body.append(conv(n_feat, n_feat, kernel_size))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        res = x[0]
        for i in range(self.n_blocks):
            res = self.body[i]([res, x[1]])


        return res


class RDAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(RDAN, self).__init__()
        n_blocks = 27
        n_feats = 64
        kernel_size = 3
        reduction = 8
        scale = int(args.scale[0])

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        # head module
        modules_head = [conv(3, n_feats, kernel_size)]
        self.head = nn.Sequential(*modules_head)

        # compress
        self.compress = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.LeakyReLU(0.1, True)
        )

        # body
        modules_body = [
            RDAG(common.default_conv, n_feats, kernel_size, reduction, n_blocks)
        ]
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

        # tail
        modules_tail = [common.Upsampler(conv, scale, n_feats, act=False),
                        conv(n_feats, 3, kernel_size)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x, k_v):
        k_v = self.compress(k_v)

        # sub mean
        x = self.sub_mean(x)

        # head
        x = self.head(x)

        # body
        res = x
        res = self.body[0]([res, k_v])
        res = self.body[-1](res)
        res = res + x

        # tail
        x = self.tail(res)

        # add mean
        x = self.add_mean(x)

        return x


class DEN(nn.Module):
    def __init__(self,args):
        super(DEN, self).__init__()
        n_feats = args.n_feats
        self.E = nn.Sequential(
            nn.Conv2d(n_feats, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        fea = self.E(x).squeeze(-1).squeeze(-1)
        T_fea = []
        for i in range(len(self.mlp)):
            fea = self.mlp[i](fea)
            if i==2:
                T_fea.append(fea)

        return fea,T_fea


class BlindSR(nn.Module):
    def __init__(self, args):
        super(BlindSR, self).__init__()

        # Generator
        self.G = RDAN(args)

        self.E = DEN(args)


    def forward(self, x, deg_repre):
        if self.training:

            # degradation-aware represenetion learning
            deg_repre, T_fea = self.E(deg_repre)

            # degradation-aware SR
            sr = self.G(x, deg_repre)

            return sr, T_fea
        else:
            # degradation-aware represenetion learning
            deg_repre, _ = self.E(deg_repre)

            # degradation-aware SR
            sr = self.G(x, deg_repre)

            return sr
