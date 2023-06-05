import torch
from torch import nn
import model.common as common
import torch.nn.functional as F


def make_model(args):
    return MLN(args)


class MLN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(MLN, self).__init__()

        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.LeakyReLU(0.1, True)
        self.head = nn.Sequential(
            nn.Conv2d(3, n_feats, kernel_size=3, padding=1),
                                  act
        )
        m_body = [
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            act,
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            act,
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            act,
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            act,
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            act,
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            act
        ]
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            nn.Conv2d(
                n_feats, args.n_colors, kernel_size,
                padding=(kernel_size // 2)
            )
        ]
        self.tail = nn.Sequential(*m_tail)

        self.body = nn.Sequential(*m_body)

        self.ad = nn.AdaptiveAvgPool2d(1)


    def forward(self, lr,lr_bic):
        res = self.head(lr)
        res = self.body(res)
        deg_repre = res
        res = self.tail[0](res)
        res = self.tail[1](res)
        res +=lr_bic


        return res, deg_repre





