import torch
from torch import nn
import model_meta.common as common
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


    def forward(self, lr,lr_bic,weights,base=''):
        #************************head********************
        res = common.conv2d(lr, weights[base + 'head.0.weight'], weights[base + 'head.0.bias'], stride=1, padding=1)
        res = F.leaky_relu(res, 0.1, True)
        #************************body********************
        res = common.conv2d(res, weights[base + 'body.0.weight'], weights[base + 'body.0.bias'], stride=1, padding=1)
        res = F.leaky_relu(res, 0.1, True)
        res = common.conv2d(res, weights[base + 'body.2.weight'], weights[base + 'body.2.bias'], stride=1, padding=1)
        res = F.leaky_relu(res, 0.1, True)
        res = common.conv2d(res, weights[base + 'body.4.weight'], weights[base + 'body.4.bias'], stride=1, padding=1)
        res = F.leaky_relu(res, 0.1, True)
        res = common.conv2d(res, weights[base + 'body.6.weight'], weights[base + 'body.6.bias'], stride=1, padding=1)
        res = F.leaky_relu(res, 0.1, True)
        res = common.conv2d(res, weights[base + 'body.8.weight'], weights[base + 'body.8.bias'], stride=1, padding=1)
        res = F.leaky_relu(res, 0.1, True)
        res = common.conv2d(res, weights[base + 'body.10.weight'], weights[base + 'body.10.bias'], stride=1, padding=1)
        res = F.leaky_relu(res, 0.1, True)
        #**********************tailx4***********************
        res = common.conv2d(res, weights[base +"tail.0.0.weight"], weights[base +"tail.0.0.bias"], stride=1, padding=1)
        res = F.pixel_shuffle(res, 2)
        res = common.conv2d(res, weights[base +"tail.0.2.weight"], weights[base +"tail.0.2.bias"], stride=1, padding=1)
        res = F.pixel_shuffle(res, 2)
        res = common.conv2d(res, weights[base + "tail.1.weight"], weights[base + "tail.1.bias"], stride=1, padding=1)
        # res = self.tail(res)
        res +=lr_bic

        return res




