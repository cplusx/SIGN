"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import numpy as np
from collections import OrderedDict
from numpy.core.fromnumeric import nonzero
import torch
from torch import nn
from torch import autograd
import torch.nn.functional as F
from torch.autograd import Variable
from resnet import _ConvBatchNormReLU, _ResBlock
from blocks import LinearBlock, Conv2dBlock, ResBlocks, ActFirstResBlock, _ASPPModule, _ConvReLU_, positionalencoding2d, _SelfAttention_, _MultiHeadedAttention_

class MSCC(nn.Module):
    """Multi-scale inputs"""

    def __init__(self, scale, pyramids=[0.5, 0.75]):
        super(MSCC, self).__init__()
        self.scale = scale
        self.pyramids = pyramids

    def forward(self, x, mask):
        KLD, h0, h1 = self.scale(x, mask)
        logits_h0 = self.get_resized_logits(x, mask, h0, 1)
 
        return KLD, logits_h0, h0, h1

    def get_resized_logits(self, x, mask, logits, scale_return_index):
        # Original
        interp = lambda l: F.interpolate(l, size=logits.shape[2:], mode="bilinear", align_corners=False)

        # Scaled
        logits_pyramid = []
        for p in self.pyramids:
            size = [int(s * p) for s in x.shape[2:]]
            h = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
            logits_pyramid.append(self.scale(h, mask)[scale_return_index])

        # Pixel-wise max
        logits_all = [logits] + [interp(l) for l in logits_pyramid]
        logits_max = torch.max(torch.stack(logits_all), dim=0)[0]

        if self.training:
            return [logits] + logits_pyramid + [logits_max]
        else:
            return logits_max

    def freeze_bn(self):
        self.scale.freeze_bn()


class DeepLabV2_local(nn.Sequential):
    """DeepLab v2"""

    def __init__(self, n_classes, n_blocks, pyramids, freeze_bn, hp):
        super(DeepLabV2_local, self).__init__()
        self.hp = hp # config file

        self.add_module(
            "layer1",
            nn.Sequential(
                OrderedDict(
                    [
                        ("conv1", _ConvBatchNormReLU(3, 64, 7, 2, 3, 1)),
                        ("pool", nn.MaxPool2d(3, 2, 1, ceil_mode=True)),
                    ]
                )
            )
        )
        self.add_module("layer2", _ResBlock(n_blocks[0], 64, 64, 256, 1, 1))
        self.add_module("layer3", _ResBlock(n_blocks[1], 256, 128, 512, 2, 1))
        self.add_module("layer4", _ResBlock(n_blocks[2], 512, 256, 1024, 1, 2))
        self.add_module("layer5", _ResBlock(n_blocks[3], 1024, 512, 2048, 1, 4))
        self.add_module("aspp", _ASPPModule(2048, n_classes, pyramids))

        self.add_module("contextual1", _ConvReLU_(n_classes, 256, 3, 1, 1, 1))
        self.add_module("contextual2", _ConvReLU_(256, 256, 3, 1, 2, 2))
        self.add_module("contextual3", _ConvReLU_(256, 256, 3, 1, 5, 5))
        self.add_module("fc_1", _ConvReLU_(3*256, 3, 3, 1, 1, 1))
        self.add_module("contextualpool", _ConvReLU_(3*256, n_classes, 1, 1, 0, 1))
        self.add_module("contextuallocalmu", _ConvReLU_(n_classes, n_classes, 3, 1, 1, 1, relu=False))
        self.add_module("contextuallocalsigma",_ConvReLU_(n_classes, n_classes, 3, 1, 1, 1))

        if 'network_type' not in hp:
            self.add_network_type1(n_classes)
            self.hp['network_type'] = 1
        elif hp['network_type'] == 1:
            self.add_network_type1(n_classes)
        elif hp['network_type'] == 2:
            self.add_network_type2(n_classes)
        elif hp['network_type'] == 3:
            self.add_network_type3(n_classes)
        elif hp['network_type'] == 4:
            self.add_network_type4(n_classes)
        elif hp['network_type'] == 5:
            self.add_network_type5(n_classes)
        elif hp['network_type'] == 6:
            self.add_network_type6(n_classes)
        elif hp['network_type'] == 7:
            self.add_network_type5(n_classes)
        elif hp['network_type'] == 8:
            self.add_network_type5(n_classes)
        elif hp['network_type'] == 2.1:
            self.add_network_type2_1(n_classes)
        else:
            raise NotImplementedError

        if freeze_bn:
            self.freeze_bn()


    # For ablation studies
    def add_network_type1(self, n_classes):
        self.add_module("positional1",_ConvReLU_(2*n_classes, 256, 3, 1, 1, 1))
        self.add_module("positional2",_ConvReLU_(256, 256, 3, 1, 2, 2))
        self.add_module("positional3",_ConvReLU_(256, 256, 3, 1, 5, 5))
        self.add_module("fc_2", _ConvReLU_(3*256, 3, 3, 1, 1, 1))
        self.add_module("positionalpool", _ConvReLU_(3*256, n_classes, 1, 1, 0, 1))
        # self.add_module("positionallocalmu", _ConvReLU_(128, n_classes, 3, 1, 1, 1, relu=False))
        # self.add_module("positionallocalsigma",_ConvReLU_(128, n_classes, 3, 1, 1, 1))

    def add_network_type1_1(self, n_classes):
        self.add_module("positional1",_ConvReLU_(2*n_classes, 256, 3, 1, 1, 1))
        self.add_module("positional2",_ConvReLU_(256, 256, 3, 1, 2, 2))
        self.add_module("positional3",_ConvReLU_(256, 256, 3, 1, 5, 5))
        self.add_module("fc_2", _ConvReLU_(3*256, 3, 3, 1, 1, 1))
        self.add_module("positionalpool", _ConvReLU_(3*256, n_classes, 1, 1, 0, 1))
        self.add_module("positionallocalmu", _ConvReLU_(128, n_classes, 3, 1, 1, 1, relu=False))
        self.add_module("positionallocalsigma",_ConvReLU_(128, n_classes, 3, 1, 1, 1))

    def add_network_type2(self, n_classes):
        self.add_module("positional1",_ConvReLU_(2*n_classes, 256, 3, 1, 1, 1))
        self.add_module("positional2",_ConvReLU_(256, 256, 3, 1, 2, 2))
        self.add_module("positional3",_ConvReLU_(256, 256, 3, 1, 5, 5))
        self.add_module("fc_2", _ConvReLU_(3*256, 3, 3, 1, 1, 1))
        self.add_module("positionalpool", _ConvReLU_(3*256, 128, 1, 1, 0, 1))
        self.add_module("positionallocalmu", _ConvReLU_(128, n_classes, 3, 1, 1, 1, relu=False))
        self.add_module("positionallocalsigma",_ConvReLU_(128, n_classes, 3, 1, 1, 1))

    def add_network_type3(self, n_classes):
        bottleneck_dim = 64
        self.add_module("positional_feedforward0", _ConvReLU_(2*n_classes, bottleneck_dim, 1, 1, 0, 1))  # NOTE fix padding on march 3rd
        self.add_module("positional_self_att1", _SelfAttention_(bottleneck_dim))
        self.add_module("positional_feedforward1", _ConvReLU_(bottleneck_dim, bottleneck_dim, 1, 1, 0, 1))
        self.add_module("positional_self_att2", _SelfAttention_(bottleneck_dim))
        self.add_module("positional_feedforward2", _ConvReLU_(bottleneck_dim, n_classes, 1, 1, 0, 1))

    def add_network_type2_1(self, n_classes):
        self.add_module("positional1",_ConvReLU_(2*n_classes, 256, 3, 1, 1, 1))
        self.add_module("positional2",_ConvReLU_(256, 256, 3, 1, 2, 2))
        self.add_module("positional3",_ConvReLU_(256, 256, 3, 1, 5, 5))
        self.add_module("fc_2", _ConvReLU_(3*256, 3, 3, 1, 1, 1))
        self.add_module("positionalpool", _ConvReLU_(3*256, n_classes, 1, 1, 0, 1))
        self.add_module("positionallocalmu", _ConvReLU_(n_classes, n_classes, 3, 1, 1, 1, relu=False))
        self.add_module("positionallocalsigma",_ConvReLU_(n_classes, n_classes, 3, 1, 1, 1))

    def add_network_type4(self, n_classes):
        bottleneck_dim = 512
        num_heads = 8
        model_dim = bottleneck_dim
        self.add_module("positional_feedforward0", _ConvReLU_(2*n_classes, model_dim, 1, 1, 0, 1)) # NOTE fix padding on march 3rd
        self.add_module("positional_self_att1", nn.MultiheadAttention(model_dim, num_heads, dropout=0.1))
        self.add_module("positional_feedforward1", _ConvReLU_(bottleneck_dim, bottleneck_dim, 1, 1, 0, 1))
        self.add_module("positional_self_att2", nn.MultiheadAttention(model_dim, num_heads, dropout=0.1))
        self.add_module("positional_feedforward2", _ConvReLU_(bottleneck_dim, n_classes, 1, 1, 0, 1))

    def add_network_type5(self, n_classes):
        bottleneck_dim = 128
        self.add_module("positional_conv1", _ConvReLU_(2*n_classes, n_classes, 1, 1, 0, 1))
        self.add_module("positional_feedforward0", _ConvReLU_(n_classes, bottleneck_dim, 1, 1, 0, 1))
        self.add_module("positional_self_att1", _SelfAttention_(bottleneck_dim))
        self.add_module("positional_feedforward1", _ConvReLU_(bottleneck_dim, bottleneck_dim, 1, 1, 0, 1))
        self.add_module("positional_self_att2", _SelfAttention_(bottleneck_dim))
        self.add_module("positional_feedforward2", _ConvReLU_(bottleneck_dim, n_classes, 1, 1, 0, 1))

    def add_network_type6(self, n_classes):
        bottleneck_dim = 512
        num_heads = 8
        model_dim = bottleneck_dim
        self.add_module("positional_conv1", _ConvReLU_(2*n_classes, n_classes, 1, 1, 0, 1))
        self.add_module("positional_feedforward0", _ConvReLU_(n_classes, model_dim, 1, 1, 0, 1))
        self.add_module("positional_self_att1", nn.MultiheadAttention(model_dim, num_heads, dropout=0.1))
        self.add_module("positional_feedforward1", _ConvReLU_(bottleneck_dim, bottleneck_dim, 1, 1, 0, 1))
        self.add_module("positional_self_att2", nn.MultiheadAttention(model_dim, num_heads, dropout=0.1))
        self.add_module("positional_feedforward2", _ConvReLU_(bottleneck_dim, n_classes, 1, 1, 0, 1))


    def forward(self, x, mask):
        bs, dim, height, width = x.shape
        self.non_border_regions = torch.where(x.sum(1) == 0., torch.zeros((bs, height, width), device=x.device), torch.ones((bs, height, width), device=x.device)) # only add PE on non border region, this is also mask in attention
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.layer5(h)
        h = self.aspp(h)

        _, h0, _= self.positional_encoding(h, mask)
        h1, h0, KLD = self.contextual_encoding(mask, h0)

        h0 = torch.sigmoid(h0)
        return KLD, h0, h1 # kl loss, feature, contextual

    def contextual_encoding(self, mask, h):
        h1 = self.contextual1(torch.sigmoid(h))
        h2 = self.contextual2(h1)
        h3 = self.contextual3(h2)
        f1 = torch.sigmoid(self.fc_1(torch.cat([h1, h2, h3], dim=1)))
        h1 = h1 * f1[:,0,:,:].unsqueeze(1)
        h2 = h2 * f1[:,1,:,:].unsqueeze(1)
        h3 = h3 * f1[:,2,:,:].unsqueeze(1)
        h1 = self.contextualpool(torch.cat([h1, h2, h3], dim=1))
        localmu = F.interpolate(self.contextuallocalmu(h1), size=h.size()[2:], mode="bilinear")
        localsigma = F.interpolate(self.contextuallocalsigma(h1), size=h.size()[2:], mode="bilinear")
        h1 = F.interpolate(self.reparameterize(localmu, localsigma), size=h.size()[2:], mode="bilinear")
        # contextual latent code      
        att = torch.sigmoid(h1)
        h_att = torch.mul(h, att)
        h0 = h + h_att
        # augmented feature,        # KL-Div loss
        localmask = F.interpolate(mask.float().unsqueeze(1).repeat(1, localmu.size(1), 1, 1), size=h.size()[2:], mode="nearest").bool()
        KLD = -0.5 * torch.sum((1 + localsigma - localmu.pow(2) - localsigma.exp())[localmask.bool()])/localmask.float().sum() if localmask.float().sum()>0 else localmask.float().mean()
        return h1, h0, KLD # contextual latent code, feature, KLD loss

    def positional_encoding(self, h, mask):
        if self.hp['network_type'] == 1:
            return self.positional_encoding_v1(h, mask, mode=3)
        if self.hp['network_type'] == 2 or self.hp['network_type'] == 2.1:
            return self.positional_encoding_v1(h, mask, mode=1)
        if self.hp['network_type'] == 3:
            return self.positional_encoding_v3(h)
        if self.hp['network_type'] == 4:
            return self.positional_encoding_v4(h)
        if self.hp['network_type'] == 5:
            return self.positional_encoding_v5(h)
        if self.hp['network_type'] == 6:
            return self.positional_encoding_v6(h)
        if self.hp['network_type'] == 7:
            return self.positional_encoding_v7(h)
        if self.hp['network_type'] == 8:
            return self.positional_encoding_v8(h)
        else:
            raise NotImplementedError

    def positional_encoding_v1(self, h, mask, mode):
        '''
        mode: 1 conv -> reparameter -> attention -> residual
        mode: 2 conv -> reparameter -> residual
        mode: 3 conv -> residual
        In theory, positional encoding already includes contextual encoding
        '''
        pe = positionalencoding2d(self.hp['n_classes'], height=h.shape[2], width=h.shape[3]).cuda().unsqueeze(0).repeat(h.shape[0], 1, 1, 1)
        h_pe = torch.cat([torch.sigmoid(h), pe], dim=1)
        h1 = self.positional1(h_pe)
        h2 = self.positional2(h1)
        h3 = self.positional3(h2)
        f1 = torch.sigmoid(self.fc_2(torch.cat([h1, h2, h3], dim=1)))
        h1 = h1 * f1[:,0,:,:].unsqueeze(1)
        h2 = h2 * f1[:,1,:,:].unsqueeze(1)
        h3 = h3 * f1[:,2,:,:].unsqueeze(1)
        h1 = self.positionalpool(torch.cat([h1, h2, h3], dim=1))

        # reparameter
        if mode == 1 or mode == 2:
            localmu = F.interpolate(self.positionallocalmu(h1), size=h.size()[2:], mode="bilinear")
            localsigma = F.interpolate(self.positionallocalsigma(h1), size=h.size()[2:], mode="bilinear")
            h1 = F.interpolate(self.reparameterize(localmu, localsigma), size=h.size()[2:], mode="bilinear")
            localmask = F.interpolate(mask.float().unsqueeze(1).repeat(1, localmu.size(1), 1, 1), size=h.size()[2:], mode="nearest").bool()
            KLD = -0.5 * torch.sum((1 + localsigma - localmu.pow(2) - localsigma.exp())[localmask.bool()])/localmask.float().sum() if localmask.float().sum()>0 else localmask.float().mean()
        else:
            KLD = torch.from_numpy(np.array(0.)).to(h.device)

        # attention
        if mode == 1:
            att = torch.sigmoid(h1)
            h_att = torch.mul(h, att)
        elif mode == 2 or mode == 3:
            h_att = h1
        else:
            raise NotImplementedError('Unrecognized mode {}'.format(mode))

        # residual
        h0 = h + h_att

        # when there is no reparameter, set noise to normal distribution
        if mode == 3:
            h1 = torch.randn(h1.shape, device=h1.device)

        return h1, h0, KLD

    def positional_encoding_v3(self, h):
        pe = positionalencoding2d(self.hp['n_classes'], height=h.shape[2], width=h.shape[3]).cuda().unsqueeze(0).repeat(h.shape[0], 1, 1, 1)
        h = torch.cat([torch.sigmoid(h), pe], dim=1)
        h = self.positional_feedforward0(h)
        h = self.positional_self_att1(h)
        h = self.positional_feedforward1(h)
        h = self.positional_self_att2(h)
        h = self.positional_feedforward2(h)
        # h = self.positional_self_att3(h)
        # h = self.positional_feedforward3(h)
        return None, h, None

    def positional_encoding_v4(self, h):
        pe = positionalencoding2d(self.hp['n_classes'], height=h.shape[2], width=h.shape[3]).cuda().unsqueeze(0).repeat(h.shape[0], 1, 1, 1)
        h = torch.cat([torch.sigmoid(h), pe], dim=1)
        h = self.positional_feedforward0(h)
        bs, dim, im_h, im_w = h.shape
        h = h.view(bs, dim, im_h*im_w).permute(0, 2, 1)
        h = self.positional_self_att1(h, h, h)[0].permute(0, 2, 1).contiguous().view(bs, dim, im_h, im_w)
        h = self.positional_feedforward1(h)
        bs, dim, im_h, im_w = h.shape
        h = h.view(bs, dim, im_h*im_w).permute(0, 2, 1)
        h = self.positional_self_att2(h, h, h)[0].permute(0, 2, 1).contiguous().view(bs, dim, im_h, im_w)
        h = self.positional_feedforward2(h)
        return None, h, None

    def positional_encoding_v5(self, h):
        pe = positionalencoding2d(self.hp['n_classes'], height=h.shape[2], width=h.shape[3]).cuda().unsqueeze(0).repeat(h.shape[0], 1, 1, 1)
        h = torch.cat([torch.sigmoid(h), pe], dim=1)
        h_ = self.positional_conv1(h)
        h = self.positional_feedforward0(h_)
        h = self.positional_self_att1(h)
        h = self.positional_feedforward1(h)
        h = self.positional_self_att2(h)
        h = self.positional_feedforward2(h)
        return None, h + h_, None

    def positional_encoding_v6(self, h):
        pe = positionalencoding2d(self.hp['n_classes'], height=h.shape[2], width=h.shape[3]).cuda().unsqueeze(0).repeat(h.shape[0], 1, 1, 1)
        h = torch.cat([torch.sigmoid(h), pe], dim=1)
        h_ = self.positional_conv1(h)
        h = self.positional_feedforward0(h_)
        bs, dim, im_h, im_w = h.shape
        h = h.view(bs, dim, im_h*im_w).permute(0, 2, 1)
        h = self.positional_self_att1(h, h, h)[0].permute(0, 2, 1).contiguous().view(bs, dim, im_h, im_w)
        h = self.positional_feedforward1(h)
        bs, dim, im_h, im_w = h.shape
        h = h.view(bs, dim, im_h*im_w).permute(0, 2, 1)
        h = self.positional_self_att2(h, h, h)[0].permute(0, 2, 1).contiguous().view(bs, dim, im_h, im_w)
        h = self.positional_feedforward2(h)
        return None, h + h_, None

    def border_aware_positional_encoding2d(self, height, width):
        bs = self.non_border_regions.shape[0]
        pe = torch.zeros(bs, self.hp['n_classes'], height, width, device=self.non_border_regions.device)
        for idx, r in enumerate(self.non_border_regions):
            non_zeros = torch.nonzero(r)
            top = non_zeros[:, 0].min()
            bottom = non_zeros[:, 0].max()
            left = non_zeros[:, 1].min()
            right = non_zeros[:, 1].max()
            pe[idx][:, top: bottom+1, left: right+1] = positionalencoding2d(self.hp['n_classes'], height=bottom-top+1, width=right-left+1)
        return pe

    def positional_encoding_v7(self, h):
        pe = positionalencoding2d(self.hp['n_classes'], height=h.shape[2], width=h.shape[3]).cuda().unsqueeze(0).repeat(h.shape[0], 1, 1, 1)
        h = torch.cat([torch.sigmoid(h), pe], dim=1)
        h0 = self.positional_conv1(h)
        h1 = self.positional_feedforward0(h0)
        h2 = self.positional_self_att1(h1) + h1
        h3 = self.positional_feedforward1(h2) + h2
        h4 = self.positional_self_att2(h3) + h3
        h5 = self.positional_feedforward2(h4)
        return None, h5, None

    def positional_encoding_v8(self, h):
        self.non_border_regions = F.interpolate(self.non_border_regions.unsqueeze(1), size=(h.shape[2], h.shape[3]), mode='nearest').squeeze(1)
        pe = self.border_aware_positional_encoding2d(height=h.shape[2], width=h.shape[3])
        h = torch.cat([torch.sigmoid(h), pe], dim=1)
        h0 = self.positional_conv1(h)
        h1 = self.positional_feedforward0(h0)
        h2 = self.positional_self_att1(h1) + h1
        h3 = self.positional_feedforward1(h2) + h2
        h4 = self.positional_self_att2(h3) + h3
        h5 = self.positional_feedforward2(h4)
        return None, h5, None


    def freeze_bn(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.BatchNorm2d) and 'positional' not in name:
                m.eval()

    def reparameterize(self, mu, logvar):
        """
        THE REPARAMETERIZATION IDEA:
        """
        if self.training:
            std = logvar.mul(0.5).exp_()  # type: Variable
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            # During inference, we simply spit out the mean of the
            # learned distribution for the current input.  We could
            # use a random sample from the distribution, but mu of
            # course has the highest probability.
            return mu


def DeepLabV2_ResNet101_local(hp):
    return DeepLabV2_local(
        n_classes=hp['n_classes'], 
        n_blocks=[3, 4, 23, 3], 
        pyramids=[6, 12, 18, 24], 
        freeze_bn=True,
        hp=hp
    )

def DeepLabV2_ResNet101_local_MSC(hp):
    return MSCC(
        scale=DeepLabV2_ResNet101_local(hp), 
        pyramids=[0.5, 0.75]
    )


class Discriminator(nn.Module):
    def __init__(self, hp):
        super(Discriminator, self).__init__()
        self.fc = Conv2dBlock(in_dim=hp['in_dim_fc'], 
                              out_dim=hp['out_dim_fc'], 
                              ks=1, 
                              st=1, 
                              padding=0, 
                              norm=hp['norm_fc'], 
                              activation=hp['activ_fc'], 
                              dropout=hp['drop_fc'])

        self.pred = nn.Conv2d(hp['out_dim_fc'], 1, 1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.cls = nn.Conv2d(hp['out_dim_fc'], hp['out_dim_cls'], 1, stride=1, padding=0)

    def forward(self, feat, mode):
        pred_map = self.fc(feat)
        if mode == 'gan':
            gan_score = self.pred(pred_map)
            gan_score_sigmoid = self.sigmoid(gan_score)
            return gan_score_sigmoid
        elif mode == 'cls':
            cls_score = self.cls(pred_map)
            return cls_score
        else:
            raise NotImplementedError('Invalid mode {} for discriminator.' % mode)


class Generator(nn.Module):
    def __init__(self, hp):
        super(Generator, self).__init__()

        self.mlp = nn.Sequential(
            Conv2dBlock(
                in_dim=hp['in_dim_mlp'], 
                out_dim=1024, 
                ks=1, 
                st=1, 
                padding=0, 
                norm='none', 
                activation='lrelu', 
                dropout=0.1
            ),
            Conv2dBlock(
                in_dim=1024, 
                out_dim=960, 
                ks=1, 
                st=1, 
                padding=0, 
                norm='none', 
                activation='lrelu', 
                dropout=0.1
            ),
            Conv2dBlock(
                in_dim=960, 
                out_dim=864, 
                ks=1, 
                st=1, 
                padding=0, 
                norm='none', 
                activation='lrelu', 
                dropout=0.3
            ),
            Conv2dBlock(
                in_dim=864, 
                out_dim=784, 
                ks=1, 
                st=1, 
                padding=0, 
                norm='none', 
                activation='lrelu', 
                dropout=0.5
            ),
            Conv2dBlock(
                in_dim=784, 
                out_dim=720, 
                ks=1, 
                st=1, 
                padding=0, 
                norm='none', 
                activation='lrelu', 
                dropout=0.5
            ),
            nn.Conv2d(720, hp['out_dim_mlp'], 1, stride=1, padding=0),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, sample):
        feat = self.mlp(sample)
        feat_sigmoid = self.sigmoid(feat)
        return feat_sigmoid
