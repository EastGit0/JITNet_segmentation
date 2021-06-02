from base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from utils.metrics import eval_metrics

class basic_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, upsample=1,
                 bn_eps=1e-5,
                 bn_momentum=0.003,
                 bn_track_running_stats=True):
        super(basic_block, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.bn = norm_layer(in_channels, eps=bn_eps, momentum=bn_momentum,
                             track_running_stats=bn_track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels,
                                 kernel_size=1, stride=stride)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1)
        self.conv1x3 = nn.Conv2d(out_channels, out_channels,
                                 kernel_size=[1, 3], stride=1, padding=[0, 1])
        self.conv3x1 = nn.Conv2d(out_channels, out_channels,
                                 kernel_size=[3, 1], stride=1, padding=[1, 0])
        self.upsample = None
        if upsample > 1:
            self.upsample = nn.Upsample(scale_factor=upsample,
                                        mode='bilinear',
                                        align_corners=True)

    def forward(self, x):
        preact = self.relu(self.bn(x))
        shortcut = self.conv1x1(preact)
        residual = self.conv3x3(preact)
        residual = self.conv1x3(residual)
        residual = self.conv3x1(residual)
        out = shortcut + residual
        if self.upsample:
            out = self.upsample(out)
        return out


class JITNetLight(BaseModel):
    def __init__(self,
                 num_classes,
                 encoder_channels=[8, 32, 64],
                 encoder_strides=[2, 2, 2],
                 decoder_channels=[32, 32, 32],
                 decoder_strides=[1, 1, 1],
                 decoder_upsamples=[4, 1, 2],
                 bn_eps=1e-5,
                 bn_momentum=0.003,
                 bn_track_running_stats=True,
                 **_):
        super(JITNetLight, self).__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, encoder_channels[0], 3, 2, 1),
            nn.BatchNorm2d(encoder_channels[0], eps=bn_eps, momentum=bn_momentum,
                           track_running_stats=bn_track_running_stats),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(encoder_channels[0], encoder_channels[1], 3, 2, 1),
            nn.BatchNorm2d(encoder_channels[1], eps=bn_eps, momentum=bn_momentum,
                           track_running_stats=bn_track_running_stats),
            nn.ReLU(inplace=True)
        )

        self.enc_blocks = []
        for i in range(2, len(encoder_channels)):
            self.enc_blocks.append(basic_block(encoder_channels[i - 1],
                                               encoder_channels[i],
                                               encoder_strides[i],
                                               bn_eps=bn_eps,
                                               bn_momentum=bn_momentum,
                                               bn_track_running_stats=bn_track_running_stats))
        self.enc_blocks = nn.ModuleList(self.enc_blocks)
        self.dec_blocks = []
        prev_c = encoder_channels[-1]
        for i in range(0, len(decoder_channels) - 2):
            self.dec_blocks.append(basic_block(prev_c,
                                               decoder_channels[i],
                                               decoder_strides[i],
                                               decoder_upsamples[i],
                                               bn_eps=bn_eps,
                                               bn_momentum=bn_momentum,
                                               bn_track_running_stats=bn_track_running_stats))
            prev_c = decoder_channels[i] + encoder_channels[-i - 2]
        self.dec_blocks = nn.ModuleList(self.dec_blocks)

        self.dec1 = nn.Sequential(
            nn.Conv2d(decoder_channels[-3], decoder_channels[-2], 3, 1, 1),
            nn.BatchNorm2d(decoder_channels[-2], eps=bn_eps, momentum=bn_momentum,
                           track_running_stats=bn_track_running_stats),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(decoder_channels[-2], decoder_channels[-1], 3, 1, 1),
            nn.BatchNorm2d(decoder_channels[-1], eps=bn_eps, momentum=bn_momentum,
                           track_running_stats=bn_track_running_stats),
            nn.ReLU(inplace=True)
        )

        self.dec_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final = nn.Conv2d(decoder_channels[-1], num_classes, 1, 1)

        self._initialize_weights()
        #if freeze_bn: self.freeze_bn(

    def forward(self, x, labels=None, return_output=False, return_intermediate=False):
        x = self.enc1(x)
        x = self.enc2(x)
        down_x = []
        for b in self.enc_blocks:
            x = b(x)
            down_x.append(x)
        intermediate = down_x[-1] if return_intermediate else None
        for i, b in enumerate(self.dec_blocks):
            x = b(x)
            if i < len(self.dec_blocks) - 1:
                dx = down_x[-i - 2]
                x = torch.cat([dx, x[:, :, :dx.shape[2], :dx.shape[3]]], dim=1)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec_upsample(x)
        x = self.final(x)

        output = x
        if not hasattr(self, 'loss'):
            return output, intermediate

        loss = self.loss(output, labels)
        seg_metrics = eval_metrics(output, labels, output.shape[1])

        if not return_output:
            return loss, seg_metrics
        else:
            return loss, output, seg_metrics

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def get_backbone_params(self):
        # There is no backbone for unet, all the parameters are trained from scratch
        return []

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()

    @staticmethod
    def _set_requires_grad(m, requires_grad):
        for p in m.parameters():
            p.requires_grad = requires_grad

    def freeze_enc(self):
        self.enc1.apply(lambda m: JITNetLight._set_requires_grad(m, False))
        self.enc2.apply(lambda m: JITNetLight._set_requires_grad(m, False))
        self.enc_blocks.apply(lambda m: JITNetLight._set_requires_grad(m, False))

    def freeze_dec(self):
        self.dec_blocks.apply(lambda m: JITNetLight._set_requires_grad(m, False))
        self.dec1.apply(lambda m: JITNetLight._set_requires_grad(m, False))
        #self.dec2.apply(lambda m: JITNetLight._set_requires_grad(m, False))


if __name__ == "__main__":
    model = JITNetLight(81)
    # print(model)

    x = torch.rand(size=[8, 3, 480, 480])
    y = model(x)
    print(y.shape)
