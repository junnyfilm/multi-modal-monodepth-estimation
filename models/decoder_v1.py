import torch
import torch.nn as nn

from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init)
from utils.util import normalize_rot_vector

class Regression(nn.Module):
    def __init__(self, in_c, out_c):
        super(Regression, self).__init__()
        self.reg_layer = nn.Sequential(nn.Linear(in_features=in_c, out_features=int(in_c/2), bias=True),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(p=0.5, inplace=False),
                                       nn.Linear(in_features=int(in_c/2), out_features=int(in_c/4), bias=True),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(p=0.5, inplace=False),
                                       nn.Linear(in_features=int(in_c/4), out_features=out_c, bias=True))
    def forward(self, x):
        return self.reg_layer(x)

class Decoder_Pose(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.pos_layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True))
        self.pos_layer_down1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True))
        self.pos_layer_down2 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.rotat_reg_layer = Regression(self.in_channels, 9)
        self.trans_reg_layer = Regression(self.in_channels, 3)

    def forward(self, feats):
        # Pose Estimation
        #                                                 <feature_size>
        #                                                 swin scale32 : 1024
        #                                                 swin scale16 : 512
        #                                                 cnn_transformer resnet50 : 512
        #                                                 cnn_transformer resnet18 : 256
        out_p = self.pos_layers(feats)                  # [bs, feature_size * 2, H/model_scale, W/model_scale]          ->  [bs, feature_size * 2, H/model_scale, W/model_scale]
        out_p = self.pos_layer_down1(out_p)             # [bs, feature_size * 2, H/model_scale, W/model_scale]          ->  [bs, feature_size * 2, H/(model_scale*2), W/(model_scale*2)]
        out_p = self.pos_layer_down2(out_p)             # [bs, feature_size * 2, H/(model_scale*2), W/(model_scale*2)]  ->  [bs, feature_size * 2, H/(model_scale*4), W/(model_scale*4)]

        out_p = self.avg_pool(out_p)                    # [bs, feature_size * 2,  H/(model_scale*4), W/(model_scale*4)]  ->  [bs, feature_size * 2, 1, 1]
        out_p = out_p.flatten(1)                        # [bs, feature_size * 2, 1, 1]                                   ->  [bs, feature_size * 2]
        out_r = self.rotat_reg_layer(out_p)             # [bs, feature_size * 2]                                         ->  [bs, 9]
        out_t = self.trans_reg_layer(out_p)             # [bs, feature_size * 2]                                         ->  [bs, 3]

        out_r = normalize_rot_vector(out_r)
        return out_r, out_t
    
    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)

class Decoder_Depth(nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super().__init__()
        self.max_depth = args.max_depth
        self.deconv = args.num_deconv
        self.in_channels = in_channels
        self.num_upscale_layer = args.num_upscale_layer

        self.deconv_layers = self._make_deconv_layer(args.num_deconv,       
                                                     args.num_filters,      
                                                     args.deconv_kernels)   

        self.conv_layers = nn.Sequential(nn.Conv2d(args.num_filters[-1], out_channels, kernel_size=3, stride=1, padding=1),
                                         nn.BatchNorm2d(out_channels),
                                         nn.ReLU(inplace=True))

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.last_layer = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, 2, kernel_size=3, stride=1, padding=1))
        
        for m in self.last_layer.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)

    def forward(self, feats):
        # Depth Estimation
        out = self.deconv_layers(feats)             # [bs, feature_size * 2, H/model_scale, W/model_scale]                                      -> [bs, num_filters[-1], (H/model_scale) * (2^num_deconv), (W/model_scale) * (2^num_deconv)]
        out = self.conv_layers(out)                 # [bs, num_filters[-1], (H/model_scale) * (2^num_deconv), (W/model_scale) * (2^num_deconv)] -> [bs, embed_dim * 2, (H/model_scale) * (2^num_deconv), (W/model_scale) * (2^num_deconv)]

        for _ in range(self.num_upscale_layer):
            out = self.up(out)                      # [bs, embed_dim * 2, (H/model_scale) * (2^num_deconv), (W/model_scale) * (2^num_deconv)]   -> [bs, embed_dim * 2, ((H/model_scale)*(2^num_deconv)) * 2, ((W/model_scale)*(2^num_deconv)) * 2]
        out = self.last_layer(out)                    # [bs, 2, H, W]
        out = torch.sigmoid(out) * self.max_depth

        out_d1, out_d2 = out.chunk(2, dim=1)
        return out_d1, out_d2

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""

        layers = []
        in_planes = self.in_channels
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=in_planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            in_planes = planes

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)

class Decoder_v1(nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super().__init__()

        self.decoder_pose = Decoder_Pose(in_channels * 2)
        self.decoder_depth = Decoder_Depth(in_channels * 2, out_channels, args)

    def forward(self, feat1, feat2):
        feats = torch.cat([feat1, feat2], dim=1)        # [bs, (model_scale * 32) * 2, H/model_scale, W/model_scale]
        out_r1, out_t1 = self.decoder_pose(feats)
        out_d1, out_d2 = self.decoder_depth(feats)

        return out_d1, out_r1, out_t1, out_d2, None, None 

    def init_weights(self):
        """Initialize model weights."""
        self.decoder_pose.init_weights()
        self.decoder_depth.init_weights()