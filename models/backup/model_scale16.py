import torch
import torch.nn as nn

from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init)
from models.swin_transformer_v2 import SwinTransformerV2
from models.cnn_transformer import cnn_transformer

class GLPDepth_scale16(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.max_depth = args.max_depth

        if 'tiny' in args.backbone:
            embed_dim = 96
            num_heads = [3, 6, 12, 24]
        elif 'base' in args.backbone:
            embed_dim = 128
            num_heads = [4, 8, 16, 32]
        elif 'large' in args.backbone:
            embed_dim = 192
            num_heads = [6, 12, 24, 48]
        elif 'huge' in args.backbone:
            embed_dim = 352
            num_heads = [11, 22, 44, 88]
        elif 'cnn_transformer' in args.backbone:
            embed_dim = 128
            hidden_dim = 512
        else:
            raise ValueError(args.backbone + " is not implemented, please add it in the models/model.py.")

        self.embed_dim = embed_dim

        if args.backbone == 'swin_base_v2':
            self.encoder = SwinTransformerV2(
                embed_dim=embed_dim,
                depths=args.depths[:-1],
                num_heads=num_heads[:-1],
                window_size=args.window_size[:-1],
                pretrain_window_size=args.pretrain_window_size[:-1],
                drop_path_rate=args.drop_path_rate,
                use_checkpoint=args.use_checkpoint,
                use_shift=args.use_shift[:-1],
                out_indices=(len(args.depths[:-1])-1,),
            )

            self.encoder.init_weights(pretrained=args.pretrained)
        elif args.backbone == 'cnn_transformer_multi_scale':
            self.encoder = cnn_transformer(args, hidden_dim=hidden_dim, n_enc_layers=6, resnet_multi_scale=True)
        elif args.backbone == 'cnn_transformer':
            self.encoder = cnn_transformer(args, hidden_dim=hidden_dim, n_enc_layers=6, resnet_multi_scale=False)

        channels_in = embed_dim * 4
        channels_out = embed_dim

        self.decoder = Decoder(channels_in, channels_out*2, args)  # Decoder(channels_in, channels_out, args)
        self.decoder.init_weights()

        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(channels_out*2, channels_out*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out*2, 2, kernel_size=3, stride=1, padding=1))

        for m in self.last_layer_depth.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)

    def forward(self, frame1, frame2):                          # [bs, 3, 480, 480]
        frames = torch.cat([frame1, frame2])
        conv_feats = self.encoder(frames)                       # [2*bs, 512, 30, 30]

        out_d, out_p = self.decoder(conv_feats)                 # out_d: [bs, 256, 480, 480]    out_p: [bs, 12]
        out_depth = self.last_layer_depth(out_d)                # [bs, 2, 480, 480]
        out_depth = torch.sigmoid(out_depth) * self.max_depth

        pred_d1, pred_d2 = out_depth.chunk(2, dim=1)
        return {'pred_d1': pred_d1, 'pred_d2': pred_d2, 'out_p': out_p}


class Regression(nn.Module):
    def __init__(self, in_c, out_c):
        super(Regression, self).__init__()
        self.reg_layer = nn.Sequential(nn.Linear(in_features=in_c, out_features=512, bias=True),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(p=0.5, inplace=False),
                                       nn.Linear(in_features=512, out_features=512, bias=True),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(p=0.5, inplace=False),
                                       nn.Linear(in_features=512, out_features=out_c, bias=True))

    def forward(self, x):
        return self.reg_layer(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super().__init__()
        self.deconv = args.num_deconv
        self.in_channels = in_channels * 2

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

        self.deconv_layers = self._make_deconv_layer(args.num_deconv,        # 3
                                                     args.num_filters,       # [32, 32, 32]
                                                     args.deconv_kernels)    # [2, 2, 2]

        self.conv_layers = nn.Sequential(nn.Conv2d(args.num_filters[-1], out_channels, kernel_size=3, stride=1, padding=1),
                                         nn.BatchNorm2d(out_channels),
                                         nn.ReLU(inplace=True))

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, conv_feats):
        conv_feats1, conv_feats2 = conv_feats[0].chunk(2, dim=0)    # [bs, 512, 30, 30] * 2

        feats = torch.cat([conv_feats1, conv_feats2], dim=1)        # [bs, 512*2, 30, 30]

        # Pose Estimation
        out_p = self.pos_layers(feats)          # [bs, 1024, 30, 30]  ->  [bs, 1024, 30, 30]
        out_p = self.pos_layer_down1(out_p)     # [bs, 1024, 30, 30]  ->  [bs, 1024, 15, 15]
        out_p = self.pos_layer_down2(out_p)     # [bs, 1024, 15, 15]  ->  [bs, 1024, 8, 8]

        out_p = self.avg_pool(out_p)            # [bs, 1024, 8, 8]  ->  [bs, 1024, 1, 1]
        out_p = out_p.flatten(1)                # [bs, 1024, 1, 1]  ->  [bs, 1024]
        out_p_r = self.rotat_reg_layer(out_p)   # [bs, 1024]        ->  [bs, 9]
        out_p_t = self.trans_reg_layer(out_p)   # [bs, 1024]        ->  [bs, 3]
        out_p = torch.cat([out_p_r, out_p_t], dim=-1)   # [bs, 12]

        # Depth Estimation
        out_d = self.deconv_layers(feats)       # [bs, 1024, 30, 30]  -> [bs, 32, 240, 240]
        out_d = self.conv_layers(out_d)         # [bs, 32, 240, 240] -> [bs, 256, 240, 240]
    
        out_d = self.up(out_d)                  # [bs, 256, 240, 240] -> [bs, 256, 480, 480]

        return out_d, out_p

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

