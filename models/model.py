import torch
import torch.nn as nn

from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init)
from models.swin_transformer_v2 import SwinTransformerV2
from models.cnn_transformer import cnn_transformer
from models.resnet_only import resnet_only
from models.decoder_v1 import Decoder_v1
from models.decoder_v2 import Decoder_v2


class IDEDepth(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        b_init_encoder = True
        if 'swin' in args.backbone:
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
            
            args.num_deconv = 3
            args.num_filters = [32, 32, 32]
            args.deconv_kernels = [2, 2, 2]

            if args.model_scale == 32:
                channels_in = embed_dim * 8
                channels_out = embed_dim
                args.num_upscale_layer = 2

                self.encoder = SwinTransformerV2(
                    embed_dim=embed_dim,
                    depths=args.depths,
                    num_heads=num_heads,
                    window_size=args.window_size,
                    pretrain_window_size=args.pretrain_window_size,
                    drop_path_rate=args.drop_path_rate,
                    use_checkpoint=args.use_checkpoint,
                    use_shift=args.use_shift,
                )
                self.encoder.init_weights(pretrained=args.pretrained)

            elif args.model_scale == 16:
                channels_in = embed_dim * 4
                channels_out = embed_dim
                args.num_upscale_layer = 1

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
            else:
                b_init_encoder = False

        elif 'cnn_transformer' in args.backbone or 'resnet_only' in args.backbone:
            if args.cnn_model == 'resnet50' or args.cnn_model == '50':
                embed_dim = 128
                hidden_dim = embed_dim * 4
                channels_in = embed_dim * 4
                channels_out = embed_dim
                args.num_deconv = 3
                args.num_filters = [32, 32, 32]
                args.deconv_kernels = [2, 2, 2]
                args.num_upscale_layer = 1
            elif args.cnn_model == 'resnet18' or args.cnn_model == '18':
                embed_dim = 128
                hidden_dim = embed_dim * 2
                channels_in = embed_dim * 2
                channels_out = embed_dim
                args.num_deconv = 2
                args.num_filters = [32, 32]
                args.deconv_kernels = [2, 2]
                args.num_upscale_layer = 2
            else:
                b_init_encoder = False
            
            if args.backbone == 'cnn_transformer_multi_scale':
                self.encoder = cnn_transformer(args, hidden_dim=hidden_dim, n_enc_layers=6, resnet_multi_scale=True)
            elif args.backbone == 'cnn_transformer':
                self.encoder = cnn_transformer(args, hidden_dim=hidden_dim, n_enc_layers=6, resnet_multi_scale=False)
            elif args.backbone == 'resnet_only_multi_scale':
                self.encoder = resnet_only(args, hidden_dim=hidden_dim, resnet_multi_scale=True)
            elif args.backbone == 'resnet_only':
                self.encoder = resnet_only(args, hidden_dim=hidden_dim, resnet_multi_scale=False)
            else:
                b_init_encoder = False
        else:
            b_init_encoder = False
        if not b_init_encoder:
            raise ValueError(args.backbone + " is not implemented, please add it in the models/model.py.")

        if args.decoder == "decoder_v1":
            self.decoder = Decoder_v1(channels_in, channels_out, args)  # Decoder(channels_in, channels_out, args)
        elif args.decoder == "decoder_v2":
            self.decoder = Decoder_v2(channels_in, channels_out, args)  # Decoder(channels_in, channels_out, args)
        self.decoder.init_weights()

    def forward(self, frame1, frame2):                          
        frames = torch.cat([frame1, frame2])                        # [bs, 3, H, W]
        conv_feats = self.encoder(frames)                           # [2*bs, model_scale * 32, H/model_scale, W/model_scale]

        conv_feats1, conv_feats2 = conv_feats[0].chunk(2, dim=0)    # [bs, model_scale * 32, H/model_scale, W/model_scale] * 2

        out_d1, out_r12, out_t12, out_d2, out_r21, out_t21  = self.decoder(conv_feats1, conv_feats2)                          # out_d: [bs, embed_dim * 2, H, W]    out_p: [bs, 12]
        
        return {'pred_d1': out_d1,   'pred_d2': out_d2, 
                'pred_r12': out_r12, 'pred_r21': out_r21, 
                'pred_t12': out_t12, 'pred_t21': out_t21
                }


