import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Basic Layers
class conv_bn_relu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(conv_bn_relu, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# Backbone
class ResNet_Multi_Scale(nn.Module):
    """ ResNet Multi-Scale Hough Transform """
    def __init__(self, args, pretrained=True, train_backbone=True):
        super(ResNet_Multi_Scale, self).__init__()
        if args.cnn_model == "resnet50" or args.cnn_model == "50":
            resnet = models.resnet50(pretrained=pretrained)
            self.num_channels = [512, 1024, 2048]
            self.scale_factor = [8, 16, 32]
        elif args.cnn_model == "resnet18" or args.cnn_model == "18":
            resnet = models.resnet18(pretrained=pretrained)
            self.num_channels = [128, 256, 512]
            self.scale_factor = [8, 16, 32]
        else:
            print("[Error] check CNN_MODEL parameter in config(resnet50 or resnet18)")
            exit(0)
        resnet = nn.Sequential(*list(resnet.children())[:-2])

        if not train_backbone:
            for name, parameter in resnet.named_parameters():
                parameter.requires_grad_(False)

        # self.forward2 = nn.Sequential(*list(resnet.children())[:5])
        self.feature3 = nn.Sequential(*list(resnet.children())[:6])
        self.feature4 = nn.Sequential(*list(resnet.children())[6:7])
        self.feature5 = nn.Sequential(*list(resnet.children())[7:8])
        
    def forward(self, x):
        # f2 = self.feature2(x)       # B, 256, H//4,  W//4
        f3 = self.feature3(x)       # B, 512, H//8,  W//8
        f4 = self.feature4(f3)      # B, 1024, H//16, W//16
        f5 = self.feature5(f4)      # B, 2048, H//32, W//32

        return f3, f4, f5

class ResNet(nn.Module):
    """ ResNet """
    def __init__(self, args, pretrained=True, train_backbone=True):
        super(ResNet, self).__init__()
        if args.cnn_model == "resnet50" or args.cnn_model == "50":
            resnet = models.resnet50(pretrained=pretrained)
            self.num_channel = 1024
            self.scale_factor = 16
        elif args.cnn_model == "resnet18" or args.cnn_model == "18":
            resnet = models.resnet18(pretrained=pretrained)
            self.num_channel = 256
            self.scale_factor = 16
        else:
            print("[Error] check CNN_MODEL parameter in config(resnet50 or resnet18)")
            exit(0)
        resnet = nn.Sequential(*list(resnet.children())[:-2])

        if not train_backbone:
            for name, parameter in resnet.named_parameters():
                parameter.requires_grad_(False)

        self.feature = nn.Sequential(*list(resnet.children())[:7])

    def forward(self, x):
        f = self.feature(x)       # B, 512, H//8,  W//8
        return f


# Network Blocks
class Feature_Extractor_Multi_Scale(nn.Module):
    def __init__(self, args, hidden_dim):
        super(Feature_Extractor_Multi_Scale, self).__init__()
        self.args = args
        self.hidden_dim = hidden_dim

        # Feature Extraction
        self.backbone = ResNet_Multi_Scale(args)
        self.num_channels = self.backbone.num_channels
        self.scale_factor = self.backbone.scale_factor

        self.feat_squeeze1 = nn.Sequential(
            conv_bn_relu(self.num_channels[0], self.hidden_dim, kernel_size=3, stride=1, padding=1),
            conv_bn_relu(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1),
        )
        self.feat_squeeze2 = nn.Sequential(
            conv_bn_relu(self.num_channels[1], self.hidden_dim, kernel_size=3, stride=1, padding=1),
            conv_bn_relu(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1),
        )
        self.feat_squeeze3 = nn.Sequential(
            conv_bn_relu(self.num_channels[2], self.hidden_dim, kernel_size=3, stride=1, padding=1),
            conv_bn_relu(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1),
        )

        self.feat_combine = nn.Sequential(nn.Conv2d(self.hidden_dim * 3, self.hidden_dim, kernel_size=1),
                                          nn.BatchNorm2d(self.hidden_dim),
                                          nn.ReLU(),
                                          nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1))

    def forward(self, img):
        f3, f4, f5 = self.backbone(img)

        f3 = self.feat_squeeze1(f3)
        f4 = self.feat_squeeze2(f4)
        f5 = self.feat_squeeze3(f5)

        f3 = F.interpolate(f3, scale_factor=0.5, recompute_scale_factor=True)
        f5 = F.interpolate(f5, scale_factor=2, recompute_scale_factor=True)

        feats = torch.cat([f3, f4, f5], dim=1)
        img_feat = self.feat_combine(feats)
        return img_feat

class Feature_Extractor(nn.Module):
    def __init__(self, args, hidden_dim):
        super(Feature_Extractor, self).__init__()
        self.args = args
        self.hidden_dim = hidden_dim

        # Feature Extraction
        self.backbone = ResNet(args)
        self.num_channel = self.backbone.num_channel
        self.scale_factor = self.backbone.scale_factor

        self.feat_squeeze = nn.Sequential(
            conv_bn_relu(self.num_channel, self.hidden_dim, kernel_size=3, stride=1, padding=1),
            conv_bn_relu(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1),
        )


    def forward(self, img):
        f = self.backbone(img)

        f = self.feat_squeeze(f)
        return f

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        b, c, h, w = x.shape
        mask = torch.zeros(b, h, w, dtype=torch.bool).cuda()     # .float()
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class Transformer_Encoder(nn.Module):
    def __init__(self, args, hidden_dim):
        super(Transformer_Encoder, self).__init__()
        self.args = args
        self.hidden_dim = hidden_dim
        self.dim_feedforward = args.transformer_ff_dim

        # Encoder
        self.dropout = nn.Dropout(0.)
        if self.hidden_dim == 256:
            num_heads=4
        elif self.hidden_dim == 512:
            num_heads=8
        self.self_attn = nn.MultiheadAttention(self.hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.ffn1 = nn.Sequential(nn.Linear(self.hidden_dim, self.dim_feedforward),
                                  nn.ReLU())
        self.ffn2 = nn.Sequential(nn.Linear(self.dim_feedforward, self.hidden_dim))
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        # self.register_buffer('self_attn_weight', None)

    def forward(self, img_feat, img_pos):
        # Self-Attn
        q = k = img_feat + img_pos
        v = img_feat
        x, self_sim = self.self_attn(q, k, v)
        x = v + self.dropout(x)
        x = self.norm1(x)

        # FFN
        x2 = self.ffn1(x)
        x2 = self.ffn2(self.dropout(x2))
        x = x + self.dropout(x2)
        img_feat = self.norm2(x)

        # # Register buffer
        # self.self_attn_weight = self_sim

        return img_feat


class cnn_transformer(nn.Module):
    def __init__(self, args, hidden_dim=512, n_enc_layers=6, resnet_multi_scale=True):
        super(cnn_transformer, self).__init__()
        self.args = args
        self.hidden_dim = hidden_dim
        self.n_enc_layers = n_enc_layers

        # Backbone
        if resnet_multi_scale:
            self.feature_extractor = Feature_Extractor_Multi_Scale(args, hidden_dim=self.hidden_dim)
        else:
            self.feature_extractor = Feature_Extractor(args, hidden_dim=self.hidden_dim)
        self.pos_embed = PositionEmbeddingSine(num_pos_feats=self.hidden_dim // 2, temperature=20, normalize=True)

        # Transformer
        self.transformer_encoder = nn.ModuleList([Transformer_Encoder(args, hidden_dim=self.hidden_dim) for _ in range(self.n_enc_layers)])

    def forward(self, img, targets=None):
        img_feat = self.feature_extractor(img)              # [b, c, h, w]
        img_pos = self.pos_embed(img_feat)                  # [b, c, h, w]
        b, c, h, w = img_feat.shape

        img_feat = img_feat.flatten(2).permute(0, 2, 1)     # [b, h*w, c]
        img_pos = img_pos.flatten(2).permute(0, 2, 1)       # [b, h*w, c]

        for layer_idx in range(self.n_enc_layers):
            img_feat = self.transformer_encoder[layer_idx](img_feat, img_pos)

        img_feat = img_feat.permute(0, 2, 1).reshape(b, c, h, w)    # [b, c, h, w]
        return [img_feat]
