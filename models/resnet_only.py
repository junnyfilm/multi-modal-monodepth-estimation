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

class resnet_only(nn.Module):
    def __init__(self, args, hidden_dim=512, resnet_multi_scale=True):
        super(resnet_only, self).__init__()
        self.args = args
        self.hidden_dim = hidden_dim

        # Backbone
        if resnet_multi_scale:
            self.feature_extractor = Feature_Extractor_Multi_Scale(args, hidden_dim=self.hidden_dim)
        else:
            self.feature_extractor = Feature_Extractor(args, hidden_dim=self.hidden_dim)

    def forward(self, img, targets=None):
        img_feat = self.feature_extractor(img)              # [b, c, h, w]
        b, c, h, w = img_feat.shape

        return [img_feat]
