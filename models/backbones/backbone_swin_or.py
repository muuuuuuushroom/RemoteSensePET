# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from typing import Dict, List


import sys 
sys.path.append('./')
sys.path.append('./models')

from util.misc import NestedTensor, is_main_process

sys.path.insert(0, '/data/zlt/PET/RTC/models')
from position_encoding import build_position_encoding
# FeatsFusion, Joiner ORIGINALLY:
from .backbone_vgg import FeatsFusion, Joiner


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase_Swin(nn.Module):

    def __init__(self, name: str, backbone: nn.Module, num_channels: int, return_interm_layers: bool):
        super().__init__()
        features = list(backbone.features.children())
        if return_interm_layers:
            if 'swin' in name:
                self.body1 = nn.Sequential(*features[:2])
                self.body2 = nn.Sequential(*features[2:4])
                self.body3 = nn.Sequential(*features[4:6])
                self.body4 = nn.Sequential(*features[6:])

                if name in ('swin_t', 'swin_s', 'swin_v2_t', 'swin_v2_s'):
                    C_size_list = [192, 384, 768]
                elif name in ('swin_b', 'swin_v2_b'):
                    C_size_list = [256, 512, 1024]
                else:
                    raise NotImplementedError
                
                self.fpn = FeatsFusion(
                    C_size_list[0], C_size_list[1], C_size_list[2], 
                    hidden_size=num_channels,
                    out_size=num_channels, 
                    out_kernel=3
                )
            else:
                raise NotImplementedError
        else:
            self.body = nn.Sequential(*features[:])
    
        self.num_channels = num_channels
        self.return_interm_layers = return_interm_layers
        

    def forward(self, tensor_list: NestedTensor):
        feats = []
        if self.return_interm_layers:
            xs = tensor_list.tensors
            for idx, layer in enumerate([self.body1, self.body2, self.body3, self.body4]):
                xs = layer(xs)
                feats.append(xs.permute(0, 3, 1, 2).contiguous())  # BHWC -> BCHW
            
            # feature fusion
            # for feat in feats:
            #     print(feat.shape)
            features_fpn = self.fpn([feats[1], feats[2], feats[3]])
            features_fpn_4x = features_fpn[0]
            features_fpn_8x = features_fpn[1]

             # get tensor mask
            m = tensor_list.mask
            assert m is not None
            mask_4x = F.interpolate(m[None].float(), size=features_fpn_4x.shape[-2:]).to(torch.bool)[0]
            mask_8x = F.interpolate(m[None].float(), size=features_fpn_8x.shape[-2:]).to(torch.bool)[0]

            out: Dict[str, NestedTensor] = {}
            out['4x'] = NestedTensor(features_fpn_4x, mask_4x)
            out['8x'] = NestedTensor(features_fpn_8x, mask_8x)
        else:
            xs = self.body(tensor_list)
            out.appand(xs)
        
        return out


class Backbone_Swin(BackboneBase_Swin):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str, return_interm_layers: bool, num_channels=None):
        backbone = getattr(torchvision.models, name)(
            pretrained=True,
            # norm_layer=FrozenBatchNorm2d
        )
        # if is_main_process():
        #     checkpoint = torch.load('/home/slcao/.cache/torch/hub/checkpoints/swin_t-704ceda3.pth')
        #     backbone.load_state_dict(checkpoint, strict=False)

        if num_channels is None:
            num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(name, backbone, num_channels, return_interm_layers)


def build_backbone_swin(args):
    position_embedding = build_position_encoding(args)
    backbone = Backbone_Swin(args.backbone, return_interm_layers=True, num_channels=args.backbone_num_channels)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


if __name__ == "__main__":
    class Configs():
        hidden_dim = 256
        backbone_num_channels = 512
        position_embedding = 'sine'
        # backbone = 'swin_v2_s'
        backbone = 'swin_s'
    args = Configs()
    backbone = build_backbone_swin(args)
    n_parameters = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print('params:', n_parameters/1e6)

    # generate random intput: NestedTensor
    x = torch.rand((8, 3, 224, 224))
    mask = torch.zeros((8, 224, 224), dtype=torch.bool)
    input = NestedTensor(x, mask)
    out, pos = backbone(input)

    print('out:')
    for k, v in out.items():
        print(k, v.tensors.shape)
    print('pos:')
    for k, v in pos.items():
        print(k, v.shape)
