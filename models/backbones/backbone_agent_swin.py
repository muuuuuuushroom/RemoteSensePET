# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
import torch.distributed as dist
from typing import Dict, List

import sys 
sys.path.append('./')
sys.path.append('./models')

from util.misc import NestedTensor, is_main_process
from position_encoding import build_position_encoding
from .backbone_vgg import FeatsFusion, Joiner
from .agent_swin import AgentSwinTransformer
from agent_transformer.config import _C, _update_config_from_file
from agent_transformer.utils import load_pretrained
from agent_transformer.logger import create_logger


pretrained_path = {
    'agent_swin_t': '/data/slcao_data/pretrain/agent_swin_t.pth',
    'agent_swin_s_288': '/data/slcao_data/pretrain/agent_swin_s_288.pth',
    'agent_swin_b_384': '/data/slcao_data/pretrain/agent_swin_b_384.pth'
}


class BackboneBase_Agent_Swin(nn.Module):
    def __init__(self, name: str, backbone: nn.Module, num_channels: int, return_interm_layers: bool):
        super().__init__()
        features = list(backbone.layers.children())
        if return_interm_layers:
            if 'swin' in name:
                self.backbone = backbone

                if name == 'agent_swin_t' or name == 'agent_swin_s_288':
                    C_size_list = [192, 384, 768]
                elif name == 'agent_swin_b_384':
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
            feats = self.backbone(xs)
            
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
            # print(f"4x:{features_fpn_4x.shape}, 8x:{features_fpn_8x.shape}")
            out: Dict[str, NestedTensor] = {}
            out['4x'] = NestedTensor(features_fpn_4x, mask_4x)
            out['8x'] = NestedTensor(features_fpn_8x, mask_8x)
        else:
            xs = self.body(tensor_list)
            out.appand(xs)
        
        return out


class Backbone_Agent_Swin(BackboneBase_Agent_Swin):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str, img_size: int, return_interm_layers: bool, num_channels=None):
        backbone = agent_swin(name, img_size)

        if num_channels is None:
            num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(name, backbone, num_channels, return_interm_layers)


def agent_swin(name, img_size):
    config = _C.clone()
    if name == 'agent_swin_t':
        _update_config_from_file(config, 'models/agent_transformer/cfgs/agent_swin_t.yaml')
    elif name == 'agent_swin_s_288':
        _update_config_from_file(config, 'models/agent_transformer/cfgs/agent_swin_s_288.yaml')
    elif name == 'agent_swin_b_384':
        _update_config_from_file(config, 'models/agent_transformer/cfgs/agent_swin_b_384.yaml')
    else:
        raise NotImplementedError

    model = AgentSwinTransformer(
        img_size=img_size,
        # img_size=config.DATA.IMG_SIZE,
        patch_size=config.MODEL.SWIN.PATCH_SIZE,
        in_chans=config.MODEL.SWIN.IN_CHANS,
        num_classes=config.MODEL.NUM_CLASSES,
        embed_dim=config.MODEL.SWIN.EMBED_DIM,
        depths=config.MODEL.SWIN.DEPTHS,
        num_heads=config.MODEL.SWIN.NUM_HEADS,
        window_size=int(img_size/config.MODEL.SWIN.PATCH_SIZE),
        mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
        qkv_bias=config.MODEL.SWIN.QKV_BIAS,
        qk_scale=config.MODEL.SWIN.QK_SCALE,
        drop_rate=config.MODEL.DROP_RATE,
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        ape=config.MODEL.SWIN.APE,
        patch_norm=config.MODEL.SWIN.PATCH_NORM,
        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        agent_num=config.MODEL.AGENT.NUM.split('-'),
        attn_type=config.MODEL.AGENT.ATTN_TYPE)

    logger = create_logger(output_dir='./', dist_rank=0, name=f"{config.MODEL.NAME}")
    load_pretrained(pretrained_path[name], model, logger)
    
    return model


def build_backbone_agent_swin(args):
    position_embedding = build_position_encoding(args)
    backbone = Backbone_Agent_Swin(args.backbone, args.patch_size, return_interm_layers=True, num_channels=args.backbone_num_channels)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


if __name__ == "__main__":
    class Configs():
        hidden_dim = 256
        patch_size = 256
        backbone_num_channels = 512
        position_embedding = 'sine'
        backbone = 'agent_swin_b_384'
    args = Configs()
    backbone = build_backbone_agent_swin(args)
    n_parameters = sum(p.numel() for p in backbone[0].parameters() if p.requires_grad)
    print('params:', n_parameters/1e6)

    # generate random intput: NestedTensor
    x = torch.rand((8, 3, 256, 256))
    mask = torch.zeros((8, 256, 256), dtype=torch.bool)
    input = NestedTensor(x, mask)
    out, pos = backbone(input)

    print('out:')
    for k, v in out.items():
        print(k, v.tensors.shape)
    print('pos:')
    for k, v in pos.items():
        print(k, v.shape)
