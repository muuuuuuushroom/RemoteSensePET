from .backbone_vgg import build_backbone_vgg
from .backbone_swin import build_backbone_swin
from .backbone_agent_swin import build_backbone_agent_swin


def build_bockbone(args):
    if 'vgg' in args.backbone:
        print('build vgg backbone')
        return build_backbone_vgg(args)
    elif args.backbone in ['swin_t', 'swin_s', 'swin_b']:
        print('build swin backbone')
        return build_backbone_swin(args)
    elif 'agent_swin' in args.backbone:
        print(f'build {args.backbone} backbone')
        return build_backbone_agent_swin(args)
    
    else:
        raise NotImplementedError