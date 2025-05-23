# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import torch.distributed as dist


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch']
        config.freeze()
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
    if 'max_accuracy' in checkpoint:
        max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy

def load_pretrained(ckpt_path, model, logger):
    logger.info(f"==============> Loading pretrained form {ckpt_path}....................")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    # msg = model.load_pretrained(checkpoint['model'])
    # logger.info(msg)
    # logger.info(f"=> Loaded successfully {ckpt_path} ")
    # del checkpoint
    # torch.cuda.empty_cache()
    state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # linear interpolate agent bias if not match
    agent_bias_keys = [k for k in state_dict.keys() if ("ah_bias" in k) or ("aw_bias" in k)
                                or ("ha_bias" in k) or ("wa_bias" in k)]
    for k in agent_bias_keys:
        if "ah_bias" in k:
            squeeze_dim, permute = -1, False
        elif "aw_bias" in k:
            squeeze_dim, permute = -2, False
        elif "ha_bias" in k:
            squeeze_dim, permute = -2, True
        else:
            squeeze_dim, permute = -3, True
        agent_bias_pretrained = state_dict[k].squeeze(dim=0).squeeze(dim=squeeze_dim)
        agent_bias_current = model.state_dict()[k].squeeze(dim=0).squeeze(dim=squeeze_dim)
        if permute:
            agent_bias_pretrained = agent_bias_pretrained.permute(0, 2, 1)
            agent_bias_current = agent_bias_current.permute(0, 2, 1)
        num_heads1, agent_num1, hw1 = agent_bias_pretrained.size()
        num_heads2, agent_num2, hw2 = agent_bias_current.size()
        if (num_heads1 != num_heads2) or (agent_num1 != agent_num2):
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if hw1 != hw2:
                # linear interpolate agent bias if not match
                agent_bias_pretrained_resized = torch.nn.functional.interpolate(
                    agent_bias_pretrained, size=hw2, mode='linear')
                if permute:
                    agent_bias_pretrained_resized = agent_bias_pretrained_resized.permute(0, 2, 1)
                state_dict[k] = agent_bias_pretrained_resized.unsqueeze(dim=0).unsqueeze(dim=squeeze_dim)

    # bicubic interpolate patch_embed.proj if not match
    patch_embed_keys = [k for k in state_dict.keys() if ("patch_embed" in k) and (".proj.weight" in k)]
    for k in patch_embed_keys:
        patch_embed_pretrained = state_dict[k]
        patch_embed_current = model.state_dict()[k]
        out1, in1, h1, w1 = patch_embed_pretrained.size()
        out2, in2, h2, w2 = patch_embed_current.size()
        if (out1 != out2) or (in1 != in2):
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if (h1 != h2) or (w1 != w2):
                # bicubic interpolate patch_embed.proj if not match
                patch_embed_pretrained_resized = torch.nn.functional.interpolate(
                    patch_embed_pretrained, size=(h2, w2), mode='bicubic')
                state_dict[k] = patch_embed_pretrained_resized

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                i, j = L1 - S1 ** 2, L2 - S2 ** 2
                absolute_pos_embed_pretrained_ = absolute_pos_embed_pretrained[:, i:, :].reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained_ = absolute_pos_embed_pretrained_.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained_, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = torch.cat([absolute_pos_embed_pretrained[:, :j, :],
                                           absolute_pos_embed_pretrained_resized], dim=1)

    # check classifier, if not match, then re-init classifier to zero
    # head_bias_pretrained = state_dict['head.bias']
    # Nc1 = head_bias_pretrained.shape[0]
    # Nc2 = model.head.bias.shape[0]
    # if (Nc1 != Nc2):
    #     if Nc1 == 21841 and Nc2 == 1000:
    #         logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
    #         map22kto1k_path = f'data/map22kto1k.txt'
    #         with open(map22kto1k_path) as f:
    #             map22kto1k = f.readlines()
    #         map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
    #         state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
    #         state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
    #     else:
    #         torch.nn.init.constant_(model.head.bias, 0.)
    #         torch.nn.init.constant_(model.head.weight, 0.)
    #         del state_dict['head.weight']
    #         del state_dict['head.bias']
    #         logger.warning(f"Error in loading classifier head, re-init classifier head to 0")

    msg = model.load_state_dict(state_dict, strict=False)
    logger.warning(msg)

    logger.info(f"=> loaded successfully '{ckpt_path}'")

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")
    
def save_checkpoint_new(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger, name=None):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    if name == None:
        old_ckpt = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch-3}.pth') 
        if os.path.exists(old_ckpt):
            os.remove(old_ckpt)

    if name!=None:
        save_path = os.path.join(config.OUTPUT, f'{name}.pth')
        logger.info(f"{save_path} saving......")
        torch.save(save_state, save_path)
        logger.info(f"{save_path} saved !!!")
    else:
        save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
        logger.info(f"{save_path} saving......")
        torch.save(save_state, save_path)
        logger.info(f"{save_path} saved !!!")

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt
