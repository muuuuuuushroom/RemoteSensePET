import argparse
import random
from pathlib import Path
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

import datasets
from datasets import build_dataset
import util.misc as utils

from models import build_model
from engine import evaluate

from util.custom_log import *


def get_args_parser():
    parser = argparse.ArgumentParser('Set Point Query Transformer', add_help=False)
    
    parser.add_argument('--cfg', default='/data/zlt/PET/RTC/outputs_con/Ship/base_box_detr/config.yaml')
    parser.add_argument('--gt_determined', default='100', help='test')
    # misc parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='/data/zlt/PET/RTC/outputs_con/Ship/base_box_detr/best_checkpoint.pth', help='resume from checkpoint')
    parser.add_argument('--vis_dir', default=None)
    parser.add_argument('--eval_pad', default='padding_center')
    parser.add_argument('--eval_robust', default=[])
    parser.add_argument('--robust_para', default=None)
    parser.add_argument('--prob_map_lc', default=None)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)
    
    # if args.dataset_file in ['Ship', 'Car', 'People']:
    #     from engine_rsc import evaluate
    # elif args.dataset_file in ['RTC', 'SOY']:
    #     from engine import evaluate

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model, criterion = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # build dataset
    val_image_set = 'val'
    dataset_val = build_dataset(image_set=val_image_set, args=args)

    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        
    val_batch_size = 1  # if args.dataset_file == 'RTC' else 4
    data_loader_val = DataLoader(dataset_val, val_batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # load pretrained model
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print(f"load successfully from ckpt: {args.resume}")
        cur_epoch = checkpoint['epoch'] - 1 if 'epoch' in checkpoint else 0

    # evaluation
    vis_dir = None if args.vis_dir == "" else args.vis_dir
    if vis_dir != None:
        if os.path.exists(vis_dir):
            import shutil
            shutil.rmtree(vis_dir)
        os.makedirs(vis_dir, exist_ok=True)  
    
    import time
    t1 = time.time()
    test_stats = evaluate(model, data_loader_val, device, vis_dir=vis_dir, args=args, criterion=criterion)
    t2 = time.time()
    
    infer_time = t2 - t1
    fps = len(dataset_val) / infer_time 
    print(args, f"\n\ninfer from: {args.output_dir}\ninferring time: {infer_time:.4f}s")
    print(f'FPS: {fps:.2f}')
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('params:', n_parameters/1e6)
    # mae, mse = test_stats['mae'], test_stats['mse']
    # line = f"epoch: {cur_epoch}"  # , mae: {mae}, mse: {mse}, r2: {test_stats['r2']}, rmae: {test_stats['rmae']} "
    print(f'epoch: {cur_epoch}\t\t\tgt > {args.gt_determined} ended with \'ac\' below:')
    count = 0
    for k, v in test_stats.items():
        if count % 2 == 0:
            print(k,'\t', v, end='\t')
        else:
            print(k,'\t', v)
        count += 1
    
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PET evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    config = load_config(args.cfg)
    args = update_args_with_config(args, config)
    
    args.opt_query_con = False if not hasattr(args, 'opt_query_con') else args.opt_query_con
    
    main(args)
