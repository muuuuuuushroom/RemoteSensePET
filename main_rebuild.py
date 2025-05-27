import argparse
import datetime
import json
import random
import time
import logging
from pathlib import Path
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from util.custom_log import *
from util.data_aug_zlt import create_hard_case_sampler_train


def get_args_parser():
    parser = argparse.ArgumentParser('Set Point Query Transformer', add_help=False)
    parser.add_argument('--cfg', type=str, default='configs_con/test.yaml', 
                        help='base cfg file for training model')
    
    parser.add_argument('--save_ckpt_freq', type=int, default=500,
                        help="the frequency to save ckpt")
    # misc parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default=None, help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    
    parser.add_argument('--eval_pad', default='padding_center')
    parser.add_argument('--eval_robust', default=[])
    parser.add_argument('--robust_para', default=None)
    
    # parser.add_argument('--prob_map_lc', default=None)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # pet decoder
    # parser.add_argument('--opt_query_decoder', default=True, type=bool, help='reference: box-detr')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank() # rank: distributed training
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model, criterion = build_model(args)
    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'n_parameters: {n_parameters}')
    if args.syn_bn: # distributed training
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # build optimizer
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.epochs)

    # build dataset
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False, drop_last=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    print(f'val data: {len(dataset_val)}\t train data: {len(dataset_train)}')

    # output directory and log 
    if utils.is_main_process:
        output_dir = os.path.join("./outputs", args.dataset_file, args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        output_dir = Path(output_dir)
        save_config(args, os.path.join(output_dir, 'config.yaml'))
        run_log_name = os.path.join(output_dir, 'run_log.txt')
        with open(run_log_name, "a") as log_file:
            log_file.write('Run Log %s\n' % time.strftime("%c"))
            log_file.write("{}\n".format(args))
            log_file.write("parameters: {}\n".format(n_parameters))

    # resume
    best_mae, best_epoch = 1e8, 0
    if args.resume:
        print(f'resume from: {args.resume}')
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            best_mae = checkpoint['best_mae']
            best_epoch = checkpoint['best_epoch']

    # training
    print("\ntraining\n")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        try:
            data_loader_train.dataset.set_epoch(epoch)
        except:
            pass
        if args.distributed:
            sampler_train.set_epoch(epoch)
        
        t1 = time.time()
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        t2 = time.time()
        print('[ep %d][lr %.7f][%.2fs]' % \
              (epoch, optimizer.param_groups[0]['lr'], t2 - t1))
        
        if utils.is_main_process:
            with open(run_log_name, "a") as log_file:
                log_file.write('\n[ep %d][lr %.7f][%.2fs]' % (epoch, optimizer.param_groups[0]['lr'], t2 - t1))

        lr_scheduler.step()

        # save checkpoint
        if (epoch+1) % args.save_ckpt_freq == 0:     #or (epoch+1) in save_list:
            checkpoint_paths = [output_dir / f'epoch_{epoch+1}.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch+1,
                    'args': args,
                    'best_mae': best_mae,
                    'best_epoch': best_epoch
                }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        # write log
        if utils.is_main_process():
            with open(run_log_name, "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        # evaluation
        if (epoch) % args.eval_freq == 0:
            print('\ntesting\n')
            t1 = time.time()
            test_stats = evaluate(model, data_loader_val, device, epoch, criterion=criterion, distributed=args.distributed, args=args)
            t2 = time.time()

            # output results
            mae, mse, r2 = test_stats['mae'], test_stats['mse'], test_stats['r2']
            rmae, rmse, racc = test_stats['rmae'], test_stats['rmse'], test_stats['racc']
            if mae < best_mae:
                best_epoch = epoch
                best_mae = mae
            print("\n==========================")
            print("\nepoch:", epoch, "mae:", mae, "mse:", mse, "r2", r2, "\n\nbest mae:", best_mae, "best epoch:", best_epoch)
            print("\n==========================\n")
            if utils.is_main_process():
                with open(run_log_name, "a") as log_file:
                    log_file.write("\nepoch:{}, mae:{}, mse:{}, r2:{}, rmae:{}, rmse:{}, racc{}, time{}\n\n stats:{}\n\nbest mae:{}, best epoch: {}\n\n".format(
                                                epoch, mae, mse, r2, rmae, rmse, racc, t2 - t1, test_stats, best_mae, best_epoch))
                                                
            # save best checkpoint
            if mae == best_mae and utils.is_main_process():
                checkpoint_paths = [output_dir / f'best_checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch+1,
                        'args': args,
                        'best_mae': best_mae,
                        'best_epoch': best_epoch
                    }, checkpoint_path)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PET training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    
    config = load_config(args.cfg)
    args = update_args_with_config(args, config)

    main(args)
