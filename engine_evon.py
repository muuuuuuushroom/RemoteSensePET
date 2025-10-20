"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import numpy as np
import cv2

import torch
import torchvision.transforms as standard_transforms
import torch.nn.functional as F

import util.misc as utils
from util.misc import NestedTensor

from sklearn.metrics import r2_score 

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def visualization(samples, img_path, pred, vis_dir, split_map=None, queries=None):
    """
    Visualize predictions
    """
    pil_to_tensor = standard_transforms.ToTensor()

    restore_transform = standard_transforms.Compose([
        DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        standard_transforms.ToPILImage()
    ])

    images = samples.tensors
    masks = samples.mask
    for idx in range(images.shape[0]):
        sample = restore_transform(images[idx])
        sample = pil_to_tensor(sample.convert('RGB')).numpy() * 255
        sample_vis = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()
        h, w = sample_vis.shape[:2]
        # draw ground-truth points (red)
        size = 3
        # draw predictions (green)
        for p in pred[idx]:
            sample_vis = cv2.circle(sample_vis, (int(p[1]), int(p[0])), size, (0, 255, 0), -1)
            
        # draw point-query
        # for i, q in enumerate(queries[idx]):
        #     q[1] *= w
        #     q[0] *= h
        #     sample_vis = cv2.circle(
        #         sample_vis, (int(q[1]), int(q[0])), size, (0, 255, 255), -1
        #         )
        #     # draw line between query and pred
        #     q_x, q_y = int(q[1]), int(q[0])
        #     p_x, p_y = int(pred[idx][i][1]*w), int(pred[idx][i][0]*h)
        #     overlay = sample_vis.copy()
        #     cv2.line(overlay, (p_x, p_y), (q_x, q_y), (0, 255, 0), 2) 
        #     alpha = 0.5  
        #     sample_vis = cv2.addWeighted(overlay, alpha, sample_vis, 1 - alpha, 0)
        
        # draw split map
        if split_map is not None:
            imgH, imgW = sample_vis.shape[:2]
            split_map = (split_map * 255).astype(np.uint8)
            split_map = cv2.applyColorMap(split_map, cv2.COLORMAP_JET)
            split_map = cv2.resize(split_map, (imgW, imgH), interpolation=cv2.INTER_NEAREST)
            sample_vis = split_map * 0.9 + sample_vis
        
        # save image
        if vis_dir is not None:
            # eliminate invalid area
            imgH, imgW = masks.shape[-2:]
            valid_area = torch.where(~masks[idx])
            valid_h, valid_w = valid_area[0][-1], valid_area[1][-1]
            sample_vis = sample_vis[:valid_h+1, :valid_w+1]

            print(img_path)
            print(img_path[0])
            name = img_path[0].split('/')[-1]
            cv2.imwrite(os.path.join(vis_dir, '{}_gt{}_pred{}.jpg'.format(name, len(pred[idx]))), sample_vis)
            
def apply_ignore_to_padding_mask(samples, targets):
    _, _, H_pad, W_pad = samples.tensors.shape

    for b, tgt in enumerate(targets):
        if 'mask_ignore' not in tgt:
            continue 
        ignore = tgt['mask_ignore']                      
        valid = (ignore[:, 2:] - ignore[:, :2]).prod(-1) > 4.0
        ignore = ignore[valid]
        if ignore.numel() == 0:
            continue

        boxes = ignore.round().long()
        boxes[:, [0, 2]].clamp_(0, H_pad)    # y1, y2
        boxes[:, [1, 3]].clamp_(0, W_pad)    # x1, x2

        for y1, x1, y2, x2 in boxes:
            samples.mask[b, y1:y2, x1:x2] = True
        

# evaluation
@torch.no_grad()
def evaluate(model, data_loader, device, epoch=0, vis_dir=None):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)

    y_pred_all = []
    results = {}
    
    print_freq = 10
    for samples, img_path, _ in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        img_h, img_w = samples.tensors.shape[-2:]

        # inference
        outputs = model(samples, test=True, targets=None)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
        outputs_points = outputs['pred_points'][0]
        outputs_offsets = outputs['pred_offsets'][0]
        outputs_queries = outputs['points_queries']
        
        # process predicted points
        predict_cnt = len(outputs_scores)
        y_pred_all.append(predict_cnt)
        
        key_name = img_path[0].split('/')[-1].replace('.jpg', '')
        results[key_name] = predict_cnt
        
        # visualize predictions
        # if vis_dir: 
            
        #     points = [[point[0]*img_h, point[1]*img_w] for point in outputs_points]     # recover to actual points
        #     split_map = (outputs['split_map_raw'][0].detach().cpu().squeeze(0) > 0.5).float().numpy()
        #     visualization(samples, img_path, [points], vis_dir, split_map=split_map, queries=outputs_queries)

    return results