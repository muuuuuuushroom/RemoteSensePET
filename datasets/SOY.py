import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import time
import json
import torchvision.transforms as standard_transforms
import warnings
warnings.filterwarnings('ignore')

import sys 
sys.path.append('./')
sys.path.append('./models')
import util.misc as utils

class SOY(Dataset):
    def __init__(
        self, 
        data_root, 
        transform=None,
        train=False,
        flip=False,
        patch_size=512, 
        preload=False, 
        global_crop_ratio=0.1,
        probs=[0.05, 0.5, 1e-4],
        total_steps=300,
    ):
        super().__init__()
        self.root_path = data_root
        
        dataset_type = "distribution/train_list" if train else "distribution/test_list"
        # dataset_type = "train" if train else "test"
        data_list_path = os.path.join(data_root, dataset_type+'.txt')
        self.data_list = [name.split(' ') for name in open(data_list_path).read().splitlines()]
        self.nSamples = len(self.data_list)

        self.transform = transform
        self.train = train
        self.flip = flip
        self.patch_size = patch_size
        self.preload = preload  # no using

        if isinstance(global_crop_ratio, str):
            self.global_sample_ratio = 0.1
            self.crop_ratio_type = 'dynamic'
        else:
            print('static global crop ratio')
            self.crop_ratio_type = 'staic'
            self.global_sample_ratio = global_crop_ratio
        
        if self.train and isinstance(global_crop_ratio, str):
            self.prob_schedule = self.re_onecycle_prob(probs[0], probs[1], probs[2], total_steps)
            self.global_sample_ratio = self.prob_schedule[0]

        self.img_loaded = {}
        self.point_loaded = {}
        # if self.preload:
        #     st = time.time()
        #     mean = [0.485, 0.456, 0.406]
        #     std = [0.229, 0.224, 0.225]
        #     print('preload data begin')
        #     import tqdm as td
        #     for sample in td.tqdm(self.data_list):
        #         img_path = os.path.join(self.root_path, sample[0].replace('images/', 'img_norm_tensor/').replace('.jpg', '.npy'))
        #         gt_path = os.path.join(self.root_path, sample[1])
        #         img = np.load(img_path)
        #         points = self.parse_json(gt_path)
        #         self.img_loaded[sample[0]] = img
        #         self.point_loaded[sample[1]] = points
        #     # self.img_loaded = np.load(f"{dataset_type}_imgs_data.npy", allow_pickle=True).item()
        #     # self.point_loaded = np.load(f"{dataset_type}_points_data.npy", allow_pickle=True).item()
        #     print(f"load all {len(self.data_list)} samples: {time.time() - st}")
            
    
    def compute_density(self, points):
        """
        Compute crowd density:
            - defined as the average nearest distance between ground-truth points
        """
        points_tensor = torch.from_numpy(points.copy())
        dist = torch.cdist(points_tensor, points_tensor, p=2)
        if points_tensor.shape[0] > 1:
            density = dist.sort(dim=1)[0][:,1].mean().reshape(-1)
        else:
            density = torch.tensor(999.0).reshape(-1)
        return density
    
    def set_epoch(self, epoch):
        """
        change the probility of global crop during every epochs
        """
        if self.train and self.crop_ratio_type == 'dynamic':
            self.global_sample_ratio = self.prob_schedule[epoch]
        print(f"global crop ratio = {self.global_sample_ratio}")

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        # load image and gt points
        img_path = os.path.join(self.root_path, self.data_list[index][0])
        npy_path = img_path.replace('images/', 'images_npy/').replace('.jpg', '.npy')
        gt_path = os.path.join(self.root_path, self.data_list[index][1])
        # img = Image.open(img_path); img = np.array(img).transpose(2,0,1).astype(np.float32); 
        # print(img.shape)
        img = np.load(npy_path)
        points = self.parse_json(gt_path)

        # img = cv2.resize(img, (512, 512))
        # points *= 512 / 2816
        # image transform
        # if self.transform is not None:
        #     img = self.transform(img)

        img = torch.from_numpy(img)


        # random scale
        if self.train:
            scale_range = [0.8, 1.2]
            min_size = min(img.shape[1:])

            scale = random.uniform(*scale_range)
            
            # interpolation
            if scale * min_size > self.patch_size:
                img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
                points *= scale

        # random crop patch
        if self.train:
            img, points = self.random_crop(img, points, patch_size=self.patch_size, ratio=self.global_sample_ratio)

        # random flip horizontal
        if random.random() > 0.5 and self.train and self.flip:
            img = torch.flip(img, dims=[2])
            points[:, 0] = self.patch_size - points[:, 0]

        # target
        target = {}
        target['points'] = torch.from_numpy(points[:,::-1].copy())  # XY->HW due to query points are in a format of HW
        target['labels'] = torch.ones([points.shape[0]]).long()

        if self.train:
            density = self.compute_density(points)
            target['density'] = density

        if not self.train:
            target['image_path'] = img_path

        return img, target, None
    
    @staticmethod
    def re_onecycle_prob(initial_prob, max_prob, final_prob, total_steps, pct_start=0.3, anneal_strategy='cos'):
        '''generate the probability curve of self.global_sample_ratio'''
        schedule = []
        for step in range(total_steps):
            if step < total_steps * pct_start:
                pct = step / (total_steps * pct_start)
                if anneal_strategy == 'cos':
                    cos_out = np.cos(np.pi * pct) + 1
                    prob = max_prob + (initial_prob - max_prob) / 2.0 * cos_out
                else:
                    prob = (max_prob - initial_prob) * pct + initial_prob  # linear
            else:
                pct = (step - total_steps * pct_start) / (total_steps * (1 - pct_start))
                if anneal_strategy == 'cos':
                    cos_out = np.cos(np.pi * pct) + 1
                    prob = final_prob + (max_prob - final_prob) / 2.0 * cos_out
                else:
                    prob = (final_prob - max_prob) * pct + max_prob  # linear
            schedule.append(prob)
        schedule.reverse()

        return schedule

    @staticmethod
    def _annealing_cos(start, end, pct):
        cos_out = np.cos(np.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out
    
    @staticmethod
    def _annealing_linear(start, end, pct):
        return (end - start) * pct + start

    @staticmethod
    def parse_json(gt_path):
        with open(gt_path, 'r') as f:
            tree = json.load(f)

        points = []
        for shape in tree['shapes']:
            points.append(shape['points'][0])
        points = np.array(points, dtype=np.float32)

        return points

    @staticmethod
    def random_crop(img, points, patch_size=256, ratio=0.1):
        
        # if np.random.rand() < ratio:
        #     # full screen crop_window
        #     crop_window = np.array([0, 0, img.size(2) - patch_size, img.size(1) - patch_size], dtype=np.int64)
        # else:
        #     # create the minimum enclosing rectangle of all points (shift by patch_size)
        #     crop_window = np.array([
        #         max(min(points[:, 0]) - patch_size/2., 0),
        #         max(min(points[:, 1]) - patch_size/2., 0),
        #         min(max(points[:, 0]) - patch_size/2., img.size(2)- patch_size),
        #         min(max(points[:, 1]) - patch_size/2., img.size(1)- patch_size),
        #     ], dtype=np.int64)

        
        # if crop_window[1] > crop_window[3]:
        #     crop_window[1], crop_window[3] = crop_window[3], crop_window[1]
        
        crop_window = np.array([0, 0, img.size(2) - patch_size, img.size(1) - patch_size], dtype=np.int64)
        
        
        # random crop
        start_h = random.randint(crop_window[1], crop_window[3]) if img.size(1) > patch_size else 0
        start_w = random.randint(crop_window[0], crop_window[2]) if img.size(2) > patch_size else 0
        end_h = start_h + patch_size
        end_w = start_w + patch_size
        idx = (points[:, 0] >= start_w) & (points[:, 0] <= end_w) & (points[:, 1] >= start_h) & (points[:, 1] <= end_h)

        # clip image and points
        result_img = img[:, start_h:end_h, start_w:end_w]
        result_points = points[idx]
        result_points[:, 1] -= start_h
        result_points[:, 0] -= start_w
        
        # resize to patchsize when necessary
        if min(result_img.shape[-2:]) < patch_size: 
            imgH, imgW = result_img.shape[-2:]
            fH, fW = patch_size/imgH, patch_size/imgW
            result_img = torch.nn.functional.interpolate(result_img.unsqueeze(0), (patch_size, patch_size)).squeeze(0)
            result_points[:, 1] *= fH
            result_points[:, 0] *= fW

        return result_img, result_points


def build_soy(image_set, args):
    data_root = args.data_path
    if image_set == 'train':
        train_set = SOY(data_root, train=True, global_crop_ratio=args.global_crop_ratio, transform=None, patch_size=args.patch_size, flip=True, total_steps=args.epochs)
        return train_set
    elif image_set == 'val':
        val_set = SOY(data_root, train=False, transform=None, patch_size=args.patch_size)
        return val_set
    else:
        raise NotImplementedError