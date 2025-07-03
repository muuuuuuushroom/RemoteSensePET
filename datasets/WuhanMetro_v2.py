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
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

import sys 
sys.path.append('./')
sys.path.append('./models')
import util.misc as utils

class WuhanMetro(Dataset):
    def __init__(
        self, 
        data_root, 
        transform=None,
        train=False,
        flip=False,
        patch_size=256, 
        preload=False, 
        global_crop_ratio=0.1,
        probs=[0.05, 0.5, 1e-4],
        total_steps=1500,
    ):
        super().__init__()
        self.root_path = data_root
        
        dataset_type = "train" if train else "test"
        data_list_path = os.path.join(data_root, dataset_type+'.txt')
        # self.data_list = [name.split(' ') for name in open(data_list_path).read().splitlines()]
        self.img_list = [os.path.join(dataset_type, name) for name in open(data_list_path).read().splitlines()]
        self.ann_list = [os.path.join('json', name.replace('.jpg', '.json')) for name in open(data_list_path).read().splitlines()]
        self.nSamples = len(self.img_list)

        self.transform = transform
        self.train = train
        self.flip = flip
        self.patch_size = patch_size
        self.preload = preload  # no using

        if isinstance(global_crop_ratio, str):
            self.global_sample_ratio = 0.1
            self.crop_ratio_type = 'dynamic'
        else:
            # print('static global crop ratio')
            self.crop_ratio_type = 'staic'
            self.global_sample_ratio = global_crop_ratio
        
        if self.train and isinstance(global_crop_ratio, str):
            self.prob_schedule = self.re_onecycle_prob(probs[0], probs[1], probs[2], total_steps)
            self.global_sample_ratio = self.prob_schedule[0]

        self.img_loaded = {}
        self.point_loaded = {}
            
    
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
        img_path = os.path.join(self.root_path, self.img_list[index])
        gt_path = os.path.join(self.root_path, self.ann_list[index])
        
        img = cv2.imread(img_path)  #; img = np.array(img).transpose(2,0,1).astype(np.float32);
        points, masks = self.parse_json_mask(gt_path)
        masks = torch.as_tensor(masks)
        if masks is not None and masks.numel() > 0:  
            for x1, y1, x2, y2 in masks.long():    
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                img[y1:y2, x1:x2, :] = 255
        
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 
        if self.transform is not None:
            img = self.transform(img)
        img = torch.Tensor(img)
        
        # points = self.parse_json(gt_path)

        if self.train:
            scale_range = [0.6, 1.8] #[0.6, 1.8] # [0.5, 0.8]
            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)
            
            # interpolation
            if scale * min_size > self.patch_size:  
                img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
                points *= scale
                masks *= scale
        
        # random crop patch
        if self.train:
            img, points = self.random_crop(img, points, patch_size=self.patch_size)
            
        # random flip horizontal
        if random.random() > 0.5 and self.train and self.flip:
            img = torch.flip(img, dims=[2])
            points[:, 0] = self.patch_size - points[:, 0]

        # target
        target = {}
        target['points'] = torch.from_numpy(points[:,::-1].copy()) 
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
        # points[:, [0, 1]] = points[:, [1, 0]]

        return points
    
    @staticmethod
    def parse_json_mask(gt_path):
        with open(gt_path, 'r') as f:
            data = json.load(f)

        pts_list, msk_list = [], []
        for shape in data.get('shapes', []):
            label = shape.get('label', '')
            stype = shape.get('shape_type', '')
            coords_arr = np.asarray(shape['points'], dtype=np.float32)  # (K, 2)

            if stype == 'point' or label == 'pedestrian':
                pts_list.append(coords_arr[0])               # (2,)
                
            elif label == 'mask' or stype == 'rectangle':
                x1, y1 = coords_arr.min(0)
                x2, y2 = coords_arr.max(0)
                msk_list.append([x1, y1, x2, y2])

        points = np.asarray(pts_list, dtype=np.float32)       # (N_pts, 2)
        masks  = np.asarray(msk_list, dtype=np.float32)       # (N_masks, 4)

        return points, masks

    @staticmethod
    def random_crop(img, points, patch_size: int = 256, ratio: float = 0.1):
        H, W = img.size(1), img.size(2)          # img 是 Tensor(C,H,W)

        # 0⃣ —— 若没有任何点，退化成全图随机 -------------------------------
        has_points = points is not None and points.size > 0

        # 1⃣ —— 计算 crop_window = [x_min, y_min, x_max, y_max] ----------
        if not has_points or np.random.rand() < ratio:
            crop_window = np.array([0, 0,
                                    max(W - patch_size, 0),
                                    max(H - patch_size, 0)])
        else:
            x_min = max(points[:, 0].min() - patch_size / 2, 0)
            y_min = max(points[:, 1].min() - patch_size / 2, 0)
            x_max = min(points[:, 0].max() - patch_size / 2, W - patch_size)
            y_max = min(points[:, 1].max() - patch_size / 2, H - patch_size)
            crop_window = np.array([x_min, y_min, x_max, y_max])

        crop_window = crop_window.astype(np.int64)

        # 2⃣ —— 若可裁尺寸不足 patch_size，直接从左上角取 -------------------
        if crop_window[2] < crop_window[0] or crop_window[3] < crop_window[1]:
            start_w = 0
            start_h = 0
        else:
            # randint(low, high) 要求 high ≥ low
            start_w = random.randint(crop_window[0],
                                     max(crop_window[0], crop_window[2]))
            start_h = random.randint(crop_window[1],
                                     max(crop_window[1], crop_window[3]))

        end_w, end_h = start_w + patch_size, start_h + patch_size

        # 3⃣ —— 裁剪图像 & 点 --------------------------------------------
        result_img = img[:, start_h:end_h, start_w:end_w]

        if has_points:
            idx = ((points[:, 0] >= start_w) & (points[:, 0] <= end_w) &
                   (points[:, 1] >= start_h) & (points[:, 1] <= end_h))
            result_points = points[idx].copy()
            result_points[:, 0] -= start_w
            result_points[:, 1] -= start_h
        else:
            result_points = np.empty((0, 2), dtype=np.float32)

        # 4⃣ —— 若裁后尺寸仍不足 patch_size，插值放大 ----------------------
        if min(result_img.shape[-2:]) < patch_size:
            imgH, imgW = result_img.shape[-2:]
            fH, fW = patch_size / imgH, patch_size / imgW
            result_img = F.interpolate(result_img.unsqueeze(0),
                                       size=(patch_size, patch_size),
                                       mode="bilinear",
                                       align_corners=False).squeeze(0)
            if result_points.size > 0:
                result_points[:, 0] *= fW
                result_points[:, 1] *= fH

        return result_img, result_points

def build_whm(image_set, args):
    data_root = args.data_path
    # args.patch_size = 
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]),
    ])
    if image_set == 'train':
        train_set = WuhanMetro(data_root, train=True, 
                               transform=transform, 
                               patch_size=args.patch_size, 
                               flip=True, 
                               total_steps=args.epochs)
        return train_set
    elif image_set == 'val':
        val_set = WuhanMetro(data_root, train=False, 
                             transform=transform, 
                             patch_size=args.patch_size)
        return val_set
    else:
        raise NotImplementedError


if __name__ == "__main__":
    class Config():
        data_path = '/data/zlt/RSPET/PET/data/WuhanMetroCount'
        patch_size = 256
        global_crop_ratio='dynamic'
        epochs = 300
    args = Config()
    dataset = build_whm('train', args)
    data_loader_train = DataLoader(dataset, batch_size=8, collate_fn=utils.collate_fn, num_workers=2)

    for i in range(10):
        st = time.time()
        img, target = next(iter(dataset))
        img = img.numpy().transpose(1,2,0)
        print(img.shape)
        print(target['points'])
        # for point in target['points']:
        #     cv2.circle(img, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)
        # cv2.imwrite('./test.png', img)
        # print(f"get item time: {time.time()-st}")
