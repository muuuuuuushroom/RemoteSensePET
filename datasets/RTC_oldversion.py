import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import glob
import json
import time
import torchvision.transforms as standard_transforms
import warnings
warnings.filterwarnings('ignore')


class RTC(Dataset):
    def __init__(self, data_root, transform=None, train=False, flip=False, patch_size=512, preload=True):
        self.root_path = data_root
        
        dataset_type = "train" if train else "test"
        data_list_path = os.path.join(data_root, dataset_type+'.txt')
        self.data_list = [name.split(' ') for name in open(data_list_path).read().splitlines()]

        self.nSamples = len(self.data_list)

        self.transform = transform
        self.train = train
        self.flip = flip
        self.patch_size = patch_size
        self.preload = preload

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

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        # load image and gt points
        img_path = os.path.join(self.root_path, self.data_list[index][0])
        gt_path = os.path.join(self.root_path, self.data_list[index][1])
        img = Image.open(img_path)
        points = self.parse_json(gt_path)
        self.img_loaded[self.data_list[index][0]] = img
        self.point_loaded[self.data_list[index][1]] = points

        # image transform
        if self.transform is not None:
            img = self.transform(img)
        img = torch.Tensor(img)

        # random scale
        # if self.train:
        #     scale_range = [0.8, 1.2]
        #     min_size = min(img.shape[1:])

        #     scale = random.uniform(*scale_range)
            
        #     # interpolation
        #     if scale * min_size > self.patch_size:
        #         img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
        #         points *= scale

        # random crop patch
        if self.train:
            img, points = self.random_crop(img, points, patch_size=self.patch_size)

        # random flip horizontal
        if random.random() > 0.5 and self.train and self.flip:
            img = torch.flip(img, dims=[2])
            points[:, 0] = self.patch_size - points[:, 0]
            
        # target
        target = {}
        target['points'] = torch.Tensor(points)
        target['labels'] = torch.ones([points.shape[0]]).long()

        if self.train:
            density = self.compute_density(points)
            target['density'] = density

        if not self.train:
            target['image_path'] = img_path

        return img, target

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
        if np.random.rand() < ratio:
            # full screen crop_window
            crop_window = np.array([0, 0, img.size(2) - patch_size, img.size(1) - patch_size], dtype=np.int64)
        else:
            # create the minimum enclosing rectangle of all points (shift by patch_size) 
            crop_window = np.array([
                max(min(points[:, 0]) - patch_size/2., 0),
                max(min(points[:, 1]) - patch_size/2., 0),
                min(max(points[:, 0]) - patch_size/2., img.size(2)- patch_size),
                min(max(points[:, 1]) - patch_size/2., img.size(1)- patch_size),
            ], dtype=np.int64)

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
        
        # resize to patchsize
        imgH, imgW = result_img.shape[-2:]
        fH, fW = patch_size/imgH, patch_size/imgW
        result_img = torch.nn.functional.interpolate(result_img.unsqueeze(0), (patch_size, patch_size)).squeeze(0)
        result_points[:, 1] *= fH
        result_points[:, 0] *= fW
        return result_img, result_points


def build_rtc(image_set, args):
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    data_root = args.data_path
    if image_set == 'train':
        train_set = RTC(data_root, train=True, transform=transform, flip=True)
        return train_set
    elif image_set == 'val':
        val_set = RTC(data_root, train=False, transform=transform)
        return val_set


if __name__ == "__main__":
    class Config():
        data_path = '/data/slcao_data/data/rice_tiller_count'
    args = Config()
    dataset = build_rtc('train', args)
    for i in range(3):
        st = time.time()
        img, target = next(iter(dataset))
        print(f'transform time: {time.time() - st}')
    
    # import time
    # dataset = build_rtc('train', args)
    # st = time.time()
    # for i in range(10):
    #     img, target = dataset[i]
    # print(f"init load: {time.time() - st}")
    # st = time.time()
    # for i in range(10):
    #     img, target = dataset[i]
    # print(f"pre load: {time.time() - st}")
