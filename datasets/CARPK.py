import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import glob
import scipy.io as io
import torchvision.transforms as standard_transforms
import warnings
import json
import xml.etree.ElementTree as ET

import torchvision.transforms.functional as TF
import math

warnings.filterwarnings("ignore")


class SHA(Dataset):
    def __init__(
        self,
        data_root,
        transform=None,
        train=False,
        flip=False,
        global_crop_ratio=0.1,
        probs=[0.05, 0.5, 1e-4],
        total_steps=1000,
        augmented=False,
        category=None,
        args=None
    ):
        self.root_path = data_root

        prefix = "train_data" if train else "test_data"
        # prefix = "train_data"
        self.prefix = prefix

        aug = "augmented_selected" if augmented else "images"

        # self.img_list = os.listdir(f"{data_root}/{prefix}/images")
        self.img_list = os.listdir(f"{data_root}/{prefix}/{aug}")
        # print(f'dataset images using: {data_root}/{prefix}/{aug}')

        # get image and ground-truth list
        self.gt_list = {}
        for img_name in self.img_list:
            img_path = f"{data_root}/{prefix}/{aug}/{img_name}"
            gt_path = f"{data_root}/{prefix}/VGG_anotation_truth/{img_name}"
            file_name, extension = os.path.splitext(img_path)
            if extension == ".jpg":
                self.gt_list[img_path] = gt_path.replace("jpg", "xml")
            elif extension == ".png":
                self.gt_list[img_path] = gt_path.replace("png", "xml")
        self.img_list = sorted(list(self.gt_list.keys()))
        self.nSamples = len(self.img_list)

        self.transform = transform
        self.train = train
        self.flip = flip
        self.patch_size = 256
        
        self.category=category
        self.test_str='padding' 
        # 'resize'
        # =args.test_str
        
        self.test_robust=None #'direction'  # dense, scale
        self.seed=42

        # training process
        if isinstance(global_crop_ratio, str):
            self.global_sample_ratio = 0.1
            self.crop_ratio_type = "dynamic"
        else:
            print("static global crop ratio")
            self.crop_ratio_type = "staic"
            self.global_sample_ratio = global_crop_ratio

        if self.train and isinstance(global_crop_ratio, str):
            self.prob_schedule = self.re_onecycle_prob(
                probs[0], probs[1], probs[2], total_steps
            )
            self.global_sample_ratio = self.prob_schedule[0]

    def compute_density(self, points):
        """
        Compute crowd density:
            - defined as the average nearest distance between ground-truth points
        """
        points_tensor = torch.from_numpy(points.copy())
        dist = torch.cdist(points_tensor, points_tensor, p=2)
        if points_tensor.shape[0] > 1:
            density = dist.sort(dim=1)[0][:, 1].mean().reshape(-1)
        else:
            density = torch.tensor(999.0).reshape(-1)
        return density

    def set_epoch(self, epoch):
        """
        change the probility of global crop during every epochs
        """
        if self.train and self.crop_ratio_type == "dynamic":
            self.global_sample_ratio = self.prob_schedule[epoch]
        print(f"global crop ratio = {self.global_sample_ratio}")

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), "index range error"

        # load image and gt points
        img_path = self.img_list[index]
        gt_path = self.gt_list[img_path]
        img, points, bboxs = load_data((img_path, gt_path), self.train)
        points = points.astype(float)
        bboxs = bboxs.astype(float)
        # image transform
        if self.transform is not None:
            img = self.transform(img)
        img = torch.Tensor(img)

        if self.train:
            scale_range = [0.3, 0.7]
            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)

            # interpolation
            if scale * min_size > self.patch_size:
                img = torch.nn.functional.upsample_bilinear(
                    img.unsqueeze(0), scale_factor=scale
                ).squeeze(0)
                points *= scale
                bboxs *= scale
                
        img_ = img.numpy()
        img_ = np.transpose(img_, (1, 2, 0))
        if self.train:
            img, points, bboxs = random_crop(
                img, points, bboxs, patch_size=self.patch_size, category=self.category
            )
        else:
            if self.test_str == 'padding':
                # img, points, bboxs = resize_padding(img, points, bboxs, self.category)
                img, points, bboxs = resize_with_padding_center(img, points, bboxs, self.test_robust, index)
            elif self.test_str == 'resize':
                img, points, bboxs = resize(img, points, bboxs, self.category)
                

        if random.random() > 0.5 and self.train and self.flip:
            img = torch.flip(img, dims=[2])
            points[:, 1] = self.patch_size - points[:, 1]

        # target
        target = {}
        target["points"] = torch.Tensor(points)
        target["bboxs"] = torch.Tensor(bboxs)
        target["labels"] = torch.ones([points.shape[0]]).long()

        if self.train:
            density = self.compute_density(points)
            target["density"] = density

        if not self.train:
            target["image_path"] = img_path

        return img, target


def load_data(img_gt_path, train):
    img_path, gt_path = img_gt_path
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    tree = ET.parse(gt_path)
    root = tree.getroot()
    elements = root.findall("object")
    points = []
    bboxs = []
    for element in elements:
        x = int(element.findall("point_2d")[0].find("center_x").text)
        y = int(element.findall("point_2d")[0].find("center_y").text)

        xmin = int(element.findall("bndbox")[0].find("xmin").text)
        ymin = int(element.findall("bndbox")[0].find("ymin").text)
        xmax = int(element.findall("bndbox")[0].find("xmax").text)
        ymax = int(element.findall("bndbox")[0].find("ymax").text)
        points.append([y, x])
        bboxs.append([xmin, ymin, xmax, ymax])
    points = np.array(points)
    bboxs = np.array(bboxs)
    return img, points, bboxs


def random_crop(img, points, bboxs, patch_size=256, category=None):
    patch_h = patch_size
    patch_w = patch_size

    # random crop
    start_h = random.randint(0, img.size(1) - patch_h) if img.size(1) > patch_h else 0
    start_w = random.randint(0, img.size(2) - patch_w) if img.size(2) > patch_w else 0
    end_h = start_h + patch_h
    end_w = start_w + patch_w
    idx = (
        (points[:, 0] >= start_h)
        & (points[:, 0] <= end_h)
        & (points[:, 1] >= start_w)
        & (points[:, 1] <= end_w)
    )

    # clip image and points
    result_img = img[:, start_h:end_h, start_w:end_w]
    result_points = points[idx]
    result_bboxs = bboxs[idx]
    result_points[:, 0] -= start_h
    result_points[:, 1] -= start_w
    result_bboxs[:, 0] -= start_w
    result_bboxs[:, 1] -= start_h
    result_bboxs[:, 2] -= start_w
    result_bboxs[:, 3] -= start_h

    # resize to patchsize
    imgH, imgW = result_img.shape[-2:]
    fH, fW = patch_h / imgH, patch_w / imgW
    result_img = torch.nn.functional.interpolate(
        result_img.unsqueeze(0), (patch_h, patch_w)
    ).squeeze(0)
    result_points[:, 0] *= fH
    result_points[:, 1] *= fW
    result_bboxs[:, 0] *= fW
    result_bboxs[:, 1] *= fH
    result_bboxs[:, 2] *= fW
    result_bboxs[:, 3] *= fH
    return result_img, result_points, result_bboxs


def get_patch_size(img_h, img_w):
    '''
        return min 128 common multiple
    '''
    patch_h = math.ceil(img_h / 128) * 128
    patch_w = math.ceil(img_w / 128) * 128

    return patch_h, patch_w

def resize(img, points, bboxs, category):
    # resize to patchsize
    imgH, imgW = img.shape[-2:]
    
    # patch_h = 512 if category == 'Ship' else 1024
    # patch_w = 1024
    
    # rewrite the logic for eval patch size dynamic scaling:
    patch_h, patch_w = get_patch_size(imgH, imgW)
    
    fH, fW = patch_h / imgH, patch_w / imgW
    result_img = torch.nn.functional.interpolate(
        img.unsqueeze(0), (patch_h, patch_w)
    ).squeeze(0)
    result_bboxs = bboxs.copy()
    result_points = points.copy()
    result_points[:, 0] *= fH
    result_points[:, 1] *= fW
    result_bboxs[:, 0] *= fW
    result_bboxs[:, 1] *= fH
    result_bboxs[:, 2] *= fW
    result_bboxs[:, 3] *= fH
    return result_img, result_points, result_bboxs

def resize_padding(img, points, bboxs, category):
    imgH, imgW = img.shape[-2:]
    patch_h, patch_w = get_patch_size(imgH, imgW)

    pad_h = patch_h - imgH
    pad_w = patch_w - imgW

    # 只pad下边和右边
    padding = (0, pad_w, 0, pad_h)

    if img.max() > 1.5:  # 如果是0~255整数图像
        pad_value = 255
    else:                # 如果是0~1浮点图像
        pad_value = 1.0

    result_img = torch.nn.functional.pad(img, padding, value=pad_value)
    result_points = points.copy()
    result_bboxs = bboxs.copy()

    return result_img, result_points, result_bboxs

def resize_with_padding_center(img, points, bboxs, test_robust, index):

    if test_robust != None:
        # augmented for evaluating robustness of the model 
        # direction
        random.seed(42 + index)
        angle = random.uniform(-30, 30)
        rotated_img = TF.rotate(img, angle, expand=True)
        
        # TF.affine(
        #         img, 
        #         angle=angle, 
        #         translate=[0, 0], 
        #         scale=1.0, 
        #         shear=[0, 0],
        #         fill=(255, 255, 255) 
        #     )
        # TF.rotate(img, angle, expand=True)
        
        # error happens for the localization of GT points due to the resolution change in TF.rotate as it expands
        # this does not happen performing affine as it cuts the img, remaining the centre
        # it is finished. for pytorch uses down-direction y-axis, we have to rotate '-angle' for points and bboxs
        
        oldH, oldW = img.shape[-2:]
        old_cx, old_cy = oldW / 2, oldH / 2
        newH, newW = rotated_img.shape[-2:]
        new_cx, new_cy = newW / 2, newH / 2

        offset_x = new_cx - old_cx
        offset_y = new_cy - old_cy
        
        angle_rad = math.radians(-angle) # evil pytorch
        img = rotated_img
        
        if points is not None and len(points) > 0:
            rotated_points = []
            for p in points:
                y, x = p[0], p[1]
                x_rot = math.cos(angle_rad) * (x - old_cx) - math.sin(angle_rad) * (y - old_cy) + old_cx + offset_x
                y_rot = math.sin(angle_rad) * (x - old_cx) + math.cos(angle_rad) * (y - old_cy) + old_cy + offset_y
                rotated_points.append([y_rot, x_rot])
            result_points = torch.tensor(rotated_points, dtype=torch.float32).numpy()
        points = result_points
            
        if bboxs is not None and len(bboxs) > 0:
            rotated_bboxs = []
            for box in bboxs:
                x1, y1, x2, y2 = box
                corners = [
                    (y1, x1),
                    (y1, x2),
                    (y2, x2),
                    (y2, x1),
                ]
                rotated_corners = []
                for y, x in corners:
                    x_rot = math.cos(angle_rad) * (x - old_cx) - math.sin(angle_rad) * (y - old_cy) + old_cx + offset_x
                    y_rot = math.sin(angle_rad) * (x - old_cx) + math.cos(angle_rad) * (y - old_cy) + old_cy + offset_y
                    rotated_corners.append([x_rot, y_rot])
                rotated_corners = torch.tensor(rotated_corners, dtype=torch.float32)
                x_min, y_min = rotated_corners.min(dim=0).values
                x_max, y_max = rotated_corners.max(dim=0).values
                rotated_bboxs.append([x_min.item(), y_min.item(), x_max.item(), y_max.item()])
            result_bboxs = torch.tensor(rotated_bboxs, dtype=torch.float32).numpy()
        bboxs = result_bboxs
        
        
    imgH, imgW = img.shape[-2:]
    patch_h, patch_w = get_patch_size(imgH, imgW)

    pad_h_total = patch_h - imgH
    pad_w_total = patch_w - imgW

    pad_top = pad_h_total // 2
    pad_bottom = pad_h_total - pad_top
    pad_left = pad_w_total // 2
    pad_right = pad_w_total - pad_left

    padding = (pad_left, pad_right, pad_top, pad_bottom)
    pad_value = 255
    # if img.max() > 1.5: 
    #     pad_value = 255
    # else:     
    #     pad_value = 1.0

    result_img = torch.nn.functional.pad(img, padding, value=pad_value)
    result_points = points.copy()
    result_bboxs = bboxs.copy()

    if points is not None and len(points) > 0:
        result_points[:, 0] += pad_top    # y
        result_points[:, 1] += pad_left   # x

    if bboxs is not None and len(bboxs) > 0:
        result_bboxs[:, 0] += pad_left    # x1
        result_bboxs[:, 1] += pad_top     # y1
        result_bboxs[:, 2] += pad_left    # x2
        result_bboxs[:, 3] += pad_top     # y2
        
    return result_img, result_points, result_bboxs


def build(image_set, args):
    transform = standard_transforms.Compose(
        [
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    data_root = args.data_path
    if image_set == "train":
        train_set = SHA(
            data_root,
            train=True,
            transform=transform,
            flip=True,
            global_crop_ratio=args.global_crop_ratio,
            total_steps=args.epochs,
            augmented=args.augmented,
            category=args.dataset_file,
            args=args
        )
        return train_set
    elif image_set == "val":
        val_set = SHA(
            data_root,
            train=False,
            transform=transform,
            global_crop_ratio=args.global_crop_ratio,
            total_steps=args.epochs,
            augmented=args.augmented,
            category=args.dataset_file,
            args=args
        )
        return val_set
