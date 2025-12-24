import os
import glob
import random
import warnings

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as standard_transforms

warnings.filterwarnings("ignore")


class JHU(Dataset):

    def __init__(self, data_root, transform=None, train=False, flip=False, eval_mode=False):
        self.root_path = data_root
        prefix = "train" if train else "val"
        prefix = "test" if eval_mode else prefix
        self.prefix = prefix
        self.transform = transform
        self.train = train
        self.flip = flip

        # keep SHA-style flag, but also allow split override
        
        img_dir = os.path.join(data_root, prefix, "images")
        gt_dir = os.path.join(data_root, prefix, "gt")

        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"images dir not found: {img_dir}")
        if not os.path.isdir(gt_dir):
            raise FileNotFoundError(f"gt dir not found: {gt_dir}")

        # image list
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
        img_list = []
        for e in exts:
            img_list.extend(glob.glob(os.path.join(img_dir, e)))
        self.img_list = sorted(img_list)

        # get image and ground-truth list (SHA-style dict mapping)
        self.gt_list = {}
        for img_path in self.img_list:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            gt_path = os.path.join(gt_dir, f"{stem}.txt")
            self.gt_list[img_path] = gt_path

        # store pairs for easier indexing
        self.img_gt_list = [(img_path, self.gt_list[img_path]) for img_path in self.img_list]
        
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
        return len(self.img_gt_list)

    def __getitem__(self, index):
        img_gt_path = self.img_gt_list[index]
        img, points = load_data(img_gt_path, self.train)

        if self.transform is not None:
            img = self.transform(img)

        # data aug like SHA (random crop + optional flip)
        if self.train:
            # SHA 默认对 tensor 做 crop，所以这里确保 img 是 tensor (C,H,W)
            if not torch.is_tensor(img):
                img = standard_transforms.ToTensor()(img)

            img, points = random_crop(img, points, patch_size=256)

            if self.flip:
                img, points = random_flip(img, points)

        target = {}
        target["points"] = torch.from_numpy(points).float() if isinstance(points, np.ndarray) else points
        target['labels'] = torch.ones([points.shape[0]]).long()
        
        target["count"] = torch.tensor(target["points"].shape[0]).long()
        
        if self.train:
            density = self.compute_density(points)
            target['density'] = density

        if not self.train:
            target['image_path'] = self.img_list[index]

        return img, target


def _read_points_txt(gt_path):
    """
    Read JHU gt txt; returns points in (row, col) == (y, x),
    to match SHA crop code which treats points[:,0] as h(row), points[:,1] as w(col).
    """
    if gt_path is None or (not os.path.exists(gt_path)):
        return np.zeros((0, 2), dtype=np.float32)

    pts = []
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                continue
            # convert (x,y) -> (y,x) to align with SHA crop logic
            pts.append([y, x])

    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


def load_data(img_gt_path, train):
    img_path, gt_path = img_gt_path
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    points = _read_points_txt(gt_path)
    return img, points


def random_crop(img, points, patch_size=256):
    """
    Mimic SHA.random_crop:
      img: Tensor (C,H,W)
      points: ndarray (N,2) in (row, col)
    """
    patch_h = patch_size
    patch_w = patch_size

    # random crop
    start_h = random.randint(0, img.size(1) - patch_h) if img.size(1) > patch_h else 0
    start_w = random.randint(0, img.size(2) - patch_w) if img.size(2) > patch_w else 0
    end_h = start_h + patch_h
    end_w = start_w + patch_w

    if points is None or len(points) == 0:
        result_img = img[:, start_h:end_h, start_w:end_w]
        return result_img, np.zeros((0, 2), dtype=np.float32)

    idx = (
        (points[:, 0] >= start_h)
        & (points[:, 0] <= end_h)
        & (points[:, 1] >= start_w)
        & (points[:, 1] <= end_w)
    )

    # clip image and points
    result_img = img[:, start_h:end_h, start_w:end_w]
    result_points = points[idx].copy()
    if result_points.shape[0] > 0:
        result_points[:, 0] -= start_h
        result_points[:, 1] -= start_w

    return result_img, result_points


def random_flip(img, points, p=0.5):
    """
    Horizontal flip (left-right) like common crowd counting aug.

    img: Tensor (C,H,W)
    points: ndarray (N,2) in (row, col)
    """
    if random.random() > p:
        return img, points

    # flip image
    img = torch.flip(img, dims=[2])  # flip W

    if points is None or len(points) == 0:
        return img, points

    # flip points: col -> (W - 1 - col)
    w = img.size(2)
    pts = points.copy()
    pts[:, 1] = (w - 1) - pts[:, 1]
    return img, pts


def build(image_set, args):
    """
    Mimic SHA build-style helper.
    """
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
    ])
    
    data_root = args.data_path
    eval_mode = args.eval_mode
    if image_set == 'train':
        train_set = JHU(data_root, train=True, transform=transform, flip=True)
        return train_set
    elif image_set == 'val':
        val_set = JHU(data_root, train=False, transform=transform, eval_mode=eval_mode)
        return val_set
