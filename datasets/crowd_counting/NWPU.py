import os
import glob
import random
import warnings
import json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as standard_transforms

warnings.filterwarnings("ignore")


class NWPU(Dataset):
    def __init__(self, data_root, transform=None, train=False, flip=False, eval_mode=False):
        self.root_path = data_root
        prefix = "train" if train else "val"
        prefix = "test" if eval_mode else prefix
        self.prefix = prefix
        self.transform = transform
        self.train = train
        self.flip = flip

        img_dir = os.path.join(data_root, prefix, "images")
        gt_dir = os.path.join(data_root, prefix, "jsons")

        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"images dir not found: {img_dir}")
        if (not eval_mode) and (not os.path.isdir(gt_dir)):
            # test split 可能没有标注；train/val 必须有
            raise FileNotFoundError(f"jsons dir not found: {gt_dir}")

        # image list
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
        img_list = []
        for e in exts:
            img_list.extend(glob.glob(os.path.join(img_dir, e)))
        self.img_list = sorted(img_list)

        # map img->gt(json). test 模式下 gt 可以为 None
        self.gt_list = {}
        for img_path in self.img_list:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            gt_path = os.path.join(gt_dir, f"{stem}.json")
            self.gt_list[img_path] = gt_path if os.path.exists(gt_path) else None

        self.img_gt_list = [(img_path, self.gt_list[img_path]) for img_path in self.img_list]

    def compute_density(self, points):
        """
        返回 shape (1,) 的 density 标量张量（更稳定，避免 N 维向量在后续广播引入歧义）
        规则：
          - N==0: density=0
          - N==1: density=0   (不要用 999 哨兵值)
          - N>=2: 取每个点到其最近邻（排除自身）的距离均值，再对所有点取均值 -> 标量
        """
        if points is None:
            return torch.tensor([0.0], dtype=torch.float32)

        if isinstance(points, np.ndarray):
            if points.shape[0] < 2:
                return torch.tensor([0.0], dtype=torch.float32)
            pts = torch.from_numpy(points.astype(np.float32, copy=False))
        elif torch.is_tensor(points):
            if points.numel() == 0 or points.shape[0] < 2:
                return torch.tensor([0.0], dtype=torch.float32, device=points.device)
            pts = points.float()
        else:
            return torch.tensor([0.0], dtype=torch.float32)

        # (N,N)
        dist = torch.cdist(pts, pts, p=2)
        # 排除自身距离 0，取每行第 2 小的作为最近邻
        nn = dist.sort(dim=1)[0][:, 1]
        density = nn.mean().reshape(1).to(dtype=torch.float32)
        return density

    def __len__(self):
        return len(self.img_gt_list)

    def __getitem__(self, index):
        img_gt_path = self.img_gt_list[index]
        img, points = load_data(img_gt_path, self.train)

        if self.transform is not None:
            img = self.transform(img)

        if self.train:
            if not torch.is_tensor(img):
                img = standard_transforms.ToTensor()(img)

            img, points = random_crop(img, points, patch_size=256)

            if self.flip:
                img, points = random_flip(img, points)

        target = {}

        # points: 统一成 torch.float32, shape (N,2)
        if isinstance(points, np.ndarray):
            points_t = torch.from_numpy(points).float()
        elif torch.is_tensor(points):
            points_t = points.float()
        else:
            points_t = torch.zeros((0, 2), dtype=torch.float32)

        target["points"] = points_t

        n = int(points_t.shape[0])
        target["labels"] = torch.ones((n,), dtype=torch.long)
        target["count"] = torch.tensor(n, dtype=torch.long)

        # train/val 都携带 image_path，便于 NaN 定位与可视化
        target["image_path"] = self.img_list[index]

        if self.train:
            # 返回标量 (1,)；空点/单点为 0，避免 999 哨兵值污染
            target["density"] = self.compute_density(points_t)

            # 可选：给个显式标记，方便 loss/criterion 分支处理
            target["empty_points"] = (n == 0)

        return img, target, None


def _read_points_json(gt_path):
    """
    NWPU json: 期望包含 {"points": [[x,y], ...], ...}
    返回 points ndarray (N,2) in (row, col) == (y, x)，以匹配 SHA/JHU 的 crop 逻辑。
    """
    if gt_path is None or (not os.path.exists(gt_path)):
        return np.zeros((0, 2), dtype=np.float32)

    with open(gt_path, "r", encoding="utf-8") as f:
        anno = json.load(f)

    pts = anno.get("points", [])
    if pts is None or len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    arr = np.asarray(pts, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)

    # (x,y) -> (y,x)
    arr = arr[:, :2][:, ::-1]
    return arr


def load_data(img_gt_path, train):
    img_path, gt_path = img_gt_path
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    points = _read_points_json(gt_path)
    return img, points


def random_crop(img, points, patch_size=256):
    patch_h = patch_size
    patch_w = patch_size

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

    result_img = img[:, start_h:end_h, start_w:end_w]
    result_points = points[idx].copy()
    if result_points.shape[0] > 0:
        result_points[:, 0] -= start_h
        result_points[:, 1] -= start_w

    return result_img, result_points


def random_flip(img, points, p=0.5):
    if random.random() > p:
        return img, points

    img = torch.flip(img, dims=[2])  # flip W

    if points is None or len(points) == 0:
        return img, points

    w = img.size(2)
    pts = points.copy()
    pts[:, 1] = (w - 1) - pts[:, 1]
    return img, pts


def build(image_set, args):
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
    ])

    data_root = args.data_path
    eval_mode = True if hasattr(args, "eval_mode") else False
    if image_set == "train":
        return NWPU(data_root, train=True, transform=transform, flip=True)
    elif image_set == "val":
        return NWPU(data_root, train=False, transform=transform, eval_mode=eval_mode)