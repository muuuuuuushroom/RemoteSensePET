import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import glob
import scipy
import scipy.io as io
import torchvision.transforms as standard_transforms
import warnings

import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree

warnings.filterwarnings('ignore')

class SHA(Dataset):
    def __init__(self, data_root, transform=None, train=False, flip=False, 
                 prob_map_lc=None, patch_size=256):
        self.root_path = data_root
        
        prefix = "train_data" if train else "test_data"
        self.prefix = prefix
        self.img_list = os.listdir(f"{data_root}/{prefix}/images")

        # get image and ground-truth list
        self.gt_list = {}
        for img_name in self.img_list:
            img_path = f"{data_root}/{prefix}/images/{img_name}"  
            gt_path = f"{data_root}/{prefix}/ground_truth/GT_{img_name}"
            self.gt_list[img_path] = gt_path.replace("jpg", "mat")
        self.img_list = sorted(list(self.gt_list.keys()))
        self.nSamples = len(self.img_list)

        self.transform = transform
        self.train = train
        self.flip = flip
        self.patch_size = patch_size
        
        self.prob_map_lc = prob_map_lc
    
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
        img_path = self.img_list[index]
        gt_path = self.gt_list[img_path]
        img, points = load_data((img_path, gt_path), self.train)
        points = points.astype(float)

        # image transform
        if self.transform is not None:
            img = self.transform(img)
        img = torch.Tensor(img)

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
            img, points = random_crop(img, points, patch_size=self.patch_size)

        # random flip
        if random.random() > 0.5 and self.train and self.flip:
            img = torch.flip(img, dims=[2])
            points[:, 1] = self.patch_size - points[:, 1]

        # target
        target = {}
        target['points'] = torch.Tensor(points)
        target['labels'] = torch.ones([points.shape[0]]).long()

        if self.train:
            density = self.compute_density(points)
            target['density'] = density

        if not self.train:
            target['image_path'] = img_path
            
        # if self.prob_map_lc == 'f4x' and self.train:
        #     prob = generate_prob_map_from_points(target['points'], self.patch_size, self.patch_size)
        # else:
        #     prob = None
        prob = None
        return img, target, prob


def load_data(img_gt_path, train):
    img_path, gt_path = img_gt_path
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    points = io.loadmat(gt_path)['image_info'][0][0][0][0][0][:,::-1]
    return img, points


def random_crop(img, points, patch_size=256):
    patch_h = patch_size
    patch_w = patch_size
    
    # random crop
    start_h = random.randint(0, img.size(1) - patch_h) if img.size(1) > patch_h else 0
    start_w = random.randint(0, img.size(2) - patch_w) if img.size(2) > patch_w else 0
    end_h = start_h + patch_h
    end_w = start_w + patch_w
    idx = (points[:, 0] >= start_h) & (points[:, 0] <= end_h) & (points[:, 1] >= start_w) & (points[:, 1] <= end_w)

    # clip image and points
    result_img = img[:, start_h:end_h, start_w:end_w]
    result_points = points[idx]
    result_points[:, 0] -= start_h
    result_points[:, 1] -= start_w
    
    # resize to patchsize
    imgH, imgW = result_img.shape[-2:]
    fH, fW = patch_h/imgH, patch_w/imgW
    result_img = torch.nn.functional.interpolate(result_img.unsqueeze(0), (patch_h, patch_w)).squeeze(0)
    result_points[:, 0] *= fH
    result_points[:, 1] *= fW
    return result_img, result_points

def generate_prob_map_from_points(points, img_h, img_w, device='cuda', alpha=0.4):
    """
    Generate a probability map using a list of points.
    
    Args:
        points (numpy.ndarray): Array of points with shape (N, 2), where N is the number of points.
        img_h (int): Height of the image.
        img_w (int): Width of the image.
        device (str): The device to run the computation on ('cuda' or 'cpu').
        alpha (float): A scaling factor for the sigma value.

    Returns:
        torch.Tensor: The generated probability map of shape (1, img_h, img_w).
    """
    # Ensure points are valid
    if len(points) == 0:
        return torch.zeros((1, img_h, img_w), dtype=torch.float32, device=device)

    # Convert points to a numpy array
    pts = np.array(points)

    # Create a KDTree to compute distances between points
    tree = KDTree(pts)
    distances, locations = tree.query(pts, k=4)

    # Initialize an empty density map
    density = torch.zeros((1, 1, img_h, img_w), dtype=torch.float32, device=device)

    for i, pt in enumerate(pts):
        x, y = pt

        # Create a 2D map with a single point set to 1
        pt2d = torch.zeros((1, 1, img_h, img_w), dtype=torch.float32, device=device)
        pt2d[0, 0, y, x] = 1.0

        # Dynamically calculate sigma based on neighbor distances
        if len(distances[i]) >= 2 and np.isfinite(distances[i][1]):
            di = distances[i][1]
            neighbor_idx = locations[i][1:]
            neighbor_distances = []
            for idx in neighbor_idx:
                if np.isfinite(distances[idx][1]):
                    neighbor_distances.append(distances[idx][1])

            if len(neighbor_distances) > 0:
                d_mtop3 = np.mean(neighbor_distances)
                d = min(di, d_mtop3)
            else:
                d = di  # fallback
            
            sigma = alpha * d
        else:
            sigma = np.average(np.array([img_h, img_w])) / 4.0  # fallback

        sigma = max(sigma, 1.0)

        # Generate a Gaussian kernel based on sigma
        kernel_size = int(6 * sigma)
        if kernel_size % 2 == 0:
            kernel_size += 1

        gaussian_kernel = generate_gaussian_kernel(kernel_size, sigma, device)
        gaussian_kernel = gaussian_kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, k, k]

        # Apply Gaussian filter to the point map
        filter = F.conv2d(pt2d, gaussian_kernel, padding=kernel_size // 2)

        # Normalize the filter to avoid numerical instability
        peak = filter[0, 0, y, x]
        if peak > 0:
            filter = filter / peak

        # Update the density map with the current filter
        density = torch.maximum(density, filter)

    return density.squeeze()

def generate_gaussian_kernel(kernel_size, sigma, device='cuda'):
    """
    Generate a Gaussian kernel for convolution.
    
    Args:
        kernel_size (int): The size of the kernel.
        sigma (float): The standard deviation of the Gaussian.
        device (str): The device to run the computation on ('cuda' or 'cpu').
        
    Returns:
        torch.Tensor: The generated Gaussian kernel.
    """
    x = torch.arange(kernel_size, device=device) - kernel_size // 2
    y = torch.arange(kernel_size, device=device) - kernel_size // 2
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel



def build(image_set, args):
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]),
    ])
    
    data_root = args.data_path
    if image_set == 'train':
        train_set = SHA(data_root, train=True, transform=transform, flip=True, 
                        prob_map_lc=args.prob_map_lc, patch_size=args.patch_size)
        return train_set
    elif image_set == 'val':
        val_set = SHA(data_root, train=False, transform=transform)
        return val_set
