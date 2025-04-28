import os  # 用于与文件系统交互
import random  # 用于生成随机数
import torch  # 导入PyTorch，一个深度学习框架
import numpy as np  # 导入NumPy库，用于数值计算
from torch.utils.data import Dataset  # 从PyTorch中导入Dataset类，用于数据加载
from PIL import Image  # 从PIL库导入Image模块，用于图像操作
import cv2  # 导入OpenCV库，用于图像处理
import glob  # 用于文件路径名模式匹配
import json #用于处理json文件
import gc
import scipy.io as io  # 用于加载.mat文件
import torchvision.transforms as standard_transforms  # 导入PyTorch视觉转换
from torchvision import transforms 
import warnings  # 用于警告控制
import numpy as np 
import matplotlib.pyplot as plt  
warnings.filterwarnings('ignore')  # 忽略警告

# 定义CORN数据集类
class CORN(Dataset):
    def __init__(self, data_root, transform=None, train=False, flip=False, rotate=True):#):
        self.root_path = data_root  # 数据集根目录
        
        prefix = "train" if train else "test"  # 根据是训练集还是测试集选择前缀
        self.prefix = prefix
        self.list = os.listdir(f"{data_root}/{prefix}/images")  # 获取图片列表
        self.gt_list = {}
        txt_path = f"{data_root}/{prefix}/annotations"##json文件夹的地址
        # 创建一个集合来存储所有.json文件的基础文件名（无扩展名）filename[:-5]这个意思是:-5 表示切片的范围从字符串的开始直到倒数第五个字符。这里 -5 表示从字符串末尾倒数第五个字符的位置。
        ##因此，如果 filename 是 "example.json"，那么 filename[:-5] 的结果将是 "example"。
        txt_files = {filename[:-4] for filename in os.listdir(txt_path) if filename.endswith('.txt')}
        #print(txt_files)
        for img_name in self.list:
            # 假设img_name是“example.jpg”，则base_name是“example”
            base_name =os.path.splitext(img_name)[0]##得到的base_name 是图像名，例如：IMG_498
            if base_name in txt_files:##txt_files是IMG_498等路径
            # 如果基础文件名在json_files集合中，构建图片和JSON文件的完整路径
                path = f"{data_root}/{prefix}/images/{img_name}"
                gt_path = os.path.join(txt_path, f"{base_name}.txt")
                self.gt_list[path] = gt_path####往创建的空字典中插入新的键值对
          
        self.list = sorted(list(self.gt_list.keys()))  # 排序图片列表
        self.nSamples = len(self.list)  # 图片数量
        self.transform = transform  # 图像转换函数
        self.train = train  # 是否是训练集
        self.flip = flip  # 是否翻转图片
        self.patch_size = 256  # 图像块的尺寸
        self.rotate = rotate   #图像是否随机旋转
    # 计算人群密度
    def compute_density(self, points):
        """
        计算人群密度：定义为地面真实点之间最近距离的平均值
        """
        points_tensor = torch.from_numpy(points.copy())
        dist = torch.cdist(points_tensor, points_tensor, p=2)
        if points_tensor.shape[0] > 1:
            density = dist.sort(dim=1)[0][:,1].mean().reshape(-1)
        else:
            density = torch.tensor(999.0).reshape(-1)
        return density


    def __len__(self):
        return self.nSamples  # 返回数据集中的样本数量

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'  # 确保索引在有效范围内
        # 用于存储所有点信息的列表  
       
        # 加载图片和地面真实点
        path = self.list[index]
        gt_path = self.gt_list[path]
        # img, points = load_data((path, gt_path), self.train)#调用函数
        # ######学长
        # img = np.array(img)
        # h, w, _ = img.shape
        # if h < 768 and w < 1024:
        #     img = np.pad(img, ((0,768-h),(0,1024-w),(0,0)),'constant',constant_values=0)
        # else:
        #     img = cv2.resize(img, (0,0), None, min(1024/w,768/h), min(1024/w,768/h))
        #     new_h, new_w, _ = img.shape
        #     img = np.pad(img, ((0,768-new_h),(0,1024-new_w),(0,0)),'constant',constant_values=0)
        # points = points.astype(float)
        img, points = load_data((path, gt_path), self.train)  # 调用函数
        img = np.array(img)
        h, w, _ = img.shape

        # # 计算填充和缩放比例
        # if h < 768 and w < 1024:
        #     # 填充操作
        #     img = np.pad(img, ((0, 768 - h), (0, 1024 - w), (0, 0)), 'constant', constant_values=0)

        # else:
        #     scale = min(1024 / w, 768 / h)  # 按比例缩放，w和h分别是原图的宽度和高度
        #     img = cv2.resize(img, (0, 0), None, scale, scale)

        #     # 获取缩放后图像的新的高度和宽度
        #     new_h, new_w, _ = img.shape  # new_h 是缩放后的高度，new_w 是缩放后的宽度，_ 代表通道数，暂时不需要
        #     img = np.pad(img, ((0, 768 - new_h), (0, 1024 - new_w), (0, 0)), 'constant', constant_values=0)

        #     # 计算缩放比例并调整点的坐标
        #     points[:, 0] = points[:, 0] * scale  # 水平坐标按缩放比例调整
        #     points[:, 1] = points[:, 1] * scale  # 垂直坐标按缩放比例调整
                # new strategy
        target_h = ((h + 127) // 128) * 128
        target_w = ((w + 127) // 128) * 128
        scale_h = target_h / h
        scale_w = target_w / w
        
        img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        if points is not None and len(points) > 0:
            points_rescaled = points.copy()
            points_rescaled[:, 0] *= scale_h
            points_rescaled[:, 1] *= scale_w 
        else:
            points_rescaled = points
            
        img = img_resized
        points = points_rescaled

        # 图像转换S
        if self.transform is not None:
            img = self.transform(img)
        img = torch.Tensor(img)
        # 随机缩放
        if self.train:
            scale_range = [0.8, 1.2]           
            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)
            # 插值
            if scale * min_size > self.patch_size:  
                img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
                points *= scale
        # 随机裁剪图像块
        if self.train:
            img, points = random_crop(img, points, patch_size=self.patch_size)
        # 随机翻转
        if random.random() > 0.5 and self.train and self.flip:
            img = torch.flip(img, dims=[2])
            points[:, 1] = self.patch_size - points[:, 1]
        # # 随机旋转
        # ''' if self.train and self.rotate:
        #     angle = random.uniform(-180, 180)
        #     img, points = rotate_image_and_points(img, points, angle)
        # '''
        # 目标
        target = {}
        target['points'] = torch.Tensor(points)
        target['labels'] = torch.ones([points.shape[0]]).long()

        if self.train:
            density = self.compute_density(points)
            target['density'] = density

        if not self.train:
            target['image_path'] = path

        return img, target  # 返回图像和目标

# 加载数据函数
def load_data(img_gt_path, train):
    path, gt_path = img_gt_path##元组使用
    #print(path)
    img = cv2.imread(path)  # 使用cv2读取图像
    if img is None:
            raise ValueError(f"无法加载图像：{path}")
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 转换颜色空间并使用PIL图像
    with open(gt_path, 'r',encoding="utf-8") as f:  
        point=[]
        data = f.readlines()
        for line in data:
            line = line.strip('\n')
            x, y = map(float, line.split(','))
            point.append([x, y])
        points = np.array(point)  
    return img, points
   

# 随机裁剪函数
def random_crop(img, points, patch_size=256):
    patch_h = patch_size
    patch_w = patch_size
    
    # 随机裁剪
    start_h = random.randint(0, img.size(1) - patch_h) if img.size(1) > patch_h else 0
    start_w = random.randint(0, img.size(2) - patch_w) if img.size(2) > patch_w else 0
    end_h = start_h + patch_h
    end_w = start_w + patch_w

    if points.ndim == 1:
        points = points.reshape(-1, 2)  # 假设每个点是2D的

    idx = (points[:, 0] >= start_h) & (points[:, 0] <= end_h) & (points[:, 1] >= start_w) & (points[:, 1] <= end_w)

    # 裁剪图像和点
    result_img = img[:, start_h:end_h, start_w:end_w]
    result_points = points[idx]
    result_points[:, 0] -= start_h
    result_points[:, 1] -= start_w
    
    # 调整大小至patch大小
    imgH, imgW = result_img.shape[-2:]
    fH, fW = patch_h/imgH, patch_w/imgW
    result_img = torch.nn.functional.interpolate(result_img.unsqueeze(0), (patch_h, patch_w)).squeeze(0)
    result_points[:, 0] *= fH
    result_points[:, 1] *= fW
    return result_img, result_points

def rotate_point(cx, cy, angle, px, py):
    # 将角度从度转换为弧度
    angle_rad = np.radians(angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    # 将点(px, py)围绕点(cx, cy)旋转
    qx = cos_angle * (px - cx) - sin_angle * (py - cy) + cx
    qy = sin_angle * (px - cx) + cos_angle * (py - cy) + cy
    return qx, qy

def rotate_image_and_points(image, points, angle):
    _, h, w = image.shape
    center = (w / 2, h / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    image_np = image.permute(1, 2, 0).numpy()
    rotated_image = cv2.warpAffine(image_np, rotation_matrix, (w, h))

    ones = np.ones(shape=(len(points), 1))
    points_homogeneous = np.hstack([points, ones])
    rotated_points = rotation_matrix.dot(points_homogeneous.T).T

    # 过滤掉不在图像边界内的点
    mask = (rotated_points[:, 0] >= 0) & (rotated_points[:, 0] < w) & (rotated_points[:, 1] >= 0) & (rotated_points[:, 1] < h)
    rotated_points = rotated_points[mask]
    rotated_image_tensor = torch.Tensor(rotated_image).permute(2, 0, 1) / 255.0
    return rotated_image_tensor, rotated_points

   
# 构建数据集函数
def build(image_set, args):
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),  # 将图片转换为Tensor
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 正规化图片
                                      std=[0.229, 0.224, 0.225]),
    ])
    
    data_root = args.data_path  # 数据集根路径
    if image_set == 'train':
        train_set = CORN(data_root, train=True, transform=transform, flip=True)  # 创建训练集
        return train_set
    elif image_set == 'val':
        val_set = CORN(data_root, train=False, transform=transform,rotate=False)  # 创建验证集
        return val_set
