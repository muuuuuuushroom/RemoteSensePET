import torch.utils.data
import torchvision

from .SHA import build as build_sha
from .RTC import build_rtc
from .CARPK import build as build_rsc

from .CORN import build as build_corn
from .SOY import build_soy

data_path = {
    'SHA': './data/ShanghaiTech/part_A/',
    'RTC': '/data/zlt/PET/RTC/data/RTC',
    'People': '/data/zlt/PET/RTC/data/People',
    
    'Ship': '/data/zlt/datasets/Ship',
    'Car': '/data/zlt/datasets/Car',
    
    'CORN': '/data/zlt/PET/origin_pet/PET/data/dataset_corn',
    
    'SOY': '/data/zlt/RemoteSensePET/data/soybeam',
}

def build_dataset(image_set, args):
    args.data_path = data_path[args.dataset_file]
    if args.dataset_file == 'SHA':
        return build_sha(image_set, args)
    elif args.dataset_file == 'RTC':
        return build_rtc(image_set, args)
    elif args.dataset_file in ['Ship', 'People', 'Car']:
        return build_rsc(image_set, args)
    elif args.dataset_file == 'CORN':
        return build_corn(image_set, args)
    elif args.dataset_file == 'SOY':
        return build_soy(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
