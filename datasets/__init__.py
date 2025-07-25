import torch.utils.data
import torchvision

from .SHA import build as build_sha
from .RTC import build_rtc
from .CARPK import build as build_rsc
# from .WuhanMetro_new import build_whm
from .WuhanMetro_v2 import build_whm
from .CORN import build as build_corn
from .SOY import build_soy
from .NWPU import build_nwpu

data_path = {
    'SHA': 'data/Crowd_Counting/ShanghaiTech/part_A_final/',
    'SHB': 'data/Crowd_Counting/ShanghaiTech/part_B_final/',
    'WuhanMetro': 'data/MetroV2',
    'RTC': '/data/zlt/PET/RTC/data/RTC',
    
    'People': 'data/People',
    'Ship': 'data/Ship',
    'Car': 'data/Car',
    
    'CORN': '/data/zlt/PET/origin_pet/PET/data/dataset_corn',
    'SOY': '/data/zlt/RemoteSensePET/data/soybeam',
    'NWPU': '/data/zlt/RemoteSensePET/data/NWPU-MOC',
}

def build_dataset(image_set, args):
    args.data_path = data_path[args.dataset_file]
    if args.dataset_file in ['SHA', 'SHB']:
        return build_sha(image_set, args)
    elif args.dataset_file == 'RTC':
        return build_rtc(image_set, args)
    elif args.dataset_file in ['Ship', 'People', 'Car']:
        return build_rsc(image_set, args)
    elif args.dataset_file == 'CORN':
        return build_corn(image_set, args)
    elif args.dataset_file == 'SOY':
        return build_soy(image_set, args)
    elif args.dataset_file == 'WuhanMetro':
        return build_whm(image_set, args)
    elif args.dataset_file == 'NWPU':
        return build_nwpu(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
