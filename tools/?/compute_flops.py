# this is the main entrypoint
# as we describe in the paper, we compute the flops over the first 100 images
# on COCO val2017, and report the average result
import torch
import time
import torchvision

import os
import cv2
import numpy as np
import argparse
import tqdm

from util.custom_log import load_config
from models.pet import build_pet
from util.flop_count import flop_count
from torchsummary import summary
from torchstat import stat
import thop


def warmup(model, inputs, N=10):
    for i in range(N):
        out = model(inputs)
    torch.cuda.synchronize()


def measure_time(model, inputs, N=10):
    warmup(model, inputs)
    s = time.time()
    for i in range(N):
        out = model(inputs)
    torch.cuda.synchronize()
    t = (time.time() - s) / N
    return t


def fmt_res(data):
    return data.mean(), data.std(), data.min(), data.max()

img_list = os.listdir('/data/slcao_data/data/rice_tiller_count')
images = []
for idx in range(100):
    # img, t = dataset[idx]
    img = torch.rand((3,2816,2816))
    # img = cv2.imread('')
    images.append(img)

device = torch.device('cuda')
results = {}
for model_name in ['configs/vgg_pet.yaml', 'configs/swin_pet.yaml', 'outputs/RTC/agent_swin_t_nc512_e300_crop512_b8r4_times3/config.yaml']:
    config = load_config(model_name)
    config['device'] = 'cuda'
    args = argparse.Namespace(**config)
    # model = torch.hub.load('facebookresearch/detr', model_name)
    model, criterion = build_pet(args)
    model.to(device)
    # model.eval()
    
    with torch.no_grad():
        tmp = []
        tmp2 = []
        for img in tqdm.tqdm(images):
            inputs = [img.to(device)]
            # summary(model, (3,512,512))
            # thop.profile(model, (torch.rand(1,3,512,512).cuda(), ))
            res = flop_count(model, (inputs,))
            t = measure_time(model, inputs)
            tmp.append(sum(res.values()))
            tmp2.append(t)
            # break

    results[model_name] = {'flops': fmt_res(np.array(tmp)), 'time': fmt_res(np.array(tmp2))}


print('=============================')
print('')
for r in results:
    print(r)
    for k, v in results[r].items():
        print(' ', k, ':', v)