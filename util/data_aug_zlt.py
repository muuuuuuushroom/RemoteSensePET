import json
import torch
import os
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def create_hard_case_sampler_train(data_path ,dataset, sampler_train, batch_size, ratio):
    # RSC-SHIP hard cases
    try:
        # hard cases is a sorted index for hard cases 
        hard_cases = read_json_file(os.path.join(data_path, 'unsatisfy_indexs.json'))
    except:
        print('dataset has no unsatisfy json file! now using normal data sampler')
        return torch.utils.data.BatchSampler(sampler_train, batch_size, drop_last=True)
    
    # file_names = [dataset[i][0] for i in range(len(dataset))]  # 假设dataset_train[i][0]返回文件名
    
    weights = [1.0] * len(dataset)
    for idx in hard_cases:
        weights[idx] *= ratio  # increase sampling ratio weight to the scale at 3 times
        
    sampler_train = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, batch_size, drop_last=True)
    
    print('hard cases ratio has been scaled')
    return batch_sampler_train
   