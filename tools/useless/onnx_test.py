import os, sys
import numpy as np
import time

sys.path.append(os.getcwd())
import onnxruntime
import torchvision.transforms as transforms
from PIL import Image


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

mean = [0.4051, 0.4392, 0.2344]
std = [0.2569, 0.2620, 0.2287]

def get_test_transform(): 
    return transforms.Compose([
        transforms.Resize([2816, 2816]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

image = Image.open('/data/slcao_data/data/rice_tiller_count/images/2024-05-16_02-13-44_Experiment2024wheat_B1_P720_U1_DIRXpY0_TH15_GW0103-S1-23SF17.jpg')
# img = get_test_transform()(image)
# img = img.unsqueeze_(0)

img = np.array(image)
img = img.astype(np.float32)
for i in range(3):
    img[i,:,:] = (img[i,:,:] - mean[i]) / std[i]
img = np.expand_dims(img, 0)
print(img.shape)

# img = np.load('/data/slcao_data/data/rice_tiller_count/img_norm_numpy/2024-05-16_02-13-44_Experiment2024wheat_B1_P720_U1_DIRXpY0_TH15_GW0103-S1-23SF17.npy')
# img = np.expand_dims(img, 0)
# print(img.shape)

# 模型加载
print('loading model')
onnx_model_path = "./test_2816.onnx"
model = onnxruntime.InferenceSession(onnx_model_path)
inputs = {model.get_inputs()[0].name: img}
test_times = 1
print('start inference')
st = time.time()
for i in range(test_times):
    # print(i)
    outs = model.run(None, inputs)
print(f"use time: {(time.time() - st) / test_times}")

print("outs[0]", outs[0])
print("outs", outs)

