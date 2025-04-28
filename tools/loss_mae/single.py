import re
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 读取 log 文件
log_file = '/data/zlt/PET/RTC/outputs_detr/Ship/swin_t_noencoder_detr/run_log.txt'

# 用于存储数据
epochs = []
train_losses = []
best_maes = []
best_epochs = []

# 逐行读取文件并解析
with open(log_file, 'r') as f:
    current_best_mae = float('inf')
    for line in f:
        # 查找训练记录行
        if '[ep' in line:
            # 使用正则表达式提取 JSON 字符串部分
            json_str = re.search(r'{.*}', line)
            if json_str:
                data = json.loads(json_str.group())
                epochs.append(data['epoch'])
                train_losses.append(data['train_loss'])
        
        # 查找 best mae 信息
        elif 'best mae' in line:
            # 使用正则提取 best mae 和 best epoch
            best_info = re.search(r'best mae:(\d+\.\d+), best epoch: (\d+)', line)
            if best_info:
                mae = float(best_info.group(1))
                epoch = int(best_info.group(2))
                # 更新 best mae 记录
                if mae < current_best_mae:
                    current_best_mae = mae
                    best_maes.append(mae)
                    best_epochs.append(epoch)

# 平滑 train_loss 曲线 (使用指数移动平均)
window = 10  # 平滑窗口大小
train_losses_smoothed = pd.Series(train_losses).ewm(span=window, adjust=False).mean().values

# 创建双 y 轴图像
fig, ax1 = plt.subplots(figsize=(12, 6))

# 绘制平滑后的 train_loss 曲线
ax1.plot(epochs, train_losses_smoothed, label='Smoothed Train Loss', color='blue')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Train Loss', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# 创建第二个 y 轴并绘制 best mae 曲线
ax2 = ax1.twinx()
ax2.plot(best_epochs, best_maes, label='Best MAE', color='red', marker='o')
ax2.set_ylabel('Best MAE', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# 添加图例
fig.tight_layout()
plt.title('Train Loss and Best MAE over Epochs')
plt.savefig('/data/zlt/PET/RTC/tools/loss_mae/tmp/tmp1.pdf')
