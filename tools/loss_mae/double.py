import re
import json
import matplotlib.pyplot as plt
import pandas as pd
import os


# 定义一个函数来解析 log 文件并提取数据
def parse_log(log_file, end_epoch = 400):
    epochs = []
    train_losses = []
    best_maes = []
    best_epochs = []
    current_best_mae = float('inf')
    
    with open(log_file, 'r') as f:
        for line in f:
            # 查找训练记录行
            if '[ep' in line:
                json_str = re.search(r'{.*}', line)
                if json_str:
                    data = json.loads(json_str.group())
                    epoch = data['epoch']
                    if epoch < end_epoch:
                        epochs.append(data['epoch'])
                        train_losses.append(data['train_loss'])
            
            # 查找 best mae 信息
            elif 'best mae' in line:
                best_info = re.search(r'best mae:(\d+\.\d+), best epoch: (\d+)', line)
                if best_info:
                    mae = float(best_info.group(1))
                    epoch = int(best_info.group(2))
                    # 更新 best mae 记录
                    if epoch < end_epoch and mae < current_best_mae:
                        current_best_mae = mae
                        
                        best_maes.append(mae)
                        best_epochs.append(epoch)

    # 平滑 train_loss 曲线
    train_losses_smoothed = pd.Series(train_losses).ewm(span=10, adjust=False).mean().values
    return epochs, train_losses_smoothed, best_epochs, best_maes

# 读取两个 log 文件的数据
num = 1500
# for process in [2, 8]:
process = str(num) + '_gaussion'
label1 = 'PET'
log_dir1 = '/data/zlt/PET/RTC/outputs_detr/Ship/swin_t_noencoder_detr'

label2 = 'Box query'
log_dir2 = '/data/zlt/PET/RTC/outputs_detr/Ship/loss_based/swin_t_noencoder_box_detr_agent_sigma_gaussion_query'

label3 = 'only box 6dec'
log_dir3 = '/data/zlt/PET/RTC/outputs_detr/Ship/swin_t_noencoder_box_detr_6dec_agent_NOoffset'

label4 = 'x'
log_dir4 = '/data/zlt/PET/RTC/outputs_detr/Ship/swin_t_noencoder_box_detr_2dec_agent_NOoffset'

log_dir1 = os.path.join(log_dir1, 'run_log.txt')
log_dir2 = os.path.join(log_dir2, 'run_log.txt')
log_dir3 = os.path.join(log_dir3, 'run_log.txt')
log_dir4 = os.path.join(log_dir4, 'run_log.txt')


epochs1, train_losses1, best_epochs1, best_maes1 = parse_log(log_dir1, end_epoch=num)
epochs2, train_losses2, best_epochs2, best_maes2 = parse_log(log_dir2, end_epoch=num)

third = False
if third:
    epochs3, train_losses3, best_epochs3, best_maes3 = parse_log(log_dir3, end_epoch=num)
    epochs4, train_losses4, best_epochs4, best_maes4 = parse_log(log_dir4, end_epoch=num)

# 绘制 train_loss 对比图
plt.figure(figsize=(12, 6))
plt.plot(epochs1, train_losses1, label=label1, color='green')
plt.plot(epochs2, train_losses2, label=label2, color='purple')
if third:
    plt.plot(epochs3, train_losses3, label=label3, color='red')
    plt.plot(epochs4, train_losses4, label=label4, color='orange')
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.title(f'{process} Train Loss Comparison')
plt.legend()
plt.show()
plt.savefig(f'/data/zlt/PET/RTC/tools/loss_mae/tmp/loss/loss_cmp_{process}.pdf')

# 绘制 best mae 对比图
plt.figure(figsize=(12, 6))
plt.plot(best_epochs1, best_maes1, label=label1, color='green', marker='o')
plt.plot(best_epochs2, best_maes2, label=label2, color='purple', marker='x')
if third:
    plt.plot(best_epochs3, best_maes3, label=label3, color='red')
    plt.plot(best_epochs4, best_maes4, label=label4, color='orange')
plt.xlabel('Epoch')
plt.ylabel('Best MAE')
plt.title(f'{process} Best MAE Comparison')
plt.legend()
plt.show()
plt.savefig(f'/data/zlt/PET/RTC/tools/loss_mae/tmp/mae/mae_cmp_{process}.pdf')



# box = 1
# point_offset = 1
# if 1:
#                 # box-detr: box agent
#                 agent = box[..., :2].unsqueeze(2) + (box[..., 2:].unsqueeze(2) / 2) * point_offset
                
#                 # set1:
#                 agent = box[..., :2].unsqueeze(2) + (box[..., :2].unsqueeze(2) / 2) * point_offset
                
#                 # set2:
#                 agent = box[..., :2].unsqueeze(2) + point_offset
                
#                 # set3:
#                 agent = box[..., :2].unsqueeze(2)