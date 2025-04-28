import re
import os
import json
import matplotlib.pyplot as plt

def remove_duplicates(best_mae_history):
    # 使用字典来过滤重复的 (epoch, mae) 组合
    unique_history = []
    seen = set()
    for entry in best_mae_history:
        # 创建 (epoch, mae) 的唯一标识
        identifier = (entry["epoch"], entry["mae"])
        if identifier not in seen:
            seen.add(identifier)
            unique_history.append(entry)
    return unique_history

def get_string_between_last_two_slashes(input_string):
    # 找到最后一个'/'的索引
    last_slash_index = input_string.rfind('/')
    
    # 找到倒数第二个'/'的索引
    second_last_slash_index = input_string.rfind('/', 0, last_slash_index)
    
    # 提取倒数第二个和最后一个'/'之间的内容
    if second_last_slash_index != -1 and last_slash_index != -1:
        return input_string[second_last_slash_index + 1:last_slash_index]
    else:
        return None

def extract_metrics(log_file_path, json_output_path):
    
    if not os.path.exists(json_output_path):
        with open(json_output_path, 'w') as json_file:
            json.dump({"metrics": [], "best_mae": None, "best_epoch": None}, json_file, indent=4)

    
    pattern = re.compile(r"epoch:(\d+), mae:(\d+\.\d+), mse:(\d+\.\d+), r2:(-?\d+\.\d+), rmae:(\d+\.\d+), "
                         r"rmse:(\d+\.\d+), racc(\d+\.\d+), time(\d+\.\d+)")
    best_pattern = re.compile(r"best mae:(\d+\.\d+), best epoch: (\d+)")
    
    metrics = []
    best_mae = None
    best_epoch = None
    best_mae_history = []

    # 读取日志文件
    with open(log_file_path, 'r') as log_file:
        for line in log_file:
            match = pattern.search(line)
            if match:
                epoch, mae, mse, r2, rmae, rmse, racc, time = match.groups()
                mae = float(mae)
                metrics.append({
                    "epoch": int(epoch),
                    "mae": mae,
                    "mse": float(mse),
                    "r2": float(r2),
                    "rmae": float(rmae),
                    "rmse": float(rmse),
                    "racc": float(racc),
                    "time": float(time)
                })

                # 如果当前 mae 小于已有的 best mae，则更新
                if best_mae is None or mae < best_mae:
                    best_mae = mae
                    best_epoch = int(epoch)
                    best_mae_history.append({
                        "epoch": best_epoch,
                        "mae": best_mae,
                        "mse": float(mse),
                        "r2": float(r2),
                        "rmae": float(rmae),
                        "rmse": float(rmse),
                        "racc": float(racc),
                        "time": float(time)
                    })
    
    # 检查JSON文件是否存在，不存在则新建
    if not os.path.exists(json_output_path):
        with open(json_output_path, 'w') as json_file:
            json.dump({"metrics": [], "best_mae": None, "best_epoch": None, "best_mae_history": []}, json_file, indent=4)

    # 读取现有的json文件内容
    with open(json_output_path, 'r') as json_file:
        data = json.load(json_file)

    # 确保 'best_mae_history' 键存在
    if "best_mae_history" not in data:
        data["best_mae_history"] = []

    # 将新提取的metrics和best值添加到json中
    data["metrics"].extend(metrics)
    if best_mae is not None:
        data["best_mae"] = best_mae
        data["best_epoch"] = best_epoch
    data["best_mae_history"].extend(best_mae_history)

    # 将更新后的数据写回json文件
    with open(json_output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    # print(f"Metrics successfully extracted and saved to {json_output_path}")
    
def save_best_mae_history_plot(json_output_path, save_path, ckpt):
    # 读取json文件
    with open(json_output_path, 'r') as json_file:
        data = json.load(json_file)

    # 提取best_mae_history中的epoch和mae
    best_mae_history = data.get("best_mae_history", [])
    if not best_mae_history:
        print("No best_mae_history data found.")
        return
    
    best_mae_history = remove_duplicates(best_mae_history)

    epochs = [entry["epoch"] for entry in best_mae_history]
    maes = [entry["mae"] for entry in best_mae_history]

    # 绘制数据点
    plt.figure(figsize=(10, 6))
    plt.scatter(epochs, maes, color='b', label='MAE Data Points')

    # 连续画出各点之间的线，不连接起首尾
    for i in range(len(epochs) - 1):
        plt.plot(epochs[i:i+2], maes[i:i+2], 'b-')

    plt.title(ckpt)
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.grid(True)
    plt.tight_layout()

    # 保存图表到指定路径
    plt.savefig(save_path)
    plt.close()
    # print(f"Plot saved at {save_path}")
    
def get_first_level_subfolders(directory):
    subfolders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    return subfolders

def load_best_mae_history(json_file_path):
    
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
        
    best_mae_history = data.get("best_mae_history", [])
    best_mae_history = remove_duplicates(best_mae_history)
            
    if not best_mae_history:
        print("No best_mae_history data found.")
        return

    epochs = [entry["epoch"] for entry in best_mae_history]
    maes = [entry["mae"] for entry in best_mae_history]
    
    return epochs, maes

def compare_two_json_files(json_file_path_1, json_file_path_2, save_path):
    # 读取第一个JSON文件
    epochs_1, maes_1 = load_best_mae_history(json_file_path_1)
    
    # 读取第二个JSON文件
    epochs_2, maes_2 = load_best_mae_history(json_file_path_2)
    
    file1 = get_string_between_last_two_slashes(json_file_path_1)
    file2 = get_string_between_last_two_slashes(json_file_path_2)

    s = 3
    # 绘制折线图
    plt.figure(figsize=(10, 6))
    # 绘制第一条曲线
    plt.plot(epochs_1, maes_1, markersize=s, marker='o', linestyle='-', color='b', label=file1)
    
    # 绘制第二条曲线
    plt.plot(epochs_2, maes_2, markersize=s, marker='o', linestyle='-', color='r', label=file2)
    
    # 设置标题和标签
    plt.title('Comparison of MAE History Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.grid(True)
    
    # 添加图例以区分两条曲线
    plt.legend()

    # 调整布局并保存图表
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Comparison plot saved at {save_path}")

if __name__ == '__main__':
    root_path = '/data/zlt/PET/RTC/outputs_detr'
    dataset = '/People/'

    paint_level = 'com'
    
    if paint_level == 'com':
        # paint in comparison:
        # ['swin_t_noencoder_detr', 'swin_t_box_detr', 'swin_t_detr', 'swin_s_noencoder_detr', 'swin_b_detr', 'swin_b_box_detr', 'vgg_noencoder_detr', 'swin_s_noencoder_box_detr', 'swin_t_noencoder_box_detr']
        front = root_path + dataset
        ckpt1 = 'swin_t_detr/best_info.json'  
        json1 = os.path.join(front, ckpt1)
        ckpt2 = 'swin_t_box_detr/best_info.json'  
        json2 = os.path.join(front, ckpt2)
        plot_save_path = os.path.join(front, 'swin_t.pdf')

        compare_two_json_files(json1, json2, plot_save_path)
    
    if paint_level == 'or':
        # paint originally:
        subfolders = get_first_level_subfolders(root_path + dataset)
        for ckpt in subfolders:
        # ckpt = 'swin_t_noencoder_detr'
            if 'vgg' in ckpt:
                continue
            front = root_path + dataset + ckpt
            back_log = 'run_log.txt'
            back_json = 'best_info.json'
            back_fig = 'best_curve.pdf'

            log_file_path = os.path.join(front, back_log)  
            json_output_path = os.path.join(front, back_json)
            save_path = os.path.join(front, back_fig)

            extract_metrics(log_file_path, json_output_path)
            save_best_mae_history_plot(json_output_path, save_path, ckpt)
            print('success:\t', ckpt)