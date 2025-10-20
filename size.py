import os
from PIL import Image

def get_image_resolutions(folder_path):
    """
    读取指定文件夹下所有图片的分辨率，并返回一个按面积从大到小排序的唯一分辨率列表。

    参数:
    folder_path (str): 要扫描的文件夹路径。

    返回:
    list: 一个包含元组 (width, height) 的列表，按分辨率大小降序排列。
          如果文件夹无效或没有找到图片，则返回空列表。
    """
    # 检查路径是否存在且是否为文件夹
    if not os.path.isdir(folder_path):
        print(f"错误：提供的路径 '{folder_path}' 不是一个有效的文件夹。")
        return []

    # 定义支持的图片文件扩展名（可根据需要添加）
    supported_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    
    # 使用集合来自动处理重复的分辨率
    unique_resolutions = set()

    print(f"正在扫描文件夹: {folder_path}\n")

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 获取文件的扩展名（并转换为小写以便比较）
        file_ext = os.path.splitext(filename)[1].lower()

        if file_ext in supported_extensions:
            file_path = os.path.join(folder_path, filename)
            try:
                # 使用 with 语句确保图片文件被正确关闭
                with Image.open(file_path) as img:
                    # 获取图片的宽和高
                    width, height = img.size
                    # 将分辨率元组添加到集合中
                    unique_resolutions.add((width, height))
            except Exception as e:
                print(f"无法读取文件 '{filename}' 的分辨率: {e}")

    # 如果没有找到任何图片
    if not unique_resolutions:
        return []

    # 将集合转换为列表，并按照面积（宽 * 高）从大到小排序
    # lambda res: res[0] * res[1] 是一个匿名函数，用于计算每个分辨率的面积
    sorted_resolutions = sorted(list(unique_resolutions), key=lambda res: res[0] * res[1], reverse=False)
    
    return sorted_resolutions

if __name__ == "__main__":
    # 提示用户输入文件夹路径
    # 您也可以直接在这里修改为固定的路径，例如： target_folder = "C:/Users/YourUser/Pictures"
    target_folder = '/data/zlt/RemoteSensePET/data/Vessel/test_data/images'

    # 获取排序后的分辨率列表
    resolutions = get_image_resolutions(target_folder)

    # 打印结果
    if resolutions:
        print("--- 图片分辨率统计结果 (从大到小) ---")
        for width, height in resolutions:
            print(f"{width} x {height}")
    else:
        print("在该文件夹中没有找到任何支持的图片文件。")