from PIL import Image
import numpy as np
import random

def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01, noise_size=9):
    """
    给图像添加椒盐噪声。
    
    参数:
    - image: 输入的PIL图像对象
    - salt_prob: 白色噪点的概率
    - pepper_prob: 黑色噪点的概率
    返回:
    - 添加了噪点后的PIL图像对象
    """
    # 将图像转换为numpy数组
    img_array = np.array(image)
    
    # 获取图像的高度和宽度
    height, width, channels = img_array.shape
    
    # 生成随机位置
    num_salt = int(np.ceil(salt_prob * (height * width) / (noise_size ** 2)))
    num_pepper = int(np.ceil(pepper_prob * (height * width) / (noise_size ** 2)))
    
    # 添加椒噪声
    for _ in range(num_salt):
        x = random.randint(0, width - noise_size)
        y = random.randint(0, height - noise_size)
        if channels == 3:
            img_array[y:y+noise_size, x:x+noise_size, :] = [255, 255, 255]
        elif channels == 4:
            for i in range(noise_size):
                for j in range(noise_size):
                    img_array[y+i, x+j, :3] = [255, 255, 255]
                    # 保留Alpha通道值
                    img_array[y+i, x+j, 3] = img_array[y, x, 3]
    
    # 添加盐噪声
    for _ in range(num_pepper):
        x = random.randint(0, width - noise_size)
        y = random.randint(0, height - noise_size)
        if channels == 3:
            img_array[y:y+noise_size, x:x+noise_size, :] = [0, 0, 0]
        elif channels == 4:
            for i in range(noise_size):
                for j in range(noise_size):
                    img_array[y+i, x+j, :3] = [0, 0, 0]
                    # 保留Alpha通道值
                    img_array[y+i, x+j, 3] = img_array[y, x, 3]
    
    # 将numpy数组转换回PIL图像
    noisy_image = Image.fromarray(img_array)
    
    return noisy_image

# 读取图像
image_path = 'v.png'
image = Image.open(image_path)

# 添加椒盐噪声
noisy_image = add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01)


# 保存加噪后的图像
noisy_image.save('afv.png')