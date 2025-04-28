import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

# 定义系统1的分子和分母系数
num1 = [1]
den1 = [1, 10, 100]

# 定义系统2的分子和分母系数
num2 = [1, 5]
den2 = [1, 10, 100]

# 定义系统3的分子和分母系数
num3 = [1]
den3 = [1, 15, 150, 500]

# 创建系统对象
sys1 = ctrl.TransferFunction(num1, den1)
sys2 = ctrl.TransferFunction(num2, den2)
sys3 = ctrl.TransferFunction(num3, den3)

# 计算阶跃响应
t1, yout1 = ctrl.step_response(sys1)
t2, yout2 = ctrl.step_response(sys2)
t3, yout3 = ctrl.step_response(sys3)

# 绘制阶跃响应曲线
plt.figure(figsize=(10, 6))
plt.plot(t1, yout1, label='o')
plt.plot(t2, yout2, label='z')
plt.plot(t3, yout3, label='p')

# 添加图例、标题和轴标签
plt.legend()
plt.title('Step Response of Three Systems')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# 显示网格
plt.grid(True)

# 显示图形
plt.savefig('show.pdf')



