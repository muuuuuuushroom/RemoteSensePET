import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqs

# 定义K值
K = 10
numerator = [K]
denominator = [1, 6, 3, - 10]
omega = np.logspace(-2, 2, 1000)  # 从0.01到100的频率范围
w, h = freqs(numerator, denominator, omega)

plt.figure(figsize=(8, 6))
plt.plot(h.real, h.imag, label=f"K={K}")

K = 20
numerator = [K]
w, h = freqs(numerator, denominator, omega)
plt.plot(h.real, h.imag, label=f"K={K}")

K = 30
numerator = [K]
w, h = freqs(numerator, denominator, omega)
plt.plot(h.real, h.imag, label=f"K={K}")


plt.scatter([-1], [0], color='red')  # 标记 -1 点
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.axvline(0, color="black", linewidth=0.5, linestyle="--")
plt.xlabel("Re")
plt.ylabel("Im")
plt.legend()
plt.grid()
plt.savefig('1.pdf')