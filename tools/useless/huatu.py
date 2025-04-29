mode = 'unexpected'

if mode == 'loss':

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # 创建一个示例时间序列
    dates = pd.date_range(start='1970-01-01', periods=20, freq='D')
    values = np.sin(np.linspace(0, 10, 20))  # 生成一些正弦波形的数据
    time_series = pd.Series(values, index=dates)



    # 人为地插入一些缺失值
    missing_indices = [5, 10, 15]  # 假设第6, 11, 16天的数据缺失
    for index in missing_indices:
        time_series.iloc[index] = np.nan

    # 使用线性插值方法填补缺失值
    interpolated_series = time_series.interpolate(method='linear')
    nan_mask = np.isnan(time_series)

    # 绘制原始时间序列和插值后的结果
    plt.figure(figsize=(10, 6))
    plt.plot(time_series, 'o-', linewidth=4, markersize=16)

    # plt.plot(interpolated_series, 'o-',  linewidth=4, markersize=16)

    plt.plot(np.where(nan_mask)[0], interpolated_series[nan_mask], 'o',
            linewidth=4, markersize=16)  
    plt.title('Linear Interpolation of a Time Series with Missing Values')
    # plt.legend()
    plt.savefig('1.pdf')

else:
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # 创建一个示例时间序列
    dates = pd.date_range(start='1970-01-01', periods=10, freq='D')
    values = np.sin(np.linspace(0, 5, 10))  # 生成一些正弦波形的数据
    time_series = pd.Series(values, index=dates)

    # 添加一些异常值
    time_series.iloc[3] = 2.5
    time_series.iloc[7] = -2.5

    # 定义滑动窗口大小和阈值系数
    window_size = 5
    k = 1

    # 计算滑动窗口内的均值和标准差
    time_series_rolling = time_series.rolling(window=window_size, center=True).agg(['mean', 'std'])

    # 计算异常值阈值
    lower_bound = time_series_rolling['mean'] - k * time_series_rolling['std']
    upper_bound = time_series_rolling['mean'] + k * time_series_rolling['std']

    # 检测异常值
    outliers = (time_series < lower_bound) | (time_series > upper_bound)

    # 校正异常值
    time_series_corrected = time_series.copy()
    time_series_corrected[outliers] = np.nan  # 将异常值标记为 NaN
    time_series_corrected = time_series_corrected.interpolate(method='linear')  # 使用线性插值进行校正

    nan_mask = np.isnan(time_series)
    # 绘制原始时间序列、异常值和校正后的序列
    plt.figure(figsize=(10, 6))

    # 绘制原始时间序列
    plt.plot(time_series, 'o-',  linewidth=4, markersize=16)
    plt.savefig('oo.pdf')

    # 绘制异常值
    plt.plot(time_series[outliers], 'x', color='red', linewidth=4, markersize=24)

    # 绘制校正后的序列
    # plt.plot(time_series_corrected, 'o-', linewidth=4, markersize=16)
    plt.plot(np.where(nan_mask)[0], time_series_corrected[nan_mask], 'o',
            linewidth=4, markersize=16) 

    # 绘制均值和标准差范围
    plt.fill_between(time_series_rolling.index, lower_bound, upper_bound, color='gray', alpha=0.3, )

    plt.title('Anomaly Detection and Correction Using Sliding Window Method')
    # plt.legend()
    plt.show()

    plt.savefig('1.pdf')