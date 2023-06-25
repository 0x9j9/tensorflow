# tensorflow
用于构建和训练机器学习模型的深度学习框架
import numpy as np
import matplotlib.pyplot as plt

def linear_regression(x, y):
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # 计算回归系数
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    
    # 计算预测值
    y_pred = slope * x + intercept
    
    # 绘制数据点和回归线
    plt.scatter(x, y, color='blue', label='Data')
    plt.plot(x, y_pred, color='red', label='Linear Regression')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

# 示例数据
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 4, 5, 6])

# 绘制线性回归
linear_regression(x, y)
