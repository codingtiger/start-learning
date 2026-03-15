import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# ── 基准目录（确保所有输出文件保存在脚本所在目录下） ────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 生成训练数据
np.random.seed(42)
n_samples = 100
X_train = np.linspace(0, 1, n_samples).reshape(-1, 1)
y_train = np.sin(2 * np.pi * X_train.flatten()) + np.random.normal(0, 0.1, n_samples)

# 生成测试数据（更密集的点用于绘图）
X_test = np.linspace(0, 1, 1000).reshape(-1, 1)
y_true = np.sin(2 * np.pi * X_test.flatten())

# 尝试不同次数的多项式特征
degrees = [1, 3, 5, 9]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, degree in enumerate(degrees):
    # 创建多项式特征管道
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])

    # 训练模型
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    # 计算评估指标，使用均方误差
    mse = mean_squared_error(y_train, y_train_pred)
    r2 = r2_score(y_train, y_train_pred)

    # 绘图
    ax = axes[idx]
    ax.scatter(X_train, y_train, s=20, alpha=0.5, label='训练数据', color='blue')
    ax.plot(X_test, y_true, 'g-', linewidth=2, label='真实函数 sin(2πx)')
    ax.plot(X_test, y_pred, 'r-', linewidth=2, label=f'拟合结果 (degree={degree})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'多项式次数 = {degree}\nMSE = {mse:.4f}, R² = {r2:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'sin_fitting_results.png'), dpi=300, bbox_inches='tight')
print("图像已保存为 'sin_fitting_results.png'")
plt.show()

# 打印最佳模型信息
print("\n=== 模型对比 ===")
for degree in degrees:
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    mse = mean_squared_error(y_train, y_train_pred)
    r2 = r2_score(y_train, y_train_pred)
    print(f"次数 {degree}: MSE = {mse:.6f}, R² = {r2:.6f}")
