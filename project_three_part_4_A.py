import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, WhiteKernel, ConstantKernel

# ---------- 设置中文字体 ----------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

# ========== 1. 读取 TXT 数据 ==========
# 使用 np.loadtxt 读取，跳过第一行（表头）
data = np.loadtxt('co2_data.txt', skiprows=1)
X = data[:, 0].reshape(-1, 1)   # 年份
y = data[:, 1]                  # CO₂ 浓度
print(f"成功读取 {len(X)} 个数据点，年份范围 {X[0,0]:.1f} ~ {X[-1,0]:.1f}")

# ========== 2. 划分训练集和测试集 ==========
split = int(0.85 * len(X))   # 85% 训练，15% 测试（约 23 个点）
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
print(f"训练集: {len(X_train)} 点，测试集: {len(X_test)} 点")

# ========== 3. 均值中心化（提升数值稳定性） ==========
y_mean = y_train.mean()
y_train_centered = y_train - y_mean

# ========== 4. 核函数 ==========
kernel = (
    ConstantKernel(30.0**2, (5.0**2, 80.0**2)) * RBF(length_scale=5.0, length_scale_bounds=(1.0, 20.0))
    + ConstantKernel(2.0**2, (0.5**2, 5.0**2)) * ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds='fixed')
    + WhiteKernel(noise_level=0.2**2, noise_level_bounds=(1e-4, 2.0))
)

gpr = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=2,
    alpha=0,
    random_state=42
)

# ========== 5. 训练模型 ==========
print("\n训练高斯过程回归模型...")
gpr.fit(X_train, y_train_centered)

print("优化后的核参数：")
print(gpr.kernel_)
print(f"对数边际似然: {gpr.log_marginal_likelihood():.2f}")

# ========== 6. 测试集评估 ==========
y_pred, y_std = gpr.predict(X_test, return_std=True)
y_pred += y_mean

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\n测试集 MAE: {mae:.3f} ppm, RMSE: {rmse:.3f} ppm")

# ========== 7. 预测至 1978 年 ==========
X_future = np.linspace(X.min(), 1978, 400).reshape(-1, 1)
y_future, y_future_std = gpr.predict(X_future, return_std=True)
y_future += y_mean

# ========== 8. 可视化 ==========
plt.figure(figsize=(10, 5))
plt.scatter(X_train, y_train, c='darkgreen', s=12, label='训练数据', alpha=0.6)
plt.scatter(X_test, y_test, c='orange', s=25, label='测试数据', alpha=0.8)
plt.plot(X_future, y_future, 'b-', linewidth=2, label='预测均值')
plt.fill_between(X_future.ravel(),
                 y_future - 1.96*y_future_std,
                 y_future + 1.96*y_future_std,
                 alpha=0.2, color='blue', label='95% 置信区间')
plt.axvline(x=X_train[-1,0], color='red', linestyle=':', label='训练/测试分界')
plt.xlabel('年份')
plt.ylabel('CO₂ 浓度 (ppm)')
plt.legend()
plt.grid(alpha=0.3)
plt.xlim(1955, 1980)
plt.tight_layout()
plt.show()

# ========== 9. 输出未来预测值 ==========
print("\n未来年份预测 (1975-1978 每年)：")
for year in [1975, 1976, 1977, 1978]:
    idx = np.argmin(np.abs(X_future.ravel() - year))
    pred = y_future[idx]
    std = y_future_std[idx]
    print(f"{year}年: {pred:.2f} ± {1.96*std:.2f} ppm (95% CI)")

print("程序运行结束。")
