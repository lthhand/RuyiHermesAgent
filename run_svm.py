# ========== SVM 四种核函数决策边界对比（真实数据：鸢尾花前两维） ==========
# 修改版：使用 Agg 后端，保存图片到文件
import matplotlib
matplotlib.use('Agg')

import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ---------- 设置中文字体 ----------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 3. 加载鸢尾花数据集，仅使用前两个特征（便于可视化）
iris = load_iris()
X = iris.data[:, :2]  # 只取花萼长度和花萼宽度
y = iris.target  # 三个类别：Setosa, Versicolour, Virginica

print(f"数据集形状: {X.shape}")
print(f"目标类别: {iris.target_names}")

# 4. 数据标准化（SVM必须做）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 6. 定义要比较的核函数及其参数
kernels = {
    '线性核 (Linear)': {'kernel': 'linear', 'C': 1.0},
    '多项式核 (Poly, d=3)': {'kernel': 'poly', 'degree': 3, 'gamma': 'scale', 'C': 1.0},
    'RBF核 (RBF, γ=0.5)': {'kernel': 'rbf', 'gamma': 0.5, 'C': 1.0},
    'Sigmoid核 (Sigmoid)': {'kernel': 'sigmoid', 'gamma': 'scale', 'coef0': 0, 'C': 1.0}
}

# 7. 创建画布，绘制子图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

# 为每个核函数训练并绘图
for idx, (title, params) in enumerate(kernels.items()):
    model = SVC(**params, decision_function_shape='ovr')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    ax = axes[idx]
    x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
    y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set1)
    scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=plt.cm.Set1,
                         edgecolors='k', s=50)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"{title}\n准确率: {acc:.3f}")
    ax.set_xlabel("花萼长度 (标准化)")
    ax.set_ylabel("花萼宽度 (标准化)")

plt.tight_layout()
plt.savefig('C:/xxq/output/svm_kernels_comparison.png', dpi=150, bbox_inches='tight')
print("图片已保存: svm_kernels_comparison.png")

# 输出各核函数准确率
print("\n各核函数测试集准确率：")
for title, params in kernels.items():
    model = SVC(**params, decision_function_shape='ovr')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  {title}: {acc:.4f}")
