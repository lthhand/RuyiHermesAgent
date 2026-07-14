# =========================== 1. 导入必要的库 ===========================
import matplotlib
matplotlib.use('TkAgg')

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

# =========================== 2. 自定义 GRU 单元 ===========================
class GRUCell(nn.Module):
    """
    GRU 单步公式：
        z_t = σ(W_z·x_t + U_z·h_{t-1} + b_z)
        r_t = σ(W_r·x_t + U_r·h_{t-1} + b_r)
        h_t' = tanh(W_h·x_t + U_h·(r_t ⊙ h_{t-1}) + b_h)
        h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h_t'
    """
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.hidden_size = hidden_size
        scale = np.sqrt(2.0 / (input_size + hidden_size))

        # ---------- 更新门 z_t ----------
        self.W_z = nn.Parameter(torch.randn(hidden_size, input_size) * scale)
        self.U_z = nn.Parameter(torch.randn(hidden_size, hidden_size) * scale)
        self.b_z = nn.Parameter(torch.zeros(hidden_size, 1))

        # ---------- 重置门 r_t ----------
        self.W_r = nn.Parameter(torch.randn(hidden_size, input_size) * scale)
        self.U_r = nn.Parameter(torch.randn(hidden_size, hidden_size) * scale)
        self.b_r = nn.Parameter(torch.zeros(hidden_size, 1))

        # ---------- 候选隐藏状态 h_t' ----------
        self.W_h = nn.Parameter(torch.randn(hidden_size, input_size) * scale)
        self.U_h = nn.Parameter(torch.randn(hidden_size, hidden_size) * scale)
        self.b_h = nn.Parameter(torch.zeros(hidden_size, 1))

    def forward(self, x, h_prev):
        """
        Args:
            x     : 当前输入，形状 (input_size, 1)
            h_prev: 上一时刻隐藏状态，形状 (hidden_size, 1)
        Returns:
            h_next: 当前时刻隐藏状态，形状 (hidden_size, 1)
        """
        z = torch.sigmoid(self.W_z @ x + self.U_z @ h_prev + self.b_z)
        r = torch.sigmoid(self.W_r @ x + self.U_r @ h_prev + self.b_r)
        h_tilde = torch.tanh(self.W_h @ x + self.U_h @ (r * h_prev) + self.b_h)
        h_next = (1 - z) * h_prev + z * h_tilde
        return h_next


# =========================== 3. 定义整个 GRU 网络 ===========================
class GRUNetwork(nn.Module):
    """
    单层 GRU 网络，用于回归预测。
    包含一个 GRU 单元和一个输出全连接层。
    """
    def __init__(self, input_size, hidden_size, output_size, seq_length):
        super(GRUNetwork, self).__init__()
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.cell = GRUCell(input_size, hidden_size)

        # 输出层
        scale = np.sqrt(2.0 / hidden_size)
        self.W_out = nn.Parameter(torch.randn(output_size, hidden_size) * scale)
        self.b_out = nn.Parameter(torch.zeros(output_size, 1))

    def forward(self, X):
        """
        Args:
            X: 输入序列，形状 (seq_length, input_size, 1)
        Returns:
            y_pred: 预测值，形状 (output_size, 1)
        """
        h = torch.zeros(self.hidden_size, 1)   # 初始隐藏状态为零
        for t in range(self.seq_length):
            h = self.cell(X[t], h)
        y_pred = self.W_out @ h + self.b_out
        return y_pred


# =========================== 4. 辅助函数：构造时间序列样本 ===========================
def create_sequences(data, target, seq_length):
    """
    将一维时间序列转换为监督学习样本。
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(target[i+seq_length])
    X = np.array(X)
    y = np.array(y)
    X = X[..., np.newaxis]           # (n, seq_len, n_features, 1)
    y = y.reshape(-1, 1, 1)          # (n, 1, 1)
    return X, y


# =========================== 5. 主程序 ===========================
def main():
    # -------------------- 5.1 加载 Mauna Loa CO₂ 数据集 --------------------
    print("正在加载 Mauna Loa CO₂ 数据...")
    co2 = fetch_openml(data_id=41187, as_frame=True, parser="pandas")
    co2_df = co2.frame
    co2_df["date"] = pd.to_datetime(co2_df[["year", "month", "day"]])
    co2_df = co2_df[["date", "co2"]].set_index("date")
    values = co2_df["co2"].values.astype(np.float32).reshape(-1, 1)
    print(f"数据总量: {len(values)} 个样本")
    print(f"时间范围: {co2_df.index.min()} 到 {co2_df.index.max()}")

    # -------------------- 5.2 数据标准化 --------------------
    scaler = StandardScaler()
    values_scaled = scaler.fit_transform(values)

    # -------------------- 5.3 划分训练/测试集（时间顺序）--------------------
    train_size = int(len(values_scaled) * 0.8)
    train_data = values_scaled[:train_size]
    test_data = values_scaled[train_size:]

    # -------------------- 5.4 构造序列样本 --------------------
    seq_length = 6
    X_train, y_train = create_sequences(train_data, train_data, seq_length)
    X_test, y_test = create_sequences(test_data, test_data, seq_length)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    print(f"训练样本数: {X_train.shape[0]}, 测试样本数: {X_test.shape[0]}")

    # -------------------- 5.5 初始化模型、优化器、损失函数 --------------------
    input_size = 1
    hidden_size = 96
    output_size = 1
    model = GRUNetwork(input_size, hidden_size, output_size, seq_length)

    optimizer = optim.SGD(model.parameters(), lr=0.00001)   # 可尝试调大学习率，如 0.001
    criterion = nn.MSELoss()

    # -------------------- 5.6 训练循环 --------------------
    epochs = 300
    train_losses = []
    test_losses = []

    print("\n开始训练...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i in range(X_train.shape[0]):
            x_seq = X_train[i]
            y_true = y_train[i]
            y_pred = model(x_seq)
            loss = criterion(y_pred, y_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / X_train.shape[0]
        train_losses.append(avg_train_loss)

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for i in range(X_test.shape[0]):
                y_pred = model(X_test[i])
                test_loss += criterion(y_pred, y_test[i]).item()
        avg_test_loss = test_loss / X_test.shape[0]
        test_losses.append(avg_test_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}  Train Loss: {avg_train_loss:.6f}  Test Loss: {avg_test_loss:.6f}")

    # -------------------- 5.7 模型评估 --------------------
    print("\n评估模型在测试集上的表现...")
    model.eval()
    y_pred_list = []
    y_true_list = []
    with torch.no_grad():
        for i in range(X_test.shape[0]):
            y_pred = model(X_test[i])
            y_pred_list.append(y_pred.item())
            y_true_list.append(y_test[i].item())

    y_pred = np.array(y_pred_list).reshape(-1, 1)
    y_true = np.array(y_true_list).reshape(-1, 1)

    y_pred_orig = scaler.inverse_transform(y_pred)
    y_true_orig = scaler.inverse_transform(y_true)
    rmse = np.sqrt(np.mean((y_true_orig - y_pred_orig) ** 2))
    print(f"测试集 RMSE = {rmse:.4f} ppm")

    # -------------------- 5.8 可视化 --------------------
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(test_losses, label='Test Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Loss Curves during Training')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(y_true_orig, y_pred_orig, alpha=0.6, edgecolors='k', linewidth=0.5)
    min_val = min(y_true_orig.min(), y_pred_orig.min())
    max_val = max(y_true_orig.max(), y_pred_orig.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('True CO₂ (ppm)')
    plt.ylabel('Predicted CO₂ (ppm)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 4))
    n_show = min(100, len(y_true_orig))
    plt.plot(y_true_orig[:n_show], label='True', linewidth=2, color='green')
    plt.plot(y_pred_orig[:n_show], label='Predicted', linewidth=2, linestyle='--', color='red')
    plt.xlabel('Time Step (month)')
    plt.ylabel('CO₂ Concentration (ppm)')
    plt.title('CO₂ Prediction on Test Set (First 100 Months)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()