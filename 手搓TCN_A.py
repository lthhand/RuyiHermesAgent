# =========================== 1. 导入必要的库 ===========================
import matplotlib
matplotlib.use('TkAgg')

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

# =========================== 2. TCN 残差块 ===========================
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(TCNBlock, self).__init__()
        self.padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=0)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding=0)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = F.pad(x, (self.padding, 0))
        out = self.conv1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = F.pad(out, (self.padding, 0))
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)
        out = out + res
        return self.relu(out)

# =========================== 3. TCN 网络 ===========================
class TCNNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_length,
                 num_layers=3, kernel_size=3, dropout=0.0):
        super(TCNNetwork, self).__init__()
        self.seq_length = seq_length

        layers = []
        in_ch = input_size
        for i in range(num_layers):
            dilation = 2 ** i
            out_ch = hidden_size
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        # X: (seq_length, input_size, 1) -> (1, input_size, seq_length)
        x = X.squeeze(-1).transpose(0, 1).unsqueeze(0)
        out = self.network(x)          # (1, hidden_size, seq_length)
        last = out[:, :, -1]           # (1, hidden_size)
        pred = self.linear(last)       # (1, output_size)
        return pred.unsqueeze(-1)      # (output_size, 1)

# =========================== 4. 辅助：构造序列样本 ===========================
def create_sequences(data, target, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(target[i+seq_length])
    X = np.array(X)                 # (n, seq_len, n_features)
    y = np.array(y)                 # (n,)
    X = X[..., np.newaxis]          # (n, seq_len, n_features, 1)
    y = y.reshape(-1, 1, 1)         # (n, 1, 1)
    return X, y

# =========================== 5. 主程序 ===========================
def main():
    print("=" * 60)
    print("TCN 时序预测示例 (仿真数据)")
    print("=" * 60)

    # -------------------- 5.1 生成仿真数据 --------------------
    np.random.seed(42)
    n = 600
    t = np.linspace(0, 30, n)
    # 复合信号：正弦波 + 线性趋势 + 噪声
    signal = 5 * np.sin(0.5 * t) + 0.1 * t + 1.0 * np.random.randn(n)
    values = signal.reshape(-1, 1)   # 单变量，形状 (n, 1)

    # 标准化
    scaler = StandardScaler()
    values_scaled = scaler.fit_transform(values)

    # 划分训练/测试
    train_size = int(n * 0.8)
    train_data = values_scaled[:train_size]
    test_data = values_scaled[train_size:]

    # 构造序列样本 (用过去 8 个点预测下一个)
    seq_length = 8
    X_train, y_train = create_sequences(train_data, train_data, seq_length)
    X_test, y_test = create_sequences(test_data, test_data, seq_length)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    print(f"训练样本: {X_train.shape[0]}, 测试样本: {X_test.shape[0]}")

    # -------------------- 5.2 初始化 TCN 模型 --------------------
    input_size = 1
    hidden_size = 32
    output_size = 1
    num_layers = 3
    kernel_size = 3

    model = TCNNetwork(input_size, hidden_size, output_size, seq_length, num_layers=num_layers, kernel_size=kernel_size)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # -------------------- 5.3 训练循环 --------------------
    epochs = 150
    train_losses, test_losses = [], []

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

        # 测试集评估
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for i in range(X_test.shape[0]):
                y_pred = model(X_test[i])
                test_loss += criterion(y_pred, y_test[i]).item()
        avg_test_loss = test_loss / X_test.shape[0]
        test_losses.append(avg_test_loss)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}  Train Loss: {avg_train_loss:.6f}  Test Loss: {avg_test_loss:.6f}")

    # -------------------- 5.4 评估与反标准化 --------------------
    model.eval()
    y_pred_list, y_true_list = [], []
    with torch.no_grad():
        for i in range(X_test.shape[0]):
            y_pred_list.append(model(X_test[i]).item())
            y_true_list.append(y_test[i].item())

    y_pred = np.array(y_pred_list).reshape(-1, 1)
    y_true = np.array(y_true_list).reshape(-1, 1)

    y_pred_orig = scaler.inverse_transform(y_pred)
    y_true_orig = scaler.inverse_transform(y_true)

    rmse = np.sqrt(np.mean((y_true_orig - y_pred_orig) ** 2))
    print(f"\n测试集 RMSE = {rmse:.4f} (原始尺度)")

    # -------------------- 5.5 可视化 --------------------
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(test_losses, label='Test Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Loss Curves (TCN)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(y_true_orig, y_pred_orig, alpha=0.6, edgecolors='k', linewidth=0.5)
    min_val = min(y_true_orig.min(), y_pred_orig.min())
    max_val = max(y_true_orig.max(), y_pred_orig.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 时间序列对比 (前100个测试点)
    plt.figure(figsize=(14, 4))
    n_show = min(100, len(y_true_orig))
    plt.plot(y_true_orig[:n_show], label='True', linewidth=2, color='green')
    plt.plot(y_pred_orig[:n_show], label='Predicted', linewidth=2, linestyle='--', color='red')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('预测 vs 真实 (前100个测试点)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print("\n演示完成！")

if __name__ == "__main__":
    main()