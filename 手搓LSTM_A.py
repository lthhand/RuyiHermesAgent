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

# =========================== 2. 自定义 LSTM 单元 ===========================
class LSTMCell(nn.Module):
    """
    公式：
        i_t = σ(W_i·x_t + U_i·h_{t-1} + b_i)
        f_t = σ(W_f·x_t + U_f·h_{t-1} + b_f)
        o_t = σ(W_o·x_t + U_o·h_{t-1} + b_o)
        g_t = tanh(W_g·x_t + U_g·h_{t-1} + b_g)
        c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
        h_t = o_t ⊙ tanh(c_t)
    """
    def __init__(self, input_size, hidden_size):
        """
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏状态维度
        """
        super(LSTMCell, self).__init__()
        self.hidden_size = hidden_size

        # 使用 Xavier 初始化，有助于训练收敛
        scale = np.sqrt(2.0 / (input_size + hidden_size))

        # ---------- 输入门 i_t = σ(W_i·x_t + U_i·h_{t-1} + b_i) ----------
        self.W_i = nn.Parameter(torch.randn(hidden_size, input_size) * scale)
        self.U_i = nn.Parameter(torch.randn(hidden_size, hidden_size) * scale)
        self.b_i = nn.Parameter(torch.zeros(hidden_size, 1))

        # ---------- 遗忘门 f_t = σ(W_f·x_t + U_f·h_{t-1} + b_f) ----------
        self.W_f = nn.Parameter(torch.randn(hidden_size, input_size) * scale)
        self.U_f = nn.Parameter(torch.randn(hidden_size, hidden_size) * scale)
        self.b_f = nn.Parameter(torch.zeros(hidden_size, 1))

        # ---------- 输出门 o_t = σ(W_o·x_t + U_o·h_{t-1} + b_o) ----------
        self.W_o = nn.Parameter(torch.randn(hidden_size, input_size) * scale)
        self.U_o = nn.Parameter(torch.randn(hidden_size, hidden_size) * scale)
        self.b_o = nn.Parameter(torch.zeros(hidden_size, 1))

        # ---------- 候选记忆 g_t = tanh(W_g·x_t + U_g·h_{t-1} + b_g) ----------
        self.W_g = nn.Parameter(torch.randn(hidden_size, input_size) * scale)
        self.U_g = nn.Parameter(torch.randn(hidden_size, hidden_size) * scale)
        self.b_g = nn.Parameter(torch.zeros(hidden_size, 1))

    def forward(self, x, h_prev, c_prev):
        """
        单步前向传播。

        Args:
            x     : 当前时刻的输入，形状 (input_size, 1)
            h_prev: 上一时刻的隐藏状态，形状 (hidden_size, 1)
            c_prev: 上一时刻的记忆单元，形状 (hidden_size, 1)

        Returns:
            h_next: 当前时刻的隐藏状态
            c_next: 当前时刻的记忆单元

        公式：
            i_t = σ(W_i·x_t + U_i·h_{t-1} + b_i)
            f_t = σ(W_f·x_t + U_f·h_{t-1} + b_f)
            o_t = σ(W_o·x_t + U_o·h_{t-1} + b_o)
            g_t = tanh(W_g·x_t + U_g·h_{t-1} + b_g)
            c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
            h_t = o_t ⊙ tanh(c_t)
        """
        # 1) 输入门
        i = torch.sigmoid(self.W_i @ x + self.U_i @ h_prev + self.b_i)
        # 2) 遗忘门
        f = torch.sigmoid(self.W_f @ x + self.U_f @ h_prev + self.b_f)
        # 3) 输出门
        o = torch.sigmoid(self.W_o @ x + self.U_o @ h_prev + self.b_o)
        # 4) 候选记忆
        g = torch.tanh(self.W_g @ x + self.U_g @ h_prev + self.b_g)
        # 5) 更新记忆单元：c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
        c_next = f * c_prev + i * g
        # 6) 更新隐藏状态：h_t = o_t ⊙ tanh(c_t)
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


# =========================== 3. 定义整个 LSTM 网络 ===========================
class LSTMNetwork(nn.Module):
    """
    单层 LSTM 网络，用于回归预测。
    包含一个 LSTM 单元和一个输出全连接层。
    输入是一个序列（过去 seq_length 个时间步），输出是下一个时间步的预测值。
    """
    def __init__(self, input_size, hidden_size, output_size, seq_length):
        """
        Args:
            input_size : 每个时间步输入的特征数
            hidden_size: LSTM 隐藏层维度
            output_size: 输出维度（回归任务通常为 1）
            seq_length : 输入序列的长度（用过去多少个点预测未来一个点）
        """
        super(LSTMNetwork, self).__init__()
        self.seq_length = seq_length
        self.hidden_size = hidden_size

        # 实例化 LSTM 单元
        self.cell = LSTMCell(input_size, hidden_size)

        # 输出层：将最后一个时间步的隐藏状态映射到预测值
        scale = np.sqrt(2.0 / hidden_size)
        self.W_out = nn.Parameter(torch.randn(output_size, hidden_size) * scale)
        self.b_out = nn.Parameter(torch.zeros(output_size, 1))

    def forward(self, X):
        """
        前向传播整个序列。

        Args:
            X: 输入序列，形状为 (seq_length, input_size, 1)
               注意：每个时间步 x_t 的形状是 (input_size, 1)

        Returns:
            y_pred: 预测值，形状为 (output_size, 1)
        """
        # 初始化隐藏状态 h 和记忆单元 c 为零向量
        h = torch.zeros(self.hidden_size, 1)
        c = torch.zeros(self.hidden_size, 1)

        # 沿时间步依次处理
        for t in range(self.seq_length):
            x_t = X[t]          # 取第 t 个时间步的输入
            h, c = self.cell(x_t, h, c)   # 更新 LSTM 状态

        # 取最后一个时间步的隐藏状态进行预测
        y_pred = self.W_out @ h + self.b_out
        return y_pred


# =========================== 4. 辅助函数：构造时间序列样本 ===========================
def create_sequences(data, target, seq_length):
    """
    将一维时间序列转换为监督学习样本。
    用过去 seq_length 个点预测下一个点。

    Args:
        data   : 形状为 (n_samples, n_features) 的特征数组
        target : 形状为 (n_samples,) 的目标数组（此处与 data 相同，即自回归）
        seq_length: 序列长度

    Returns:
        X : 输入样本，形状 (n_samples - seq_length, seq_length, n_features, 1)
        y : 目标值，形状 (n_samples - seq_length, 1, 1)
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        # 取连续的 seq_length 个点作为输入
        X.append(data[i:i+seq_length])
        # 下一个点作为目标
        y.append(target[i+seq_length])
    X = np.array(X)                  # (n, seq_len, n_features)
    y = np.array(y)                  # (n,)
    X = X[..., np.newaxis]           # 增加一维表示特征通道 (n, seq_len, n_features, 1)
    y = y.reshape(-1, 1, 1)          # (n, 1, 1)
    return X, y


# =========================== 5. 主程序 ===========================
def main():

    # -------------------- 5.1 加载 Mauna Loa CO₂ 数据集 --------------------
    print("正在加载 Mauna Loa CO₂ 数据...")
    # 从 OpenML 获取数据集 (data_id=41187)
    co2 = fetch_openml(data_id=41187, as_frame=True, parser="pandas")
    co2_df = co2.frame
    # 构造日期索引
    co2_df["date"] = pd.to_datetime(co2_df[["year", "month", "day"]])
    co2_df = co2_df[["date", "co2"]].set_index("date")
    # 提取 CO₂ 浓度值（单位：ppm）
    values = co2_df["co2"].values.astype(np.float32).reshape(-1, 1)
    print(f"数据总量: {len(values)} 个样本")
    print(f"时间范围: {co2_df.index.min()} 到 {co2_df.index.max()}")

    # -------------------- 5.2 数据标准化 --------------------
    # 使用 Z-score 标准化，使模型训练更稳定
    scaler = StandardScaler()
    values_scaled = scaler.fit_transform(values)   # 返回 (n_samples, 1)

    # -------------------- 5.3 按时间顺序划分训练集和测试集 --------------------
    # 时间序列数据不能随机打乱，必须保持时间顺序
    train_size = int(len(values_scaled) * 0.8)  # 80% 作为训练集
    train_data = values_scaled[:train_size]
    test_data = values_scaled[train_size:]

    # -------------------- 5.4 构造序列样本 --------------------
    seq_length = 6   # 用过去 6个月（一年）预测下一个月
    X_train, y_train = create_sequences(train_data, train_data, seq_length)
    X_test, y_test = create_sequences(test_data, test_data, seq_length)

    # 转换为 PyTorch 张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    print(f"训练样本数: {X_train.shape[0]}, 测试样本数: {X_test.shape[0]}")

    # -------------------- 5.5 初始化模型、优化器和损失函数 --------------------
    input_size = 1          # 单变量时间序列，每个时间步只有 CO₂ 浓度一个特征
    hidden_size = 96
    output_size = 1
    model = LSTMNetwork(input_size, hidden_size, output_size, seq_length)

    # 使用 Adam 优化器，学习率 0.01
    optimizer = optim.SGD(model.parameters(), lr=0.00001)
    # 回归任务使用均方误差损失
    criterion = nn.MSELoss()

    # -------------------- 5.6 训练循环 --------------------
    epochs = 300
    train_losses = []   # 记录每个 epoch 的训练损失
    test_losses = []    # 记录每个 epoch 的测试损失

    print("\n开始训练...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for i in range(X_train.shape[0]):
            x_seq = X_train[i]   # 形状 (seq_length, 1, 1)
            y_true = y_train[i]  # 形状 (1, 1)

            # 前向传播：计算预测值
            y_pred = model(x_seq)

            # 计算损失
            loss = criterion(y_pred, y_true)

            # 反向传播：PyTorch 自动计算所有参数的梯度
            optimizer.zero_grad()   # 清空之前的梯度
            loss.backward()         # 自动求导
            optimizer.step()        # 更新参数

            total_loss += loss.item()

        avg_train_loss = total_loss / X_train.shape[0]
        train_losses.append(avg_train_loss)

        # 在每个 epoch 结束后评估测试集损失
        model.eval()
        test_loss = 0
        with torch.no_grad():   # 关闭梯度计算，节省内存和计算
            for i in range(X_test.shape[0]):
                y_pred = model(X_test[i])
                test_loss += criterion(y_pred, y_test[i]).item()
        avg_test_loss = test_loss / X_test.shape[0]
        test_losses.append(avg_test_loss)

        # 每 10 个 epoch 打印一次损失值，便于观察收敛情况
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

    # 反标准化，恢复到原始尺度（ppm）
    y_pred_orig = scaler.inverse_transform(y_pred)
    y_true_orig = scaler.inverse_transform(y_true)

    # 计算均方根误差 RMSE
    rmse = np.sqrt(np.mean((y_true_orig - y_pred_orig) ** 2))

    print(f"测试集 RMSE = {rmse:.4f} ppm")

    # -------------------- 5.8 可视化 --------------------
    # 图1：训练损失和测试损失曲线
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(test_losses, label='Test Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Loss Curves during Training')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 图2：预测值与真实值的散点图
    plt.subplot(1, 2, 2)
    plt.scatter(y_true_orig, y_pred_orig, alpha=0.6, edgecolors='k', linewidth=0.5)
    # 绘制理想对角线 (y=x)
    min_val = min(y_true_orig.min(), y_pred_orig.min())
    max_val = max(y_true_orig.max(), y_pred_orig.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('True CO₂ (ppm)')
    plt.ylabel('Predicted CO₂ (ppm)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 图3：测试集上的时间序列预测对比（展示前100个点）
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


# =========================== 6. 程序入口 ===========================
if __name__ == "__main__":
    main()