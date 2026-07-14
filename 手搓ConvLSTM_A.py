import matplotlib
matplotlib.use('TkAgg')

import os
import torch
import imageio
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset


# ========================== 1. ConvLSTM 核心组件 ==========================
class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, use_peephole=True):
        super(ConvLSTMCell, self).__init__()
        self.out_channels = out_channels
        self.use_peephole = use_peephole

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(padding, int):
            padding = (padding, padding)

        # 输入到隐藏的卷积
        self.conv_xi = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_xf = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_xc = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_xo = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

        # 隐藏到隐藏的卷积
        self.conv_hi = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.conv_hf = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.conv_hc = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.conv_ho = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)

        # 偏置 (1x1 可广播)
        self.b_i = nn.Parameter(torch.zeros(out_channels, 1, 1))
        self.b_f = nn.Parameter(torch.zeros(out_channels, 1, 1))
        self.b_c = nn.Parameter(torch.zeros(out_channels, 1, 1))
        self.b_o = nn.Parameter(torch.zeros(out_channels, 1, 1))

        # 窥视孔权重
        if use_peephole:
            self.Wci = nn.Parameter(torch.zeros(out_channels, 1, 1))
            self.Wcf = nn.Parameter(torch.zeros(out_channels, 1, 1))
            self.Wco = nn.Parameter(torch.zeros(out_channels, 1, 1))
        else:
            self.register_parameter('Wci', None)
            self.register_parameter('Wcf', None)
            self.register_parameter('Wco', None)

    def forward(self, x, h_prev, c_prev):
        i = torch.sigmoid(self.conv_xi(x) + self.conv_hi(h_prev) + (self.Wci * c_prev if self.use_peephole else 0) + self.b_i)
        f = torch.sigmoid(self.conv_xf(x) + self.conv_hf(h_prev) + (self.Wcf * c_prev if self.use_peephole else 0) + self.b_f)
        c_tilde = torch.tanh(self.conv_xc(x) + self.conv_hc(h_prev) + self.b_c)
        c = f * c_prev + i * c_tilde
        o = torch.sigmoid(self.conv_xo(x) + self.conv_ho(h_prev) + (self.Wco * c if self.use_peephole else 0) + self.b_o)
        h = o * torch.tanh(c)
        return h, c


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_sizes, num_layers,
                 batch_first=True, return_all_layers=False, use_peephole=True):
        super(ConvLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.kernel_sizes = kernel_sizes
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(num_layers):
            in_ch = input_dim if i == 0 else hidden_dims[i-1]
            out_ch = hidden_dims[i]
            ks = kernel_sizes[i] if i < len(kernel_sizes) else kernel_sizes[-1]
            pad = ks[0]//2 if isinstance(ks, tuple) else ks//2
            cell = ConvLSTMCell(in_ch, out_ch, ks, pad, use_peephole)
            cell_list.append(cell)
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)  # (seq, b, c, h, w)
            seq_len, batch_size, _, h, w = input_tensor.size()
        else:
            batch_size, seq_len, _, h, w = input_tensor.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size, (h, w))

        layer_output_list = []
        layer_hidden_list = []
        cur_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                x_t = cur_input[:, t, :, :, :] if self.batch_first else cur_input[t, :, :, :, :]
                h, c = self.cell_list[layer_idx](x_t, h, c)
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1 if self.batch_first else 0)
            cur_input = layer_output
            layer_output_list.append(layer_output)
            layer_hidden_list.append((h, c))   # 每层最后一个时刻的状态

        if not self.return_all_layers:
            # 只返回最后一层的输出和最后一层的最后状态
            return layer_output_list[-1], layer_hidden_list[-1]
        return layer_output_list, layer_hidden_list

    def _init_hidden(self, batch_size, spatial_dim):
        h, w = spatial_dim
        hidden_state = []
        for i in range(self.num_layers):
            dim = self.hidden_dims[i]
            device = self.cell_list[0].conv_xi.weight.device
            hidden_state.append((
                torch.zeros(batch_size, dim, h, w, device=device),
                torch.zeros(batch_size, dim, h, w, device=device)
            ))
        return hidden_state


# ===================== 2. 预测模型（编码器-预测器） =====================
class MovingMNIST_Predictor(nn.Module):
    def __init__(self, input_dim=1, hidden_dims=[64, 64, 64], kernel_size=3,
                 num_layers=3, pred_steps=10):
        super(MovingMNIST_Predictor, self).__init__()
        self.pred_steps = pred_steps
        self.num_layers = num_layers
        self.hidden_dims = hidden_dims

        # 编码器：返回所有层的状态（用于初始化预测器）
        self.encoder = ConvLSTM(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            kernel_sizes=[kernel_size] * num_layers,
            num_layers=num_layers,
            batch_first=True,
            return_all_layers=True,      # 改为 True，返回所有层输出和状态
            use_peephole=True
        )

        # 预测器：结构与编码器相同，但独立参数
        self.forecaster = ConvLSTM(
            input_dim=hidden_dims[-1],   # 第一层输入为编码器最后一层的输出通道
            hidden_dims=hidden_dims,
            kernel_sizes=[kernel_size] * num_layers,
            num_layers=num_layers,
            batch_first=True,
            return_all_layers=False,     # 只需要最后一层输出
            use_peephole=True
        )

        self.output_conv = nn.Conv2d(hidden_dims[-1], input_dim, kernel_size=1)

    def forward(self, x):
        # x: (batch, seq_in, 1, H, W)
        # 编码器输出所有层
        enc_outputs, enc_hidden = self.encoder(x)  # enc_hidden: list of (h, c) for each layer

        # 预测器的初始状态 = 编码器各层的最后状态（一一对应）
        fore_states = enc_hidden  # 直接复用，格式一致

        # 预测器的初始输入：编码器最后一层最后一个时刻的输出
        dec_input = enc_outputs[-1][:, -1, :, :, :]  # (batch, hidden_dim, H, W)

        predictions = []
        for _ in range(self.pred_steps):
            dec_input_seq = dec_input.unsqueeze(1)  # (batch, 1, hidden_dim, H, W)
            # 预测器单步前向，返回最后一层输出和更新后的状态列表
            pred_out, fore_states = self._forecaster_step(dec_input_seq, fore_states)
            img_pred = self.output_conv(pred_out[:, 0, :, :, :])  # (batch, 1, H, W)
            predictions.append(img_pred)
            dec_input = pred_out[:, 0, :, :, :]   # 下一时刻的输入

        pred_seq = torch.stack(predictions, dim=1)  # (batch, pred_steps, 1, H, W)
        return pred_seq

    def _forecaster_step(self, x_seq, hidden_state):
        """
        单步执行预测器（逐层手动循环）
        x_seq: (batch, 1, in_ch, h, w)
        hidden_state: list of (h, c) for each layer
        返回：(最后一层输出, 新的状态列表)
        """
        batch_size, seq_len, in_ch, h, w = x_seq.size()
        cur_input = x_seq
        new_hidden = []
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]          # 获取该层当前状态
            x_t = cur_input[:, 0, :, :, :]          # 该层的输入（只有一个时间步）
            h_new, c_new = self.forecaster.cell_list[layer_idx](x_t, h, c)
            layer_out = h_new.unsqueeze(1)          # (batch, 1, out_ch, h, w)
            cur_input = layer_out                   # 下一层的输入
            new_hidden.append((h_new, c_new))
        # 最后一层的输出即为 layer_out，状态列表为 new_hidden
        return layer_out, new_hidden


# ===================== 3. 手写数字运动数据集 =====================
class MovingMNISTDataset(Dataset):
    def __init__(self, num_samples=10000, seq_len=20, img_size=64, digit_size=28,
                 train=True):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.img_size = img_size
        self.digit_size = digit_size

        # 从 torchvision 加载 MNIST
        try:
            from torchvision.datasets import MNIST
            from torchvision import transforms
            mnist = MNIST(root='./data', train=train, download=True,
                          transform=transforms.ToTensor())
            self.data = [mnist[i][0].squeeze().numpy() for i in range(min(num_samples, len(mnist)))]
        except:
            print("Warning: torchvision not found, using random circles.")
            self.data = [self._gen_random_circle() for _ in range(min(num_samples, 1000))]

        self.num_digits = len(self.data)

    def _gen_random_circle(self):
        img = np.zeros((self.digit_size, self.digit_size), dtype=np.float32)
        r = np.random.randint(5, 12)
        cx, cy = np.random.randint(r, self.digit_size-r, size=2)
        y, x = np.ogrid[:self.digit_size, :self.digit_size]
        mask = (x - cx)**2 + (y - cy)**2 <= r**2
        img[mask] = 1.0
        return img

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        digit = self.data[idx % self.num_digits]
        pos_x = np.random.randint(0, self.img_size - self.digit_size)
        pos_y = np.random.randint(0, self.img_size - self.digit_size)
        vel_x = np.random.uniform(0.5, 2.0) * (1 if np.random.rand() > 0.5 else -1)
        vel_y = np.random.uniform(0.5, 2.0) * (1 if np.random.rand() > 0.5 else -1)

        frames = []
        for _ in range(self.seq_len):
            frame = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            if pos_x + self.digit_size > self.img_size or pos_x < 0:
                vel_x = -vel_x
            if pos_y + self.digit_size > self.img_size or pos_y < 0:
                vel_y = -vel_y
            pos_x += vel_x
            pos_y += vel_y
            pos_x = np.clip(pos_x, 0, self.img_size - self.digit_size)
            pos_y = np.clip(pos_y, 0, self.img_size - self.digit_size)
            x0, y0 = int(pos_x), int(pos_y)
            frame[y0:y0+self.digit_size, x0:x0+self.digit_size] = digit
            frames.append(frame)

        sequence = np.stack(frames, axis=0)  # (seq_len, H, W)
        sequence = torch.FloatTensor(sequence).unsqueeze(1)  # (seq_len, 1, H, W)
        return sequence


# ===================== 4. 动图生成函数 =====================
def save_prediction_gif(model, dataset, device, input_steps=10, pred_steps=10, save_path='prediction.gif', num_frames=10):
    """从数据集中取一个样本，生成输入、真实、预测三行动图"""
    model.eval()
    with torch.no_grad():
        # 取第一个样本
        sample = dataset[0]  # (seq_len, 1, H, W)
        seq_len = sample.size(0)
        input_seq = sample[:input_steps].unsqueeze(0).to(device)  # (1, input_steps, 1, H, W)
        target_seq = sample[input_steps:input_steps+pred_steps].unsqueeze(0)  # (1, pred_steps, 1, H, W)
        pred_seq = model(input_seq).cpu()  # (1, pred_steps, 1, H, W)

        input_frames = input_seq.squeeze(0).cpu().numpy()  # (input_steps, 1, H, W)
        target_frames = target_seq.squeeze(0).cpu().numpy()
        pred_frames = pred_seq.squeeze(0).numpy()

        # 生成逐帧图片
        temp_dir = 'temp_frames'
        os.makedirs(temp_dir, exist_ok=True)
        frames = []
        for t in range(num_frames):
            fig, axes = plt.subplots(3, 1, figsize=(1.5, 4.5))
            # 输入帧：显示输入序列的最后 num_frames 帧（如果输入不足则循环）
            if input_steps >= num_frames:
                idx = input_steps - num_frames + t
            else:
                idx = t if t < input_steps else input_steps-1
            axes[0].imshow(input_frames[idx, 0], cmap='gray')
            axes[0].axis('off')
            axes[0].set_title('Input', fontsize=8)

            if t < pred_steps:
                axes[1].imshow(target_frames[t, 0], cmap='gray')
                axes[2].imshow(pred_frames[t, 0], cmap='gray')
            else:
                axes[1].axis('off')
                axes[2].axis('off')
            axes[1].axis('off')
            axes[2].axis('off')
            axes[1].set_title('True', fontsize=8)
            axes[2].set_title('Pred', fontsize=8)
            plt.tight_layout()
            fname = os.path.join(temp_dir, f'frame_{t:03d}.png')
            plt.savefig(fname, dpi=80)
            plt.close(fig)
            frames.append(imageio.imread(fname))

        # 合成GIF
        imageio.mimsave(save_path, frames, fps=4)
        print(f"GIF saved to {save_path}")

        # 清理临时文件夹
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
        os.rmdir(temp_dir)


# ===================== 5. 训练主程序 =====================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 超参数
    batch_size = 16
    seq_len = 15        # 总长度
    input_steps = 10    # 输入帧数
    pred_steps = 5     # 预测帧数
    hidden_dims = [64]
    num_layers = 1
    kernel_size = 3
    lr = 1e-3
    epochs = 2         # 若时间不足可减少

    # 数据
    train_set = MovingMNISTDataset(num_samples=1000, seq_len=seq_len, train=True)
    val_set = MovingMNISTDataset(num_samples=100, seq_len=seq_len, train=False)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    # 模型
    model = MovingMNIST_Predictor(
        input_dim=1,
        hidden_dims=hidden_dims,
        kernel_size=kernel_size,
        num_layers=num_layers,
        pred_steps=pred_steps
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print("Start training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch in pbar:
            x = batch[:, :input_steps, :, :, :].to(device)
            target = batch[:, input_steps:input_steps+pred_steps, :, :, :].to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1} Avg Loss: {avg_loss:.6f}')

        # 每5个epoch保存静态对比图
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_batch = next(iter(val_loader))
                x_val = val_batch[:1, :input_steps].to(device)
                target_val = val_batch[:1, input_steps:input_steps+pred_steps].to(device)
                pred_val = model(x_val).cpu()
                fig, axes = plt.subplots(2, pred_steps, figsize=(15, 3))
                for i in range(pred_steps):
                    axes[0, i].imshow(pred_val[0, i, 0].numpy(), cmap='gray')
                    axes[0, i].axis('off')
                    axes[1, i].imshow(target_val[0, i, 0].cpu().numpy(), cmap='gray')
                    axes[1, i].axis('off')
                axes[0, 0].set_title('Predicted')
                axes[1, 0].set_title('Ground Truth')
                plt.savefig(f'pred_epoch_{epoch+1}.png')
                plt.close()

    print("Training complete!")

    # 生成测试动图
    print("Generating prediction GIF on a test sample...")
    save_prediction_gif(model, val_set, device, input_steps, pred_steps, save_path='prediction.gif', num_frames=pred_steps)


if __name__ == "__main__":
    train()