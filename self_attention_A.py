import torch
import torch.nn.functional as F

# 模拟一个 4 个词的句子，每个词用 3 维向量表示
X = torch.randn(4, 3)   # [seq_len=4, d_model=3]

# 手写 Q、K、V 的权重矩阵（随机初始化，实际训练中可学习）
W_Q = torch.randn(3, 3)
W_K = torch.randn(3, 3)
W_V = torch.randn(3, 3)

# 计算 Query, Key, Value
Q = X @ W_Q   # [4, 3]
K = X @ W_K   # [4, 3]
V = X @ W_V   # [4, 3]

# 缩放点积注意力（没有 mask，适合展示全局注意力）
d_k = Q.size(-1)                                    # 缩放因子
scores = Q @ K.T / torch.sqrt(torch.tensor(d_k))    # 注意力分数矩阵 [4, 4]
attn_weights = F.softmax(scores, dim=-1)            # 每一行是一个查询对所有键的注意力
output = attn_weights @ V                           # 加权求和得到输出 [4, 3]

print("输入序列 X:\n", X)
print("注意力权重矩阵 (4×4):\n", attn_weights)
print("自注意力输出:\n", output)
