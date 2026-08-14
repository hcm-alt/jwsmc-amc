"""
AMCNet-Transformer 模型定义（PyTorch 版）
================================================
将原 AMCNet v3 中的 LSTM 序列建模层替换为 Transformer Encoder。
其余部分（CNN 特征提取器、通道注意力、MLP 分类头）与原版保持一致，
这样能隔离"序列建模方式"这一个变量，做公平对比。

整体结构（输入 -> 输出）:
    (B, 256, 2) I/Q序列
      └─> CNN Stem: 4 个卷积块(Conv1d+BN+ReLU+MaxPool)，复刻 v3
            → (B, 256, 16)   [C=256 通道, T=16 时间步]
      └─> 通道注意力(SE-like): 对 256 个通道加权（可选）
      └─> 转置 -> (B, 16, 256)  [T 个 token, 每个 256 维]
      └─> 位置编码(正弦式)
      └─> Transformer Encoder: N 个 TransformerBlock (自注意力+前馈网络)
      └─> 序列池化: 对 T 个 token 取均值 (GAP) -> (B, 256)
      └─> MLP 分类头（与 v3 完全一致）-> (B, 4)

对比原 v3 的变化:
    原: ... CNN Stem -> 通道注意力 -> LSTM(256) -> LayerNorm -> MLP头
    新: ... CNN Stem -> 通道注意力 -> 位置编码 -> Transformer×N -> GAP -> MLP头
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. 卷积块：复刻 v3 的 "2×(Conv1d+BN+ReLU) + MaxPool + Dropout"
# =====================================================================
class ConvBlock(nn.Module):
    """CNN 特征提取块。

    与 v3 的每个卷积块结构一致：
        Conv1d -> BN -> ReLU  （重复 num_convs 次，默认 2 次）
        MaxPool1d(2)           （时间步减半: 256->128->64->32->16）
        Dropout

    Args:
        in_channels: 输入通道数（第1个块=2, 因为输入是 I/Q 双通道）
        out_channels: 输出通道数（64 -> 128 -> 256 -> 256）
        num_convs: 块内卷积层数（前3块=2, 第4块=1, 与 v3 一致）
        kernel_size: 卷积核大小, 默认 3（padding 保持长度不变）
        dropout: 该块最后的 Dropout 概率
    """
    def __init__(self,in_channels,out_channels,num_convs=2,
                 kernel_size=3,dropout=0.2):
        super().__init__()
        convs = []
        for i in range(num_convs):
            cin = in_channels if i == 0 else out_channels
            convs += [
                nn.Conv1d(cin,out_channels,kernel_size,
                          padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
            ]

        self.convs = nn.Sequential(*convs)
        self.pool = nn.MaxPool1d(2)          # 时间长度减半
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        x = self.convs(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x

# =====================================================================
# 2. 通道注意力（SE-like）：复刻 v3 的通道注意力模块
# =====================================================================
class ChannelAttention(nn.Module):
    """通道注意力（Squeeze-and-Excitation）。

    与 v3 的逻辑一致：
        GAP(全局平均池化) -> Dense(128, relu) -> Dense(256, sigmoid) -> 通道加权

    作用: 让网络学会"哪些特征通道更重要"，对低SNR信号尤其有帮助。
    输入: (B, C, T)
    输出: (B, C, T)（对通道维度加权后的特征）
    """
    def __init__(self,channels,reduction_ratio=2):
        super().__init__()
        hidden = channels // reduction_ratio     # 256 -> 128
        self.gap = nn.AdaptiveAvgPool1d(1)       # 把 T 个时间步压成 1 个
        self.fc1 = nn.Linear(channels,hidden)
        self.fc2 = nn.Linear(hidden,channels)

    def forward(self,x):                         # x: (B, C, T)
        w = self.gap(x).squeeze(-1)              # (B, C, 1) -> (B, C) 全局描述
        w = F.relu(self.fc1(w))                  # 降维
        w = torch.sigmoid(self.fc2(w))           # 归一化到 (0,1) 的通道权重
        return x * w.unsqueeze(-1)               # 逐通道加权 (B, C, T)

# =====================================================================
# 3. 正弦位置编码：给 token 注入"先后顺序"信息
# =====================================================================
class PositionalEncoding(nn.Module):
    """标准正弦位置编码（来自原版 Transformer 论文）。

    为什么需要位置编码?
        - 自注意力本身是"位置无关"的（它只计算两两 token 的相似度）
        - 不告诉模型"谁在前谁在后"，模型就分不清时间顺序
        - 而 I/Q 信号的符号序列是有先后关系的，顺序信息至关重要

    实现: 对每个位置 pos 和每个维度 i，生成固定的正弦/余弦值。
        偶维: PE(pos, 2i)   = sin(pos / 10000^(2i/d))
        奇维: PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    特点是"位置差异在低维用高频、高维用低频"表示，学习/泛化更稳定。
    这里用 register_buffer 注册，随模型一起保存，但不参与梯度更新。
    每一个时间步用一个位置编码（行）表示，一个位置编码由256个维度（列）表示

    Args:
        d_model: 特征维度（= token 的向量维度, 本模型 256）
        max_len: 支持的最大序列长度（本模型 T<=16, 留足余量用 64）
    """
    def __init__(self,d_model,max_len=64):
        super().__init__()
        pe = torch.zeros(max_len,d_model)
        position = torch.arange(max_len,dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2,dtype=torch.float32)
                             *(-math.log(10000.0) / d_model))
        pe[:,0::2] = torch.sin(position * div_term)  # 偶数维度用 sin
        pe[:,1::2] = torch.cos(position * div_term)  # 奇数维度用 cos
        self.register_buffer('pe',pe)   # 不参与训练、随模型保存

    def forward(self,x):                 # x: (B, T, d_model)(批次大小是32,池化后的时间步（采样点）长度是16,特征维度是256)
        return x + self.pe[:x.size(1)]   # 截取pe的前 T=16 行，加到x上，完成位置编码

# =====================================================================
# 4. Transformer Encoder 块（手写的 pre-LN 结构）
# =====================================================================
class TransformerBlock(nn.Module):
    """单层 Transformer Encoder（pre-LN 结构）。

    结构（与论文 "Attention Is All You Need" 的 Encoder 层一致）:
        ①  LayerNorm
        ②  多头自注意力 MultiHeadSelfAttention(q=v=k=自己)
        ③  残差连接: x = x + Dropout(注意力输出)
        ④  LayerNorm
        ⑤  前馈网络 FFN: Linear(256->1024) + GELU + Linear(1024->256)
        ⑥  残差连接: x = x + Dropout(FFN输出)

    为什么用 pre-LN（先归一化再进子层）?
        - 相比原论文的 post-LN，pre-LN 训练更稳定、对学习率不那么敏感
        - 这是现代 Transformer 训练实践（如 GPT、BERT 变体）的常用选择

    Args:
        d_model: 特征维度
        nhead: 注意力头数
        dim_ff: 前馈网络隐藏层维度（通常 4×d_model）
        dropout: Dropout 概率
    """
    def __init__(self,d_model,nhead,dim_ff,dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model,nhead,
                                dropout=dropout,batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model,dim_ff),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(dim_ff,d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):     # x: (B, T, d_model) 
         # ---- 子层1: 自注意力 + 残差 ----
        normed = self.norm1(x)
        attn_out,_ = self.self_attn(normed,normed,normed)  # q=k=v
        x = x + self.dropout(attn_out) 
        # ---- 子层2: 前馈网络 + 残差 ----
        x = x + self.dropout(self.ffn(self.norm2(x)))  # 残差连接
        return x

# =====================================================================
# 5. AMCNet-Transformer 主模型
# =====================================================================
class AMCNetTransformer(nn.Module):
    """AMCNet-Transformer 主模型（用于 4 类调制识别）。

    结构（见文件头部示意图）：
        CNN Stem（4块） -> 通道注意力 -> 位置编码
        -> Transformer Encoder(N层) -> GAP -> MLP头

    Args:
        in_channels: 输入通道数，固定 2（I/Q）
        d_model: 特征维度（=CNN输出通道数, =Transformer的token维度）
        nhead: 注意力头数
        num_layers: Transformer Encoder 层数（建议先 2，再试 4）
        dim_ff: 前馈网络隐藏维度
        dropout: Transformer 内部 Dropout
        num_classes: 调制类别数（4）
        use_channel_attn: 是否保留 v3 的通道注意力
        max_len: 位置编码支持的最大序列长度
    """
    def __init__(self,in_channels=2,d_model=256,nhead=4,num_layers=2,
                 dim_ff=1024, dropout=0.1,num_classes=4,
                 use_channel_attn=True,max_len=64):
        super().__init__()

        # ---- CNN 特征提取器（复刻 v3 的 4 个卷积块）----
        # 输入 (B, 2, 256)，逐块下采样: 256 -> 128 -> 64 -> 32 -> 16
        self.stem = nn.Sequential(
            ConvBlock(in_channels, 64, num_convs=2, dropout=0.2),
            ConvBlock(64, 128,  num_convs=2, dropout=0.2),
            ConvBlock(128, 256, num_convs=2, dropout=0.2),
            ConvBlock(256, 256, num_convs=1, dropout=0.0)
        )
        self.use_channel_attn = use_channel_attn
        if use_channel_attn:
            self.channel_attn = ChannelAttention(d_model)

        # ---- Transformer 部分（本次改造的核心）----
        self.pos_enc = PositionalEncoding(d_model,max_len=max_len)
        self.encoder = nn.Sequential(*[TransformerBlock(d_model,nhead,dim_ff,dropout) 
                                       for _ in range(num_layers)])

        # ---- MLP 分类头（与 v3 完全一致）----
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, 128),    nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):                   # x: (B, 256, 2)
        # Conv1d 需要 (B, 通道, 长度)，而输入是 (B, 长度, 通道)，先转置
        x = x.transpose(1, 2)               # (B, 2, 256)
        x = self.stem(x)                    # (B, 256, 16)
        if self.use_channel_attn:
            x = self.channel_attn(x)        # 通道注意力加权
        x = x.transpose(1, 2)               # (B, 16, 256): T=16个token, 每个256维
        x = self.pos_enc(x)                 # 加位置编码
        x = self.encoder(x)                 # N 层 Transformer
        x = x.mean(dim=1)                   # GAP: 对16个token取平均 -> (B, 256)
        return self.head(x)                 # (B, 4)


# =====================================================================
# 6. 独立运行本文件时: 用随机数据做一次前向验证
# =====================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    model = AMCNetTransformer(num_layers=2)     # 默认 2 层 Encoder
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")
    # 随机生成一个 batch 的输入 (B=4, 256, 2)，模拟真实数据形状
    dummy = torch.randn(4, 256, 2)
    out = model(dummy)
    print(f"输入: {tuple(dummy.shape)} -> 输出: {tuple(out.shape)}  (应输出 4 类分数)")
    assert out.shape == (4, 4), "输出形状不对！请检查模型结构"
    print("前向传播验证通过 ✓")

