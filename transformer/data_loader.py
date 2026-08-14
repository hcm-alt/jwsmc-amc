# -*- coding: utf-8 -*-
"""
AMCNet-Transformer 数据加载模块（PyTorch 版）
================================================
本文件完全复刻 scripts/train_amcnet_v3.py 的数据预处理与划分逻辑，
确保 Transformer 模型与 LSTM 基线使用【完全相同】的训练/验证/测试数据，
从而保证 "Transformer vs LSTM" 的对比绝对公平。

数据说明（与原始项目一致）：
    - 输入数据:  data/K=4_modulation_dataset.pkl
        一个 dict，key = SNR(整数dB，-18~+18，间隔2，共19档)，value = (4000, 256, 2) 的 float 数组
        含义: 4000 个 I/Q 信号样本，每个样本 256 个采样点、2 个通道(I/Q)
    - 标签数据:  data/K=4_mod_labelset.pkl
        同样按 SNR 组织的 dict，value = (4000, 4) 的 one-hot 标签
        4 类调制方式: BPSK / QPSK / 16QAM / 64QAM

预处理流程（与 v3 一致）:
    1. 逐 SNR 组做 max-abs 归一化: x = x / max(|x|)
    2. 按 SNR 升序拼接所有组 -> (76000, 256, 2)
    3. 固定随机种子 np.random.seed(42) 后打乱
    4. 按 80% / 10% / 10% 切分为 训练 / 验证 / 测试
    5. 计算类别权重(类别不均衡时使用)与样本权重(低SNR<-6dB的样本权重1.3)
"""
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset


# =====================================================================
# 1. 数据集类：把 numpy 数组包装成 PyTorch Dataset
# =====================================================================
"""   调制识别数据集,
#     每个样本返回三元组:
#         x: (256, 2) 的 float32 张量, I/Q 双通道序列
#         y: 长整型标量, 调制类别索引 (0=BPSK, 1=QPSK, 2=16QAM, 3=64QAM)
#         w: float 标量, 该样本的训练权重 (低SNR样本=1.3, 其余=1.0)        
#     """
class AMCDataset(Dataset):
    def __init__(self,x,y,w=None):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(np.argmax(y,axis=1))
        self.w = (torch.ones(len(x),dtype=torch.float32) 
                  if w is None else torch.from_numpy(w) )

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        # 1. 取出当前样本（已经是 torch.Tensor）
        x, y, w = self.x[i], self.y[i], self.w[i]
        
        # 2. 数据增强：只对低 SNR 样本（权重 > 1.0）进行随机噪声注入
        #    你的权重逻辑是：SNR < -10dB -> 2.0，-10~-6dB -> 1.5，其余 -> 1.0
        #    所以 w > 1.0 代表这个样本是低信噪比的困难样本
        if w > 1.0:
            # 以 60% 的概率加噪声，避免过度增强导致特征失真
            if torch.rand(1).item() < 0.6:
                # 随机生成 0.01 ~ 0.05 的噪声标准差
                noise_std = torch.rand(1).item() * 0.04 + 0.01
                # 生成与 x 同形状的标准正态分布噪声
                noise = torch.randn_like(x) * noise_std
                # 将噪声叠加到原始信号上
                x = x + noise
        
        return x, y, w

# 2. 数据加载函数：复刻 train_amcnet_v3.py 的划分逻辑
"""加载 AMC 数据并按 v3 的规则切分。

    Args:
        data_path: 数据集目录（默认相对路径 "data"，从 transformer/ 目录运行时应传 "../data"）
        seed: 随机种子，默认 42，与 v3 完全一致
        verbose: 是否打印划分信息

    Returns:
        dict, 包含:
            x_train/x_val/x_test: (N,256,2) 归一化后的输入
            y_train/y_val/y_test: (N,4) one-hot 标签
            w_train: (N,) 训练样本权重
            class_weights: (4,) 类别权重
            train_snr_tags/val_snr_tags/test_snr_tags: 每个样本属于哪个 SNR 档位
            snr_list: 升序的 SNR 列表
    """
def load_amc_data_transformer(data_path="data",seed=42,verbose=True):
    with open(f"{data_path}/K=4_modulation_dataset.pkl","rb") as f:
        x_dict = pickle.load(f)              # {SNR: (4000,256,2)}
    with open(f"{data_path}/K=4_mod_labelset.pkl", 'rb') as f:
        y_dict = pickle.load(f)          # {SNR: (4000,4)}

    for snr in x_dict.keys():
        x_dict[snr] = x_dict[snr] / np.max(np.abs(x_dict[snr]))
        x_dict[snr] = x_dict[snr].astype(np.float32)
        y_dict[snr] = y_dict[snr].astype(np.float32)

    # ---- 3) 按 SNR 升序拼接 ----
    snr_list = sorted(x_dict.keys())
    x_all = np.concatenate([x_dict[s] for s in snr_list],axis=0)
    y_all = np.concatenate([y_dict[s] for s in snr_list],axis=0)

    # ---- 4) 固定种子打乱（与 v3 完全一致）----从这里开始写
    np.random.seed(seed)
    indices = np.random.permutation(len(x_all))
    x_all = x_all[indices]
    y_all = y_all[indices]

    # ---- 5) 80% / 10% / 10% 切分 ----
    n = len(x_all)
    n_train = int(n*0.8)
    n_val = int(n*0.1)
    x_train,y_train = x_all[:n_train],y_all[:n_train]
    x_val,y_val = x_all[n_train:n_train+n_val],y_all[n_train:n_train+n_val]
    x_test,y_test = x_all[n_train+n_val:],y_all[n_train+n_val:]


    # ---- 6) 类别权重（类别不均衡时提升少数类的权重）----
    class_counts = np.sum(y_train,axis=0)
    class_weights = len(y_train)/(4.0*class_counts)
    print("各类别训练样本数:",class_counts.astype(int))

# ---- 7) 样本权重：给低SNR样本更高权重（v3 的核心技巧）----
    # 由于打乱前数据是按 SNR 升序拼接的，用 cumsum 即可反查每个原始索引属于哪个 SNR
    snr_sample_count = [len(x_dict[s]) for s in snr_list]          # 每档样本数=4000
    snr_cum = np.cumsum([0] + snr_sample_count)                    # 累计边界
    """根据打乱前的原始索引反查 SNR 档位"""
    def _snr_of_index(idx):
        for j,s in enumerate(snr_list):
            if snr_cum[j] <= idx < snr_cum[j + 1]:
                return s
            return snr_list[-1]

    # 对每个样本用它的"打乱前索引"查 SNR
    train_snr_tags = np.array([_snr_of_index(indices[i]) for i in range(n_train)])
    val_snr_tags   = np.array([_snr_of_index(indices[i]) for i in range(n_train, n_train + n_val)])
    test_snr_tags = np.array(_snr_of_index(indices[i]) for i in range(n_train+n_val,n))

    # SNR < -6 dB 的样本权重 1.3，其余 1.0（与 v3 一致）
    w_train = np.where(train_snr_tags < -10, 2.5,       # 极低SNR: 2.5
                   np.where(train_snr_tags < -6, 1.8,  # 中低SNR: 1.8
                            1.0)).astype(np.float32)   # 其余: 1.0

    if verbose:
        print(f"数据加载完成: 训练集{x_train.shape} 验证集{x_val.shape} 测试集{x_test.shape}")
        print(f"SNR 范围: {snr_list[0]}dB ~ {snr_list[-1]}dB, 共 {len(snr_list)} 档")

    return {
        'x_train': x_train, 'y_train': y_train, 'w_train': w_train,
        'x_val': x_val,     'y_val': y_val,
        'x_test': x_test,   'y_test': y_test,
        'class_weights': class_weights,
        'train_snr_tags': train_snr_tags,
        'val_snr_tags': val_snr_tags,
        'test_snr_tags': test_snr_tags,
        'snr_list': snr_list,
    }


# =====================================================================
# 3. 独立运行本文件时可快速验证数据加载是否正确
# =====================================================================
if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    data = load_amc_data_transformer(os.path.join(here, "..", "data"))
    print("\n类别权重:", data['class_weights'])
    print("训练样本权重: 低SNR数量 =", int((data['w_train'] > 1.0).sum()),
          ", 占比 =", f"{(data['w_train'] > 1.0).mean()*100:.1f}%")
    print("SNR 档位:", data['snr_list'])
    # 用 Dataset 取一个样本检查
    ds = AMCDataset(data['x_train'], data['y_train'], data['w_train'])
    x0, y0, w0 = ds[0]
    print(f"样本检查: x={tuple(x0.shape)} y={int(y0)} w={float(w0)}")

