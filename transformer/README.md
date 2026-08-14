# AMCNet-Transformer（PyTorch 改造版）

将原 JWSMC 项目（TensorFlow/Keras）中 AMCNet的 LSTM 序列建模层替换为 Transformer Encoder。
原项目代码与模型全部保持原样未改动，本目录是独立的一份新实现。

## 文件说明

| 文件 | 作用 |
|------|------|
| `data_loader.py` | 数据加载。**完全复刻** v3 的归一化/划分/类别权重/低SNR样本权重逻辑，保证对比公平 |
| `model.py` | 模型定义：ConvBlock + ChannelAttention + PositionalEncoding + TransformerBlock + AMCNetTransformer |
| `train.py` | 训练脚本（对齐 v3 的优化器/调度/早停策略） |
| `evaluate.py` | 评估脚本：独立测试集 + 按SNR逐档 + 总体/低SNR/高SNR汇总 + 曲线图 |

## 环境与运行

```bash
conda activate torch_gpu        # PyTorch 2.10 + CUDA
cd /home/mywsl/jwsmc/transformer

# 1. 验证数据加载
python data_loader.py

# 2. 验证模型前向（用随机数据）
python model.py

# 3. 全量训练 150 epochs（RTX 3060 预计 20-40 分钟）
python train.py

# 4. 评估（对照 v3 LSTM 基线）
python evaluate.py --ckpt checkpoints/amcnet_transformer_best.pt
```

## 模型结构（对比 v3）

```
输入 (256, 2) I/Q
  → CNN Stem 4块（与v3一致）          → (16, 256) 即 16个token × 256维
  → 通道注意力（与v3一致）
  → 正弦位置编码                       ← 新增
  → Transformer Encoder × N层          ← 替换 LSTM
  → GAP 序列池化
  → MLP 分类头（与v3一致）             → 4类 softmax
```

默认超参：`d_model=256, nhead=4, num_layers=2, dim_ff=1024, dropout=0.1`

## 调参顺序建议

若未达标（总体≥84% / 低SNR≥60% / 高SNR≥94%）：
1. `--num-layers 4`（Encoder 加深）
2. `--dropout 0.2`（防过拟合）
3. 位置编码换成可学习式（在 model.py 中把 PositionalEncoding 改为 nn.Parameter）
4. 尝试去掉通道注意力或调 GAP → CLS token

## 注意事项

- `evaluate.py` 的 `--num-layers` 必须与训练时一致，否则加载权重会报错
- 训练/评估输出在 `checkpoints/`、`results/` 目录下，不污染原项目
