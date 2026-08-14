# -*- coding: utf-8 -*-
"""
AMCNet-Transformer 训练脚本（PyTorch 版）
================================================
训练策略完全对齐原版 scripts/train_amcnet_v3.py：
    - 优化器: Adam, lr=1e-3
    - 学习率调度: ReduceLROnPlateau（验证准确率 10 轮不升则减半）
    - 早停: 验证准确率 30 轮不升则停止
    - 损失: 交叉熵 + 类别权重(weight) + 低SNR样本权重(sample_weight=1.3)
    - batch_size=64, epochs=150

与 v3 唯一的区别是: 序列建模层由 LSTM 换成了 Transformer Encoder。

用法:
    conda activate torch_gpu
    cd /home/mywsl/jwsmc/transformer
    python train.py                          # 全量训练 150 epochs
    python train.py --epochs 2               # 快速冒烟测试（2 个 epoch）
    python train.py --num-layers 4           # 试试更深的 Encoder
"""
import argparse
import copy
import os
import pickle
import time

import matplotlib
matplotlib.use('Agg')          # 无显示环境下也能保存图片
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loader import load_amc_data_transformer, AMCDataset
from model import AMCNetTransformer


# =====================================================================
# 1. 工具函数
# =====================================================================
def set_seed(seed=42):
    """固定所有随机种子，保证训练可复现。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """训练一个 epoch。返回 (平均损失, 准确率)。"""
    model.train()                       # 切换到训练模式（启用 Dropout/BN 统计）
    total_loss, total_correct, total = 0.0, 0, 0
    for x, y, w in loader:
        x, y, w = x.to(device), y.to(device), w.to(device)
        optimizer.zero_grad()
        logits = model(x)               # (B, 4)
        loss = criterion(logits, y)     # 逐样本 loss（内部已乘类别权重）
        loss = (loss * w).mean()        # 再乘样本权重（低SNR=1.3）后取平均
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += len(y)
    return total_loss / total, total_correct / total


@torch.no_grad()                        # 验证/测试不计算梯度，省显存、更快
def evaluate(model, loader, criterion, device):
    """在给定数据集上评估。返回 (平均损失, 准确率)。"""
    model.eval()                        # 切换为评估模式
    total_loss, total_correct, total = 0.0, 0, 0
    for x, y, w in loader:
        x, y, w = x.to(device), y.to(device), w.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss = (loss * w).mean()
        total_loss += loss.item() * len(y)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += len(y)
    return total_loss / total, total_correct / total


def plot_history(history, save_path):
    """画出训练/验证的 loss 与 acc 曲线。"""
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history['train_loss'], label='train')
    axes[0].plot(epochs, history['val_loss'], label='val')
    axes[0].set_xlabel('epoch'); axes[0].set_ylabel('loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, history['train_acc'], label='train')
    axes[1].plot(epochs, history['val_acc'], label='val')
    axes[1].set_xlabel('epoch'); axes[1].set_ylabel('accuracy')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# =====================================================================
# 2. 主函数
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="训练 AMCNet-Transformer")
    parser.add_argument("--data-path", default="../data", help="数据集目录")
    parser.add_argument("--save-dir", default="checkpoints", help="模型/曲线保存目录")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-layers", type=int, default=2, help="Transformer Encoder 层数")
    parser.add_argument("--nhead", type=int, default=4, help="注意力头数")
    parser.add_argument("--dim-ff", type=int, default=1024, help="前馈网络维度")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=30, help="早停耐心值")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader 进程数")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    # ---- 1) 数据 ----
    data = load_amc_data_transformer(args.data_path)
    train_ds = AMCDataset(data['x_train'], data['y_train'], data['w_train'])
    val_ds   = AMCDataset(data['x_val'],   data['y_val'])
    test_ds  = AMCDataset(data['x_test'],  data['y_test'])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)

    # ---- 2) 模型 ----
    model = AMCNetTransformer(
        d_model=256, nhead=args.nhead, num_layers=args.num_layers,
        dim_ff=args.dim_ff, dropout=args.dropout, num_classes=4,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")

    # ---- 3) 优化器 + 学习率调度 ----
    # weight_decay 近似替代 v3 中卷积层的 L2 正则化(1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6
    )

    # ---- 4) 损失函数：类别权重 + reduction='none'（为了叠加样本权重）----
    class_w = torch.from_numpy(data['class_weights']).float().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w, reduction='none')

    # ---- 5) 训练循环 ----
    best_val_acc, best_state, wait = 0.0, None, 0
    history = {'train_loss': [], 'train_acc': [],
               'val_loss': [], 'val_acc': [], 'lr': []}

    print(f"\n开始训练 ({args.epochs} epochs, batch={args.batch_size}, lr={args.lr})")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(va_acc)                     # 验证准确率不升则降 lr

        cur_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(tr_loss); history['train_acc'].append(tr_acc)
        history['val_loss'].append(va_loss);   history['val_acc'].append(va_acc)
        history['lr'].append(cur_lr)
        print(f"[{epoch:3d}/{args.epochs}] "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc*100:.2f}% | "
              f"val_loss={va_loss:.4f} val_acc={va_acc*100:.2f}% | "
              f"lr={cur_lr:.2e} | {time.time()-t0:.1f}s")

        # ---- 早停 + 保存最佳模型 ----
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = copy.deepcopy(model.state_dict())   # 深拷贝当前最佳权重
            wait = 0
            torch.save({'model_state_dict': best_state,
                        'val_acc': best_val_acc,
                        'args': vars(args)},
                       os.path.join(args.save_dir, 'amcnet_transformer_best.pt'))
            print(f"  → 新的最佳验证准确率: {best_val_acc*100:.2f}%, 已保存")
        else:
            wait += 1
            if wait >= args.patience:
                print(f"早停触发: 验证准确率 {args.patience} 轮未提升")
                break


    # ---- 6) 保存训练历史 + 曲线 ----
    with open(os.path.join(args.save_dir, 'history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    plot_history(history, os.path.join(args.save_dir, 'training_curves.png'))

    # ---- 7) 用最佳权重在测试集上评估一次 ----
    if best_state is not None:
        model.load_state_dict(best_state)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        print(f"\n最佳模型（验证acc={best_val_acc*100:.2f}%）在独立测试集上: "
              f"acc={te_acc*100:.2f}%")
        print(f"\n训练完成！最佳模型: {args.save_dir}/amcnet_transformer_best.pt")
        print(f"训练曲线: {args.save_dir}/training_curves.png")


if __name__ == "__main__":
    main()

