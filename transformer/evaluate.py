# -*- coding: utf-8 -*-
"""
AMCNet-Transformer 评估脚本（PyTorch 版）
================================================
1. 在"独立测试集"(10% 留出数据)上评估总体准确率
2. 按 SNR 逐档评估（与 v3 的报告口径一致：每档使用该 SNR 的全部数据）
   并汇总: 总体 / 低SNR(< -6dB) / 高SNR(>= 0dB)
3. 绘制 SNR-准确率曲线，可与原 LSTM 基线(84%总体/56.8%低SNR/96.5%高SNR)对比

用法:
    conda activate torch_gpu
    cd /home/mywsl/jwsmc/transformer
    python evaluate.py --ckpt checkpoints/amcnet_transformer_best.pt
"""
import argparse
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from data_loader import load_amc_data_transformer
from model import AMCNetTransformer


@torch.no_grad()
def predict_accuracy(model, x, y, device, batch_size=512):
    """对一整块 numpy 数据逐批预测并返回准确率。"""
    model.eval()
    correct = 0
    for i in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[i:i + batch_size]).float().to(device)
        yb = np.argmax(y[i:i + batch_size], axis=1)         # one-hot -> 类别索引
        logits = model(xb)                                  # (B, 4)
        pred = logits.argmax(dim=1).cpu().numpy()
        correct += (pred == yb).sum()
    return correct / len(x)


def main():
    parser = argparse.ArgumentParser(description="评估 AMCNet-Transformer")
    parser.add_argument("--data-path", default="../data", help="数据集目录")
    parser.add_argument("--ckpt", default="checkpoints/amcnet_transformer_best.pt",
                        help="训练好的权重文件")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="必须与训练时的 num_layers 一致，否则权重加载报错")
    parser.add_argument("--out-dir", default="results", help="结果输出目录")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- 1) 加载模型 ----
    model = AMCNetTransformer(num_layers=args.num_layers).to(device)
    # weights_only=False: 因为保存的 checkpoint 里含有训练超参 dict（非纯张量）
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"加载模型成功: {args.ckpt} (训练时的验证acc={ckpt['val_acc']*100:.2f}%)")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ---- 2) 在独立测试集上评估（严谨的泛化指标）----
    data = load_amc_data_transformer(args.data_path)
    test_acc = predict_accuracy(model, data['x_test'], data['y_test'], device)
    print(f"\n[独立测试集(10%留出)] 总体准确率: {test_acc*100:.2f}%")

    # ---- 3) 按 SNR 逐档评估（与 v3 报告口径一致：用该档全部数据）----
    with open(f"{args.data_path}/K=4_modulation_dataset.pkl", 'rb') as f:
        x_dict = pickle.load(f)
    with open(f"{args.data_path}/K=4_mod_labelset.pkl", 'rb') as f:
        y_dict = pickle.load(f)

    snr_list = sorted(x_dict.keys())
    snr_acc = {}
    for snr in snr_list:
        xs = x_dict[snr] / np.max(np.abs(x_dict[snr]))   # 与训练时的归一化一致
        xs = xs.astype(np.float32)
        acc = predict_accuracy(model, xs, y_dict[snr], device)
        snr_acc[snr] = acc
        print(f"  SNR={snr:3d}dB: {acc*100:6.2f}%")

    # ---- 4) 汇总：总体 / 低SNR / 高SNR ----
    overall = float(np.mean(list(snr_acc.values())))                     # 各档等权平均
    low  = float(np.mean([snr_acc[s] for s in snr_list if s < -6]))      # 低SNR档
    high = float(np.mean([snr_acc[s] for s in snr_list if s >= 0]))      # 高SNR档
    print("\n" + "=" * 60)
    print(f"[与v3同口径] 总体={overall*100:.2f}%  低SNR={low*100:.2f}%  高SNR={high*100:.2f}%")
    print("v3 LSTM基线参考: 总体≈84-85%  低SNR≈56.8-61%  高SNR≈94-96.5%")
    print("=" * 60)

    # 保存逐档结果（方便以后对比/画图）
    with open(os.path.join(args.out_dir, 'snr_acc.pkl'), 'wb') as f:
        pickle.dump({'snr_acc': snr_acc, 'overall': overall,
                     'low': low, 'high': high, 'test_acc': test_acc}, f)

    # ---- 5) 画 SNR-准确率曲线 ----
    snrs = [int(s) for s in snr_list]
    accs = [snr_acc[s] * 100 for s in snr_list]
    plt.figure(figsize=(10, 5))
    plt.plot(snrs, accs, 'o-', label='AMCNet-Transformer')
    plt.axhline(overall * 100, color='gray', ls='--',
                label=f'总体平均 {overall*100:.1f}%')
    plt.axvline(-6, color='r', ls=':', alpha=0.6)      # 低SNR分界线
    plt.text(-6, min(accs), '低SNR', color='r', fontsize=9)
    plt.xlabel('SNR (dB)')
    plt.ylabel('准确率 (%)')
    plt.title('AMCNet-Transformer 各SNR准确率')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.out_dir, 'snr_accuracy.png'), dpi=150)
    print(f"\n图表已保存: {args.out_dir}/snr_accuracy.png")


if __name__ == "__main__":
    main()

