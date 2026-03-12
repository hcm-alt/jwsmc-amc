"""
全面测试所有AMCNet模型并按SNR分析
"""
import numpy as np
import pickle
import tensorflow as tf
import os

def load_amc_data(data_path="/home/mywsl/jwsmc/data"):
    """加载AMC数据"""
    with open(f"{data_path}/K=4_modulation_dataset.pkl", 'rb') as f:
        x_dict = pickle.load(f)
    with open(f"{data_path}/K=4_mod_labelset.pkl", 'rb') as f:
        y_dict = pickle.load(f)
    return x_dict, y_dict


def normalize_data(x_dict):
    """标准化数据（与训练时一致）"""
    for snr in x_dict.keys():
        x_dict[snr] = x_dict[snr] / np.max(np.abs(x_dict[snr]))
        x_dict[snr] = x_dict[snr].astype(np.float32)
    return x_dict


def test_model_by_snr(model_path, x_dict, y_dict, model_name):
    """按SNR测试模型"""
    print(f"\n{'='*60}")
    print(f"测试模型: {model_name}")
    print(f"{'='*60}")
    
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # 整体测试
    x_all = np.concatenate([x_dict[snr] for snr in sorted(x_dict.keys())], axis=0)
    y_all = np.concatenate([y_dict[snr] for snr in sorted(y_dict.keys())], axis=0)
    
    results = model.evaluate(x_all, y_all, verbose=0)
    print(f"总体准确率: {results[1]:.4f} ({results[1]*100:.2f}%)")
    
    # 按SNR分析
    print("\n按SNR分析:")
    low_snr_acc = []
    high_snr_acc = []
    
    for snr in sorted(x_dict.keys()):
        x_snr = x_dict[snr]
        y_snr = y_dict[snr]
        _, acc = model.evaluate(x_snr, y_snr, verbose=0)
        
        if snr < -6:
            low_snr_acc.append(acc)
            print(f"  SNR={snr:3d}dB: {acc:.4f} ({acc*100:.2f}%) → 低SNR")
        else:
            high_snr_acc.append(acc)
            print(f"  SNR={snr:3d}dB: {acc:.4f} ({acc*100:.2f}%)")
    
    low_avg = np.mean(low_snr_acc) if low_snr_acc else 0
    high_avg = np.mean(high_snr_acc) if high_snr_acc else 0
    
    print(f"\n低SNR平均: {low_avg:.4f} ({low_avg*100:.2f}%)")
    print(f"高SNR平均: {high_avg:.4f} ({high_avg*100:.2f}%)")
    
    return results[1], low_avg, high_avg


def main():
    print("="*60)
    print("全面测试所有AMCNet模型")
    print("="*60)
    
    # 加载数据
    x_dict, y_dict = load_amc_data()
    x_dict = normalize_data(x_dict)
    
    print(f"\n数据SNR范围: {sorted(x_dict.keys())}")
    print(f"每个SNR样本数: {len(x_dict[sorted(x_dict.keys())[0]])}")
    
    # 模型列表
    models = [
        ("/home/mywsl/jwsmc/model/AMCNet_best.h5", "AMCNet_best"),
        ("/home/mywsl/jwsmc/model/AMCNet_balanced_best.h5", "AMCNet_balanced_best"),
        ("/home/mywsl/jwsmc/model/AMCNet_low_snr_best.h5", "AMCNet_low_snr_best"),
    ]
    
    results = []
    for model_path, model_name in models:
        if os.path.exists(model_path):
            overall, low, high = test_model_by_snr(model_path, x_dict, y_dict, model_name)
            results.append((model_name, overall, low, high))
        else:
            print(f"\n模型不存在: {model_path}")
    
    # 总结
    print("\n" + "="*60)
    print("模型性能总结")
    print("="*60)
    print(f"{'模型名称':<30} {'总体':>10} {'低SNR':>10} {'高SNR':>10}")
    print("-"*60)
    for name, overall, low, high in results:
        print(f"{name:<30} {overall*100:>9.2f}% {low*100:>9.2f}% {high*100:>9.2f}%")
    
    # 找出最佳模型
    best = max(results, key=lambda x: x[1])
    print(f"\n最佳总体准确率: {best[0]}")
    
    # 目标检查
    print("\n" + "="*60)
    print("目标检查")
    print("="*60)
    print(f"目标: 总体>80%, 低SNR>50%, 高SNR>95%")
    for name, overall, low, high in results:
        status = "✓" if overall >= 0.80 and low >= 0.50 and high >= 0.95 else "✗"
        print(f"{status} {name}: 总体{overall*100:.1f}%, 低{low*100:.1f}%, 高{high*100:.1f}%")


if __name__ == "__main__":
    main()
