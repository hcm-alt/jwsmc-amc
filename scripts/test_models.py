"""
模型测试脚本
加载训练好的WSSNet和AMCNet模型，在测试集上评估性能
"""
import numpy as np
import pickle
import os
import sys

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf

def load_wss_data(data_path="../data"):
    """加载频谱感知(WSS)任务的数据集"""
    print("=" * 60)
    print("加载WSSNet测试数据...")
    print("=" * 60)
    
    # 加载输入数据（30×128×2）
    with open(f"{data_path}/K=6_spectrum_dataset.pkl", 'rb') as f:
        x_dict = pickle.load(f)
    # 加载标签（30维0/1）
    with open(f"{data_path}/K=6_freq_labelset.pkl", 'rb') as f:
        y_dict = pickle.load(f)
    
    # 合并所有SNR的数据
    x_all = np.concatenate([x_dict[snr] for snr in x_dict.keys()], axis=0)
    y_all = np.concatenate([y_dict[snr] for snr in y_dict.keys()], axis=0)
    
    # 按论文比例拆分：训练集22800，验证集7600，测试集7600
    x_train, x_temp, y_train, y_temp = train_test_split(
        x_all, y_all, test_size=15200/22800, random_state=42
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"WSS数据加载完成:")
    print(f"  训练集: x_train={x_train.shape}, y_train={y_train.shape}")
    print(f"  验证集: x_val={x_val.shape}, y_val={y_val.shape}")
    print(f"  测试集: x_test={x_test.shape}, y_test={y_test.shape}")
    return x_train, x_val, x_test, y_train, y_val, y_test


def load_amc_data(data_path="../data"):
    """加载调制识别(AMC)任务的数据集"""
    print("\n" + "=" * 60)
    print("加载AMCNet测试数据...")
    print("=" * 60)
    
    # 加载输入数据（256×2）
    with open(f"{data_path}/K=4_modulation_dataset.pkl", 'rb') as f:
        x_dict = pickle.load(f)
    # 加载标签（4维独热码）
    with open(f"{data_path}/K=4_mod_labelset.pkl", 'rb') as f:
        y_dict = pickle.load(f)
    
    # 合并所有SNR的数据
    x_all = np.concatenate([x_dict[snr] for snr in x_dict.keys()], axis=0)
    y_all = np.concatenate([y_dict[snr] for snr in y_dict.keys()], axis=0)
    
    # 按论文比例拆分
    x_train, x_temp, y_train, y_temp = train_test_split(
        x_all, y_all, test_size=15200/114000, random_state=42
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"AMC数据加载完成:")
    print(f"  训练集: x_train={x_train.shape}, y_train={y_train.shape}")
    print(f"  验证集: x_val={x_val.shape}, y_val={y_val.shape}")
    print(f"  测试集: x_test={x_test.shape}, y_test={y_test.shape}")
    return x_train, x_val, x_test, y_train, y_val, y_test


def test_wssnet(model_path, x_test, y_test):
    """测试WSSNet模型"""
    print("\n" + "=" * 60)
    print("测试WSSNet模型...")
    print("=" * 60)
    
    # 加载模型
    print(f"加载模型: {model_path}")
    model = tf.keras.models.load_model(model_path)
    model.summary()
    
    # 评估模型
    print("\n在测试集上评估模型...")
    results = model.evaluate(x_test, y_test, verbose=1)
    
    # 获取预测结果
    y_pred = model.predict(x_test, verbose=1)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # 计算各项指标
    # 逐样本计算准确率（至少有一个频段预测正确就算对）
    sample_accuracy = np.mean(np.any(y_pred_binary == y_test, axis=1))
    # 完全匹配准确率
    full_accuracy = np.mean(np.all(y_pred_binary == y_test, axis=1))
    # 频段级准确率
    band_accuracy = np.mean(y_pred_binary == y_test)
    
    print("\n" + "=" * 40)
    print("WSSNet 测试结果:")
    print("=" * 40)
    print(f"  整体Loss: {results[0]:.4f}")
    print(f"  整体Binary Accuracy: {results[1]:.4f}")
    print(f"  样本级准确率(至少对一个频段): {sample_accuracy:.4f}")
    print(f"  完全匹配准确率: {full_accuracy:.4f}")
    print(f"  频段级准确率: {band_accuracy:.4f}")
    
    return results, y_pred


def test_amcnet(model_path, x_test, y_test):
    """测试AMCNet模型"""
    print("\n" + "=" * 60)
    print("测试AMCNet模型...")
    print("=" * 60)
    
    # 加载模型
    print(f"加载模型: {model_path}")
    model = tf.keras.models.load_model(model_path)
    model.summary()
    
    # 评估模型
    print("\n在测试集上评估模型...")
    results = model.evaluate(x_test, y_test, verbose=1)
    
    # 获取预测结果
    y_pred = model.predict(x_test, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # 计算准确率
    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    
    # 混淆矩阵
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # 分类报告
    mod_types = ['BPSK', 'QPSK', 'QAM16', 'QAM64']
    report = classification_report(y_true_classes, y_pred_classes, 
                                   target_names=mod_types, digits=4)
    
    print("\n" + "=" * 40)
    print("AMCNet 测试结果:")
    print("=" * 40)
    print(f"  Loss: {results[0]:.4f}")
    print(f"  Accuracy: {results[1]:.4f}")
    print(f"\n混淆矩阵:")
    print(cm)
    print(f"\n分类报告:")
    print(report)
    
    return results, y_pred


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("开始模型测试")
    print("=" * 60)
    
    # 数据路径
    data_path = "./data"
    model_path = "./model"
    
    # ==================== WSSNet 测试 ====================
    # 加载WSS数据
    x_train_wss, x_val_wss, x_test_wss, y_train_wss, y_val_wss, y_test_wss = load_wss_data(data_path)
    
    # 测试WSSNet
    wss_results, wss_pred = test_wssnet(
        f"{model_path}/WSSNet_best.h5",
        x_test_wss,
        y_test_wss
    )
    
    # ==================== AMCNet 测试 ====================
    # 加载AMC数据
    x_train_amc, x_val_amc, x_test_amc, y_train_amc, y_val_amc, y_test_amc = load_amc_data(data_path)
    
    # 测试AMCNet
    amc_results, amc_pred = test_amcnet(
        f"{model_path}/AMCNet_best.h5",
        x_test_amc,
        y_test_amc
    )
    
    # ==================== 总结 ====================
    print("\n" + "=" * 60)
    print("测试完成！模型性能总结:")
    print("=" * 60)
    print(f"WSSNet - Loss: {wss_results[0]:.4f}, Binary Accuracy: {wss_results[1]:.4f}")
    print(f"AMCNet - Loss: {amc_results[0]:.4f}, Accuracy: {amc_results[1]:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
