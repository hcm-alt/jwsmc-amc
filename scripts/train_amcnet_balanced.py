"""
AMCNet 低SNR优化版
目标：重点提升低SNR（-18~-6dB）准确率，整体准确率≥80%
核心策略：低SNR加权训练+定向数据增强+适度平衡采样
"""
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import numpy as np
import pickle
import os

# ===================== 核心：低SNR加权的数据加载 =====================
def load_amc_data_low_snr_focus(data_path="data"):
    """
    核心优化：
    1. 保留低SNR样本占比（不欠采样）
    2. 为低/中/高SNR样本分配不同权重（低SNR权重最高）
    3. 仍保留归一化核心逻辑
    """
    print("="*60)
    print("加载AMC数据 (聚焦低SNR+加权训练)...")
    print("="*60)
    
    # 加载数据
    with open(f"{data_path}/K=4_modulation_dataset.pkl", 'rb') as f:
        x_dict = pickle.load(f)
    with open(f"{data_path}/K=4_mod_labelset.pkl", 'rb') as f:
        y_dict = pickle.load(f)
    
    # 归一化（必须保留）
    for snr in x_dict.keys():
        x_dict[snr] = x_dict[snr] / np.max(np.abs(x_dict[snr]))
        x_dict[snr] = x_dict[snr].astype(np.float32)
        y_dict[snr] = y_dict[snr].astype(np.float32)
    
    # 定义SNR分组（聚焦低SNR）
    snrs = sorted(x_dict.keys())
    low_snr = [s for s in snrs if s < -6]       # 重点提升的低SNR区间
    mid_snr = [s for s in snrs if -6 <= s < 6]  # 中SNR
    high_snr = [s for s in snrs if s >= 6]      # 高SNR
    
    print(f"低SNR组（重点优化）: {low_snr}")
    print(f"中SNR组: {mid_snr}")
    print(f"高SNR组: {high_snr}")
    
    # 采样策略：低SNR不欠采样，中/高SNR适度欠采样（保留低SNR占比）
    samples_per_low = len(np.concatenate([x_dict[snr] for snr in low_snr], axis=0))
    samples_per_mid = min(len(np.concatenate([x_dict[snr] for snr in mid_snr], axis=0)), samples_per_low * 1.2)
    samples_per_high = min(len(np.concatenate([x_dict[snr] for snr in high_snr], axis=0)), samples_per_low * 1.2)
    
    # 合并数据并记录SNR标签（用于分配权重）
    x_train_list, y_train_list, snr_label_list = [], [], []
    x_val_list, y_val_list = [], []
    x_test_list, y_test_list = [], []
    
    # 处理低SNR组（核心：不欠采样，保留全部样本）
    x_low = np.concatenate([x_dict[snr] for snr in low_snr], axis=0)
    y_low = np.concatenate([y_dict[snr] for snr in low_snr], axis=0)
    low_indices = np.random.permutation(len(x_low))
    x_low = x_low[low_indices]
    y_low = y_low[low_indices]
    # 拆分低SNR：70%训练，15%验证，15%测试
    n_low_train = int(len(x_low) * 0.7)
    n_low_val = int(len(x_low) * 0.15)
    x_train_list.append(x_low[:n_low_train])
    y_train_list.append(y_low[:n_low_train])
    snr_label_list.extend([0]*n_low_train)  # 低SNR标签：0
    x_val_list.append(x_low[n_low_train:n_low_train+n_low_val])
    y_val_list.append(y_low[n_low_train:n_low_train+n_low_val])
    x_test_list.append(x_low[n_low_train+n_low_val:])
    y_test_list.append(y_low[n_low_train+n_low_val:])
    
    # 处理中SNR组（适度欠采样）
    x_mid = np.concatenate([x_dict[snr] for snr in mid_snr], axis=0)
    y_mid = np.concatenate([y_dict[snr] for snr in mid_snr], axis=0)
    mid_indices = np.random.permutation(len(x_mid))
    x_mid = x_mid[mid_indices[:int(samples_per_mid)]]
    y_mid = y_mid[mid_indices[:int(samples_per_mid)]]
    n_mid_train = int(len(x_mid) * 0.7)
    n_mid_val = int(len(x_mid) * 0.15)
    x_train_list.append(x_mid[:n_mid_train])
    y_train_list.append(y_mid[:n_mid_train])
    snr_label_list.extend([1]*n_mid_train)  # 中SNR标签：1
    x_val_list.append(x_mid[n_mid_train:n_mid_train+n_mid_val])
    y_val_list.append(y_mid[n_mid_train:n_mid_train+n_mid_val])
    x_test_list.append(x_mid[n_mid_train+n_mid_val:])
    y_test_list.append(y_mid[n_mid_train+n_mid_val:])
    
    # 处理高SNR组（适度欠采样）
    x_high = np.concatenate([x_dict[snr] for snr in high_snr], axis=0)
    y_high = np.concatenate([y_dict[snr] for snr in high_snr], axis=0)
    high_indices = np.random.permutation(len(x_high))
    x_high = x_high[high_indices[:int(samples_per_high)]]
    y_high = y_high[high_indices[:int(samples_per_high)]]
    n_high_train = int(len(x_high) * 0.7)
    n_high_val = int(len(x_high) * 0.15)
    x_train_list.append(x_high[:n_high_train])
    y_train_list.append(y_high[:n_high_train])
    snr_label_list.extend([2]*n_high_train)  # 高SNR标签：2
    x_val_list.append(x_high[n_high_train:n_high_train+n_high_val])
    y_val_list.append(y_high[n_high_train:n_high_train+n_high_val])
    x_test_list.append(x_high[n_high_train+n_high_val:])
    y_test_list.append(y_high[n_high_train+n_high_val:])
    
    # 合并数据并打乱
    x_train = np.concatenate(x_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    x_val = np.concatenate(x_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)
    x_test = np.concatenate(x_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)
    
    # 打乱训练集
    np.random.seed(42)
    train_indices = np.random.permutation(len(x_train))
    x_train = x_train[train_indices]
    y_train = y_train[train_indices]
    snr_label_list = np.array(snr_label_list)[train_indices]
    
    # 分配样本权重：低SNR=3.0，中SNR=1.0，高SNR=0.8（重点关注低SNR）
    sample_weights = np.where(snr_label_list == 0, 3.0, 
                              np.where(snr_label_list == 1, 1.0, 0.8))
    
    print(f"\n数据分布（聚焦低SNR）:")
    print(f"训练集: x_train={x_train.shape}, y_train={y_train.shape} (低SNR占比≈{len([l for l in snr_label_list if l==0])/len(snr_label_list)*100:.1f}%)")
    print(f"验证集: x_val={x_val.shape}, y_val={y_val.shape}")
    print(f"测试集: x_test={x_test.shape}, y_test={y_test.shape}")
    
    return x_train, x_val, x_test, y_train, y_val, y_test, sample_weights

# ===================== 定向数据增强：仅增强低SNR样本 =====================
def augment_low_snr_data(x, y, sample_weights):
    """
    核心优化：
    1. 仅对低SNR样本（权重=3.0）添加噪声增强
    2. 中/高SNR样本不增强，避免过拟合
    """
    print("执行定向数据增强（仅低SNR样本添加噪声）...")
    
    x_augmented = [x]
    y_augmented = [y]
    weights_augmented = [sample_weights]
    
    # 仅对低SNR样本做增强（添加更强的噪声，模拟极端低SNR环境）
    low_snr_mask = sample_weights == 3.0
    x_low = x[low_snr_mask]
    y_low = y[low_snr_mask]
    weights_low = sample_weights[low_snr_mask]
    
    if len(x_low) > 0:
        # 为低SNR样本添加随机噪声（噪声因子=0.06，适配低SNR强噪声）
        noise = np.random.normal(0, 0.06, x_low.shape)
        x_low_aug = x_low + noise
        # 幅度缩放（模拟信号强度波动）
        scale = np.random.uniform(0.7, 1.3, (x_low_aug.shape[0], 1, 1))
        x_low_aug = x_low_aug * scale
        
        x_augmented.append(x_low_aug)
        y_augmented.append(y_low)
        weights_augmented.append(weights_low)
    
    # 合并增强后的数据
    x_final = np.concatenate(x_augmented, axis=0)
    y_final = np.concatenate(y_augmented, axis=0)
    weights_final = np.concatenate(weights_augmented, axis=0)
    
    print(f"增强后: x={x_final.shape}, y={y_final.shape} (低SNR样本翻倍)")
    return x_final, y_final, weights_final

# ===================== 模型优化：增强低SNR特征提取 =====================
def build_amcnet_low_snr(input_shape=(256, 2)):
    """
    核心优化：
    1. 增加卷积层深度，提升低SNR弱特征提取能力
    2. 添加多尺度卷积，适配不同SNR的特征尺度
    3. 降低正则化，避免低SNR特征丢失
    """
    inputs = layers.Input(shape=input_shape)
    
    # 多尺度卷积（核心：提取不同尺度的低SNR特征）
    x1 = layers.Conv1D(128, 3, padding='same', kernel_regularizer=regularizers.l2(1e-6))(inputs)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)
    
    x3 = layers.Conv1D(128, 5, padding='same', kernel_regularizer=regularizers.l2(1e-6))(inputs)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Activation('relu')(x3)
    
    x = layers.Concatenate()([x1, x3])  # 多尺度特征融合
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.05)(x)
    
    # 增强特征提取层（适配低SNR弱特征）
    x = layers.Conv1D(256, 3, padding='same', kernel_regularizer=regularizers.l2(1e-6))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.05)(x)
    
    x = layers.Conv1D(256, 3, padding='same', kernel_regularizer=regularizers.l2(1e-6))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.05)(x)
    
    # 注意力机制（聚焦关键低SNR特征）
    attn = layers.Attention()([x, x])
    x = layers.Add()([x, attn])
    x = layers.GlobalAveragePooling1D()(x)
    
    # 分类头（适度增强，避免过拟合）
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(4, activation='softmax')(x)
    
    model = Model(inputs, outputs, name="AMCNet_Low_SNR")
    return model

# ===================== 训练逻辑：加权训练+适配低SNR =====================
def train_amcnet_low_snr():
    print("="*60)
    print("开始训练AMCNet (聚焦低SNR优化版)")
    print("="*60)
    
    # 加载数据（含样本权重）
    x_train, x_val, x_test, y_train, y_val, y_test, sample_weights = load_amc_data_low_snr_focus()
    
    # 定向增强低SNR样本
    x_train_aug, y_train_aug, weights_aug = augment_low_snr_data(x_train, y_train, sample_weights)
    
    # 创建优化模型
    model = build_amcnet_low_snr()
    
    # 编译：适配低SNR的学习率（稍低，避免震荡）
    optimizer = tf.keras.optimizers.Adam(learning_rate=6e-5)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\n模型参数数量: {model.count_params():,}")
    model.summary()
    
    # 创建目录
    os.makedirs("model", exist_ok=True)
    
    # 回调函数：监控低SNR为主的验证集
    callbacks = [
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            "model/AMCNet_low_snr_best.h5",
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 训练：传入样本权重，强制模型关注低SNR
    print("\n开始训练（低SNR加权）...")
    history = model.fit(
        x_train_aug, y_train_aug,
        sample_weight=weights_aug,  # 核心：低SNR样本权重=3.0
        validation_data=(x_val, y_val),
        epochs=60,
        batch_size=96,  # 兼顾速度和稳定性
        callbacks=callbacks,
        verbose=1
    )
    
    # 测试（重点输出低SNR准确率）
    print("\n" + "="*60)
    print("AMCNet 低SNR优化版 测试集评估...")
    print("="*60)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print(f"AMCNet测试集损失: {test_loss:.4f}")
    print(f"AMCNet测试集准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # 按SNR分析（重点标注低SNR）
    print("\n按SNR分析准确率（重点关注低SNR）:")
    with open("data/K=4_modulation_dataset.pkl", 'rb') as f:
        x_dict = pickle.load(f)
    with open("data/K=4_mod_labelset.pkl", 'rb') as f:
        y_dict = pickle.load(f)
    
    for snr in sorted(x_dict.keys()):
        x_dict[snr] = x_dict[snr] / np.max(np.abs(x_dict[snr]))
        _, acc_snr = model.evaluate(x_dict[snr], y_dict[snr], verbose=0)
        # 重点标注低SNR
        if snr < -6:
            print(f"  SNR={snr:3d}dB: {acc_snr:.4f} ({acc_snr*100:.2f}%) → 低SNR（优化后）")
        else:
            print(f"  SNR={snr:3d}dB: {acc_snr:.4f} ({acc_snr*100:.2f}%)")
    
    return model, history

if __name__ == "__main__":
    try:
        model, history = train_amcnet_low_snr()
        print("\n" + "="*60)
        print("训练完成!")
        print("="*60)
    except Exception as e:
        print(f"报错类型: {type(e).__name__}")
        print(f"报错信息: {e}")
        import traceback
        traceback.print_exc()
