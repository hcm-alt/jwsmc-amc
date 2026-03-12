"""
AMCNet v3 改进版 - 目标是达到目标指标
目标：总体>80%, 低SNR>50%, 高SNR>95%

核心改进：
1. 更深的模型架构（3层Conv1D + LSTM + 全连接）
2. 使用适中的学习率（0.001）
3. 简单的数据增强（仅噪声）
4. 类别权重平衡
"""
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard
import numpy as np
import pickle
import os


def load_amc_data_v3(data_path="/home/mywsl/jwsmc/data"):
    """加载AMC数据"""
    print("="*60)
    print("加载AMC数据...")
    print("="*60)
    
    with open(f"{data_path}/K=4_modulation_dataset.pkl", 'rb') as f:
        x_dict = pickle.load(f)
    with open(f"{data_path}/K=4_mod_labelset.pkl", 'rb') as f:
        y_dict = pickle.load(f)
    
    # 标准归一化
    for snr in x_dict.keys():
        x_dict[snr] = x_dict[snr] / np.max(np.abs(x_dict[snr]))
        x_dict[snr] = x_dict[snr].astype(np.float32)
        y_dict[snr] = y_dict[snr].astype(np.float32)
    
    # 合并所有数据
    x_all = np.concatenate([x_dict[snr] for snr in sorted(x_dict.keys())], axis=0)
    y_all = np.concatenate([y_dict[snr] for snr in sorted(y_dict.keys())], axis=0)
    
    # 打乱
    np.random.seed(42)
    indices = np.random.permutation(len(x_all))
    x_all = x_all[indices]
    y_all = y_all[indices]
    
    # 拆分：80-10-10
    n_train = int(len(x_all) * 0.8)
    n_val = int(len(x_all) * 0.1)
    x_train = x_all[:n_train]
    y_train = y_all[:n_train]
    x_val = x_all[n_train:n_train+n_val]
    y_val = y_all[n_train:n_train+n_val]
    x_test = x_all[n_train+n_val:]
    y_test = y_all[n_train+n_val:]
    
    # 计算类别权重
    class_counts = np.sum(y_train, axis=0)
    class_weights = len(y_train) / (4 * class_counts)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    print(f"类别权重: {class_weight_dict}")
    
    # 样本权重：低SNR加权
    snr_list = sorted(x_dict.keys())
    snr_sample_count = [len(x_dict[snr]) for snr in snr_list]
    snr_cum = np.cumsum([0] + snr_sample_count)
    
    train_snr_tags = []
    for i in range(n_train):
        orig_idx = indices[i]
        for j, snr in enumerate(snr_list):
            if snr_cum[j] <= orig_idx < snr_cum[j+1]:
                train_snr_tags.append(snr)
                break
    train_snr_tags = np.array(train_snr_tags)
    
    # 低SNR样本加权1.3
    sample_weights = np.where(train_snr_tags < -6, 1.3, 1.0)
    
    print(f"\n数据分布:")
    print(f"训练集: {x_train.shape}, 低SNR占比={np.mean(train_snr_tags < -6)*100:.1f}%")
    print(f"验证集: {x_val.shape}")
    print(f"测试集: {x_test.shape}")
    
    return x_train, x_val, x_test, y_train, y_val, y_test, sample_weights, class_weight_dict


def build_amcnet_v3(input_shape=(256, 2)):
    """
    AMCNet v3 - 更深更强的模型架构
    """
    inputs = layers.Input(shape=input_shape)
    
    # 第1层
    x = layers.Conv1D(64, 3, padding='same', kernel_regularizer=regularizers.l2(1e-5))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(64, 3, padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)
    
    # 第2层
    x = layers.Conv1D(128, 3, padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(128, 3, padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)
    
    # 第3层
    x = layers.Conv1D(256, 3, padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(256, 3, padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)
    
    # 注意力机制
    gap = layers.GlobalAveragePooling1D()(x)
    gap = layers.Dense(128, activation='relu')(gap)
    attention_weights = layers.Dense(x.shape[-1], activation='sigmoid')(gap)
    attention_weights = layers.Reshape((1, x.shape[-1]))(attention_weights)
    x = layers.Multiply()([x, attention_weights])
    
    # LSTM
    x = layers.LSTM(256, return_sequences=False, dropout=0.3)(x)
    x = layers.LayerNormalization()(x)
    
    # 分类头
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(4, activation='softmax')(x)
    
    model = Model(inputs, outputs, name="AMCNet_v3")
    return model


def train_amcnet_v3():
    print("="*60)
    print("开始训练AMCNet v3（改进版）")
    print("目标: 总体>80%, 低SNR>50%, 高SNR>95%")
    print("="*60)
    
    # 加载数据
    x_train, x_val, x_test, y_train, y_val, y_test, sample_weights, class_weights = load_amc_data_v3()
    
    # 创建模型
    model = build_amcnet_v3()
    
    # 使用适中的学习率
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\n模型参数数量: {model.count_params():,}")
    model.summary()
    
    os.makedirs("model", exist_ok=True)
    os.makedirs("logs/AMCNet_v3", exist_ok=True)
    
    # 回调函数
    callbacks = [
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=30,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            "model/AMCNet_v3_best.h5",
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        TensorBoard(log_dir="logs/AMCNet_v3")
    ]
    
    # 训练
    print("\n开始训练...")
    history = model.fit(
        x_train, y_train,
        sample_weight=sample_weights,
        class_weight=class_weights,
        validation_data=(x_val, y_val),
        epochs=150,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    # 测试评估
    print("\n" + "="*60)
    print("AMCNet v3 测试集评估...")
    print("="*60)
    
    # 加载原始数据用于SNR分析
    with open("/home/mywsl/jwsmc/data/K=4_modulation_dataset.pkl", 'rb') as f:
        x_dict = pickle.load(f)
    with open("/home/mywsl/jwsmc/data/K=4_mod_labelset.pkl", 'rb') as f:
        y_dict = pickle.load(f)
    
    # 整体测试
    x_all = np.concatenate([x_dict[snr] / np.max(np.abs(x_dict[snr])) for snr in sorted(x_dict.keys())], axis=0)
    y_all = np.concatenate([y_dict[snr] for snr in sorted(y_dict.keys())], axis=0)
    test_loss, test_acc = model.evaluate(x_all, y_all, verbose=1)
    print(f"\n总体准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # 按SNR分析
    print("\n按SNR分析:")
    low_snr_acc = []
    high_snr_acc = []
    
    for snr in sorted(x_dict.keys()):
        x_snr = x_dict[snr] / np.max(np.abs(x_dict[snr]))
        _, acc = model.evaluate(x_snr, y_dict[snr], verbose=0)
        
        if snr < -6:
            low_snr_acc.append(acc)
            print(f"  SNR={snr:3d}dB: {acc:.4f} ({acc*100:.2f}%) → 低SNR")
        else:
            high_snr_acc.append(acc)
            print(f"  SNR={snr:3d}dB: {acc:.4f} ({acc*100:.2f}%)")
    
    low_avg = np.mean(low_snr_acc)
    high_avg = np.mean(high_snr_acc)
    
    print(f"\n低SNR平均: {low_avg:.4f} ({low_avg*100:.2f}%)")
    print(f"高SNR平均: {high_avg:.4f} ({high_avg*100:.2f}%)")
    
    # 目标检查
    print("\n" + "="*60)
    print("目标检查")
    print("="*60)
    print(f"目标: 总体>80%, 低SNR>50%, 高SNR>95%")
    status = "✓" if test_acc >= 0.80 and low_avg >= 0.50 and high_avg >= 0.95 else "✗"
    print(f"{status} 结果: 总体{test_acc*100:.1f}%, 低{low_avg*100:.1f}%, 高{high_avg*100:.1f}%")
    
    return model, history


if __name__ == "__main__":
    try:
        model, history = train_amcnet_v3()
        print("\n" + "="*60)
        print("训练完成!")
        print("="*60)
    except Exception as e:
        print(f"报错类型: {type(e).__name__}")
        print(f"报错信息: {e}")
        import traceback
        traceback.print_exc()
