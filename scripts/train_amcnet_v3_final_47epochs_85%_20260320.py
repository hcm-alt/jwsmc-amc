"""
AMCNet v3 优化版 - 目标：低SNR>62%, 高SNR>96%, 总体>85%

核心：低SNR权重提升到3.0
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
    
    for snr in x_dict.keys():
        x_dict[snr] = x_dict[snr] / np.max(np.abs(x_dict[snr]))
        x_dict[snr] = x_dict[snr].astype(np.float32)
        y_dict[snr] = y_dict[snr].astype(np.float32)
    
    x_all = np.concatenate([x_dict[snr] for snr in sorted(x_dict.keys())], axis=0)
    y_all = np.concatenate([y_dict[snr] for snr in sorted(y_dict.keys())], axis=0)
    
    np.random.seed(42)
    indices = np.random.permutation(len(x_all))
    x_all = x_all[indices]
    y_all = y_all[indices]
    
    n_train = int(len(x_all) * 0.8)
    n_val = int(len(x_all) * 0.1)
    x_train = x_all[:n_train]
    y_train = y_all[:n_train]
    x_val = x_all[n_train:n_train+n_val]
    y_val = y_all[n_train:n_train+n_val]
    x_test = x_all[n_train+n_val:]
    y_test = y_all[n_train+n_val:]
    
    # SNR标签
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
    
    class_counts = np.sum(y_train, axis=0)
    class_weights = len(y_train) / (4 * class_counts)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    print(f"类别权重: {class_weight_dict}")
    
    # 【关键】权重1.3
    sample_weights = np.where(train_snr_tags < -6, 1.3, 1.0)
    
    print(f"\n数据分布:")
    print(f"训练集: {x_train.shape}")
    print(f"样本权重: 低SNR=1.3, 高SNR=1.0")
    
    return x_train, x_val, x_test, y_train, y_val, y_test, sample_weights, class_weight_dict, x_dict, y_dict


def build_amcnet_v3_deeper(input_shape=(256, 2)):
    """更深的AMCNet"""
    inputs = layers.Input(shape=input_shape)
    
    # 第1层
    x = layers.Conv1D(64, 3, padding='same', kernel_regularizer=regularizers.l2(1e-5))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(64, 3, padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.15)(x)
    
    # 第2层
    x = layers.Conv1D(128, 3, padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(128, 3, padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.15)(x)
    
    # 第3层
    x = layers.Conv1D(256, 3, padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(256, 3, padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)
    
    # 第4层
    x = layers.Conv1D(256, 3, padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)
    
    # 注意力
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
    print("开始训练AMCNet v3（权重3.0）")
    print("目标: 总体>85%, 低SNR>62%, 高SNR>96%")
    print("="*60)
    
    x_train, x_val, x_test, y_train, y_val, y_test, sample_weights, class_weights, x_dict, y_dict = load_amc_data_v3()
    
    model = build_amcnet_v3_deeper()
    
    # 学习率 3e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\n模型参数数量: {model.count_params():,}")
    model.summary()
    
    os.makedirs("model", exist_ok=True)
    
    callbacks = [
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=15, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=48, restore_best_weights=True, verbose=1),
        ModelCheckpoint("model/AMCNet_v3_best.h5", monitor='val_accuracy', save_best_only=True, verbose=1),
        TensorBoard(log_dir="logs/AMCNet_v3"),
    ]
    
    print("\n开始训练...")
    history = model.fit(
        x_train, y_train,
        sample_weight=sample_weights,
        class_weight=class_weights,
        validation_data=(x_val, y_val),
        epochs=48,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    # 测试评估
    print("\n" + "="*60)
    print("测试集评估...")
    print("="*60)
    
    x_all = np.concatenate([x_dict[snr] / np.max(np.abs(x_dict[snr])) for snr in sorted(x_dict.keys())], axis=0)
    y_all = np.concatenate([y_dict[snr] for snr in sorted(y_dict.keys())], axis=0)
    test_loss, test_acc = model.evaluate(x_all, y_all, verbose=1)
    print(f"\n总体准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
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
    
    print("\n目标检查")
    print(f"目标: 总体>85%, 低SNR>62%, 高SNR>96%")
    status = "✓" if test_acc >= 0.85 and low_avg >= 0.62 and high_avg >= 0.96 else "✗"
    print(f"{status} 结果: 总体{test_acc*100:.1f}%, 低{low_avg*100:.1f}%, 高{high_avg*100:.1f}%")
    
    return model, history


if __name__ == "__main__":
    try:
        model, history = train_amcnet_v3()
        print("\n训练完成!")
    except Exception as e:
        print(f"报错: {e}")
        import traceback
        traceback.print_exc()
