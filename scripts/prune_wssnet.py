"""
WSSNet轻量化 - 剪枝(Pruning)
只做剪枝，不微调（内存不足）
"""
import tensorflow as tf
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import os
import gc

def load_wss_data(data_path="/home/mywsl/jwsmc/data"):
    with open(f"{data_path}/K=6_spectrum_dataset.pkl", 'rb') as f:
        x_dict = pickle.load(f)
    with open(f"{data_path}/K=6_freq_labelset.pkl", 'rb') as f:
        y_dict = pickle.load(f)
    
    x_all = np.concatenate([x_dict[snr] for snr in x_dict.keys()], axis=0)
    y_all = np.concatenate([y_dict[snr] for snr in y_dict.keys()], axis=0)
    x_all = x_all.astype(np.float32)
    
    x_train, x_temp, y_train, y_temp = train_test_split(
        x_all, y_all, test_size=15200/22800, random_state=42
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=42
    )
    
    del x_dict, y_dict, x_all, y_all, x_temp, y_temp
    gc.collect()
    
    return x_train, x_val, x_test, y_train, y_val, y_test


def count_params(model):
    trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    return trainable, non_trainable, trainable + non_trainable


def manual_prune(model, sparsity=0.3):
    """简单剪枝：只剪枝一次，不保存mask"""
    pruned_count = 0
    total_count = 0
    
    for layer in model.layers:
        if hasattr(layer, 'get_weights') and layer.get_weights():
            weights = layer.get_weights()
            new_weights = []
            
            for w in weights:
                total_count += w.size
                if len(w.shape) > 1:
                    threshold = np.percentile(np.abs(w), sparsity * 100)
                    mask = np.abs(w) > threshold
                    pruned_count += np.sum(~mask)
                    new_weights.append((w * mask).astype(np.float32))
                else:
                    new_weights.append(w)
            
            layer.set_weights(new_weights)
    
    print(f"剪枝统计: {pruned_count}/{total_count} = {pruned_count/total_count*100:.1f}%")
    return model


def evaluate_model(model, x_test, y_test):
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    return loss, acc


def prune_wssnet():
    print("="*60)
    print("WSSNet 剪枝轻量化")
    print("="*60)
    
    x_train, x_val, x_test, y_train, y_val, y_test = load_wss_data()
    print(f"\n数据: 训练{x_train.shape}, 测试{x_test.shape}")
    
    print("\n加载原始模型...")
    original_model = tf.keras.models.load_model("/home/mywsl/jwsmc/model/WSSNet_best.h5")
    
    orig_trainable, orig_non, orig_total = count_params(original_model)
    print(f"原始模型参数量: {orig_total:,}")
    
    _, orig_acc = evaluate_model(original_model, x_test, y_test)
    print(f"原始模型测试准确率: {orig_acc*100:.2f}%")
    
    del original_model
    gc.collect()
    
    print("\n开始剪枝 (30% sparsity)...")
    pruned_model = tf.keras.models.load_model("/home/mywsl/jwsmc/model/WSSNet_best.h5")
    pruned_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
    
    pruned_model = manual_prune(pruned_model, sparsity=0.3)
    
    # 统计非零参数
    non_zero = 0
    for layer in pruned_model.layers:
        if hasattr(layer, 'get_weights') and layer.get_weights():
            for w in layer.get_weights():
                non_zero += np.count_nonzero(w)
    
    print(f"\n剪枝后非零参数量: {non_zero:,}")
    print(f"实际压缩率: {(1 - non_zero/orig_total)*100:.1f}%")
    
    _, prune_acc = evaluate_model(pruned_model, x_test, y_test)
    print(f"剪枝后测试准确率: {prune_acc*100:.2f}%")
    
    acc_diff = (orig_acc - prune_acc) * 100
    print(f"准确率变化: {acc_diff:+.2f}%")
    
    os.makedirs("/home/mywsl/jwsmc/model", exist_ok=True)
    save_path = "/home/mywsl/jwsmc/model/WSSNet_pruned.h5"
    pruned_model.save(save_path)
    print(f"\n模型已保存到: {save_path}")
    
    print("\n" + "="*60)
    print("参数量对比")
    print("="*60)
    print(f"{'指标':<25} {'原始模型':<15} {'剪枝后':<15} {'变化':<15}")
    print("-"*60)
    print(f"{'总参数量':<25} {orig_total:<15,} {orig_total:<15,} {'-':<15}")
    print(f"{'非零参数量':<25} {orig_total:<15,} {non_zero:<15,} {f'{(1-non_zero/orig_total)*100:.1f}%↓':<15}")
    print(f"{'测试准确率':<25} {orig_acc*100:<14.2f}% {prune_acc*100:<14.2f}% {acc_diff:+.2f}%")
    print("="*60)
    
    return pruned_model


if __name__ == "__main__":
    try:
        model = prune_wssnet()
        print("\n剪枝完成!")
    except Exception as e:
        print(f"报错: {e}")
        import traceback
        traceback.print_exc()
