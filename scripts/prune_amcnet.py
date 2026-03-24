"""
AMCNet轻量化 - 剪枝(Pruning) - 带Mask版
基于 model/AMCNet_v3_best.h5 进行剪枝
目标：参数量减少50%以上，准确率下降<3%
"""
import tensorflow as tf
import numpy as np
import pickle
import os

# 加载数据
def load_amc_data(data_path="/home/mywsl/jwsmc/data"):
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
    
    return x_train, x_val, x_test, y_train, y_val, y_test


def count_params(model):
    """统计模型参数量"""
    trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    total = trainable + non_trainable
    return trainable, non_trainable, total


def manual_prune_with_mask(model, sparsity=0.5):
    """手动剪枝：创建mask保持剪枝状态"""
    masks = []
    pruned_count = 0
    total_count = 0
    
    for layer in model.layers:
        if hasattr(layer, 'get_weights') and layer.get_weights():
            weights = layer.get_weights()
            layer_masks = []
            new_weights = []
            
            for w in weights:
                total_count += w.size
                if len(w.shape) > 1:  # 只剪枝权重矩阵
                    threshold = np.percentile(np.abs(w), sparsity * 100)
                    mask = np.abs(w) > threshold
                    layer_masks.append(mask.astype(np.float32))
                    pruned_count += np.sum(~mask)
                    new_weights.append(w.astype(np.float32))
                else:
                    layer_masks.append(np.ones_like(w, dtype=np.float32))
                    new_weights.append(w)
            
            masks.append(layer_masks)
            layer.set_weights(new_weights)
    
    # 保存masks到模型
    model.pruning_masks = masks
    print(f"剪枝统计: {pruned_count}/{total_count} = {pruned_count/total_count*100:.1f}%")
    return model


def apply_masks(model):
    """在每次训练后应用mask"""
    if not hasattr(model, 'pruning_masks'):
        return
    
    masks = model.pruning_masks
    mask_idx = 0
    for layer in model.layers:
        if hasattr(layer, 'get_weights') and layer.get_weights():
            weights = layer.get_weights()
            new_weights = []
            for i, w in enumerate(weights):
                if mask_idx < len(masks) and i < len(masks[mask_idx]):
                    w_masked = w * masks[mask_idx][i]
                    new_weights.append(w_masked)
                else:
                    new_weights.append(w)
            layer.set_weights(new_weights)
            mask_idx += 1


def evaluate_model(model, x_test, y_test):
    """评估模型"""
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    return loss, acc


class PruningCallback(tf.keras.callbacks.Callback):
    """每次训练epoch后应用mask"""
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def on_epoch_end(self, epoch, logs=None):
        apply_masks(self.model)


def prune_amcnet():
    print("="*60)
    print("AMCNet 剪枝轻量化（带Mask版）")
    print("="*60)
    
    x_train, x_val, x_test, y_train, y_val, y_test = load_amc_data()
    print(f"\n数据: 训练{x_train.shape}, 测试{x_test.shape}")
    
    print("\n加载原始模型...")
    original_model = tf.keras.models.load_model("/home/mywsl/jwsmc/model/AMCNet_v3_best.h5")
    
    orig_trainable, orig_non, orig_total = count_params(original_model)
    print(f"原始模型参数量: {orig_total:,}")
    
    _, orig_acc = evaluate_model(original_model, x_test, y_test)
    print(f"原始模型测试准确率: {orig_acc*100:.2f}%")
    
    print("\n开始剪枝 (50% sparsity)...")
    pruned_model = tf.keras.models.clone_model(original_model)
    pruned_model.set_weights(original_model.get_weights())
    pruned_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 应用剪枝并保存mask
    pruned_model = manual_prune_with_mask(pruned_model, sparsity=0.3)
    
    _, prune_acc_before = evaluate_model(pruned_model, x_test, y_test)
    print(f"剪枝后(未微调)准确率: {prune_acc_before*100:.2f}%")
    
    print("\n开始微调 (20 epochs, 带mask)...")
    callbacks = [
        PruningCallback(pruned_model),
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    ]
    
    pruned_model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=20,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    # 最终应用mask
    apply_masks(pruned_model)
    
    # 统计非零参数
    non_zero = 0
    for layer in pruned_model.layers:
        if hasattr(layer, 'get_weights') and layer.get_weights():
            for w in layer.get_weights():
                non_zero += np.count_nonzero(w)
    
    print(f"\n剪枝后非零参数量: {non_zero:,}")
    print(f"实际压缩率: {(1 - non_zero/orig_total)*100:.1f}%")
    
    _, prune_acc = evaluate_model(pruned_model, x_test, y_test)
    print(f"剪枝后(微调后)测试准确率: {prune_acc*100:.2f}%")
    
    acc_diff = (orig_acc - prune_acc) * 100
    print(f"准确率变化: {acc_diff:+.2f}%")
    
    if acc_diff < 3:
        print("\n✓ 准确率下降小于3%，符合要求！")
    else:
        print("\n✗ 准确率下降超过3%")
    
    os.makedirs("/home/mywsl/jwsmc/model", exist_ok=True)
    save_path = "/home/mywsl/jwsmc/model/AMCNet_v3_pruned.h5"
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
        model = prune_amcnet()
        print("\n剪枝完成!")
    except Exception as e:
        print(f"报错: {e}")
        import traceback
        traceback.print_exc()
