import numpy as np
import pickle
from sklearn.model_selection import train_test_split

def load_wss_data(data_path="./data"):
    """加载频谱感知(WSS)任务的数据集(K=6_xxx.pkl)"""
    # 加载输入数据（30×128×2）
    with open(f"{data_path}/K=6_spectrum_dataset.pkl", 'rb') as f:
        x_dict = pickle.load(f)
    # 加载标签（30维0/1）
    with open(f"{data_path}/K=6_freq_labelset.pkl", 'rb') as f:
        y_dict = pickle.load(f)
    
    # 合并所有SNR的数据（按论文拆分训练/验证/测试集）
    x_all = np.concatenate([x_dict[snr] for snr in x_dict.keys()], axis=0)
    y_all = np.concatenate([y_dict[snr] for snr in y_dict.keys()], axis=0)
    
    # 按论文比例拆分：训练集22800，验证集7600，测试集7600
    x_train, x_temp, y_train, y_temp = train_test_split(
        x_all, y_all, test_size=15200/22800, random_state=42  # 15200=7600+7600
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=42
    )
    
    # 输出维度信息（验证是否正确）
    print(f"WSS数据加载完成:")
    print(f"训练集:x_train={x_train.shape}, y_train={y_train.shape}")
    print(f"验证集:x_val={x_val.shape}, y_val={y_val.shape}")
    print(f"测试集:x_test={x_test.shape}, y_test={y_test.shape}")
    return x_train, x_val, x_test, y_train, y_val, y_test

def load_amc_data(data_path="./data", use_snr_balance=False):
    """
    加载调制识别(AMC)任务的数据集(K=4_xxx.pkl)
    
    Args:
        data_path: 数据路径
        use_snr_balance: 是否使用SNR平衡采样（优化3）
    """
    # 加载输入数据（256×2）
    with open(f"{data_path}/K=4_modulation_dataset.pkl", 'rb') as f:
        x_dict = pickle.load(f)
    # 加载标签（4维独热码）
    with open(f"{data_path}/K=4_mod_labelset.pkl", 'rb') as f:
        y_dict = pickle.load(f)
    
    if use_snr_balance:
        # SNR分层采样：平衡高低SNR数据的权重
        print("使用SNR平衡采样...")
        
        # 获取所有SNR
        snrs = sorted(x_dict.keys())
        print(f"SNR范围: {snrs[0]}dB 到 {snrs[-1]}dB")
        
        # 将SNR分为高、中、低三组
        low_snr = [s for s in snrs if s < 0]
        mid_snr = [s for s in snrs if 0 <= s < 10]
        high_snr = [s for s in snrs if s >= 10]
        
        print(f"低SNR: {low_snr}")
        print(f"中SNR: {mid_snr}")
        print(f"高SNR: {high_snr}")
        
        # 从每个SNR组中采样
        samples_per_group = 3800  # 每组采样数量
        
        x_train_list, y_train_list = [], []
        x_val_list, y_val_list = [], []
        x_test_list, y_test_list = [], []
        
        for snr_group, group_name in [(low_snr, '低SNR'), (mid_snr, '中SNR'), (high_snr, '高SNR')]:
            if not snr_group:
                continue
                
            # 合并当前SNR组的数据
            x_group = np.concatenate([x_dict[snr] for snr in snr_group], axis=0)
            y_group = np.concatenate([y_dict[snr] for snr in snr_group], axis=0)
            
            # 打乱顺序
            indices = np.random.permutation(len(x_group))
            x_group = x_group[indices]
            y_group = y_group[indices]
            
            # 按比例拆分
            n_train = int(len(x_group) * 0.6)
            n_val = int(len(x_group) * 0.2)
            
            x_train_list.append(x_group[:n_train])
            y_train_list.append(y_group[:n_train])
            x_val_list.append(x_group[n_train:n_train+n_val])
            y_val_list.append(y_group[n_train:n_train+n_val])
            x_test_list.append(x_group[n_train+n_val:])
            y_test_list.append(y_group[n_train+n_val:])
        
        # 合并所有组
        x_train = np.concatenate(x_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        x_val = np.concatenate(x_val_list, axis=0)
        y_val = np.concatenate(y_val_list, axis=0)
        x_test = np.concatenate(x_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)
        
        # 打乱最终数据
        indices = np.random.permutation(len(x_train))
        x_train, y_train = x_train[indices], y_train[indices]
        indices = np.random.permutation(len(x_val))
        x_val, y_val = x_val[indices], y_val[indices]
        indices = np.random.permutation(len(x_test))
        x_test, y_test = x_test[indices], y_test[indices]
    else:
        # 原始方式：直接合并所有SNR
        x_all = np.concatenate([x_dict[snr] for snr in x_dict.keys()], axis=0)
        y_all = np.concatenate([y_dict[snr] for snr in y_dict.keys()], axis=0)
        
        # 按论文比例拆分：训练集66400，验证集22800，测试集22800
        x_train, x_temp, y_train, y_temp = train_test_split(
            x_all, y_all, test_size=15200/114000, random_state=42
        )
        x_val, x_test, y_val, y_test = train_test_split(
            x_temp, y_temp, test_size=0.5, random_state=42
        )
    
    # 输出维度信息（验证是否正确）
    print(f"AMC数据加载完成:")
    print(f"训练集:x_train={x_train.shape}, y_train={y_train.shape}")
    print(f"验证集:x_val={x_val.shape}, y_val={y_val.shape}")
    print(f"测试集:x_test={x_test.shape}, y_test={y_test.shape}")
    return x_train, x_val, x_test, y_train, y_val, y_test

# 测试加载（运行脚本时验证）
if __name__ == "__main__":
    try:
        print("开始加载WSS数据...")
        load_wss_data()
        print("-"*50)
        print("开始加载AMC数据...")
        load_amc_data()
        print("所有数据加载完成！")
    except Exception as e:
        print(f"报错信息：{type(e).__name__} - {e}")