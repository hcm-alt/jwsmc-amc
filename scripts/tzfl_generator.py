# 配置变量文件
import numpy as np
import pickle
import os
from scipy.signal import butter, filtfilt
from scipy.signal import firwin, lfilter, resample_poly
from scipy.signal.windows import hamming

# 参数设置
B = 75e6  # 总带宽
B_0 = 2.5e6  # 子带宽
N = 32  # 符号数量
P = 30  # 多陪集采样通道数
L = 30  # 子带数
K = 4  # 活跃子带数

# 升余弦脉冲成型参数
SYMBOL_RATE = 1.25e6  
SAMPLE_RATE = 150e6  
SAMPLES_PER_SYMBOL = int(SAMPLE_RATE / SYMBOL_RATE)  # 每个符号的采样点数
FILTER_SPAN = 10  # 滤波器符号跨度
ALPHA = 0.35  # 滚降因子

# 调制类型
modulation_types = ['BPSK', 'QPSK', 'QAM16', 'QAM64']
modulation_to_one_hot = {mod: np.eye(len(modulation_types))[i] for i, mod in enumerate(modulation_types)}

# SNR 范围
snr_vals = np.arange(-18, 20, 2)

# 数据集存储路径
base_path2 = "/mnt/d/Myproject/jwsmc/data"
os.makedirs(base_path2, exist_ok=True)

# 生成数据集
def generate_dataset():
    # 初始化数据存储结构
    subband_dataset = {}
    subband_labels = {}

    for snr in snr_vals:
        # 初始化当前 SNR 下的存储数组
        subband_data = np.empty((4000, 256, 2), dtype=np.float32)  # 存储子带 IQ 数据
        subband_label = np.empty((4000, len(modulation_types)), dtype=np.float32)  # 存储独热码标签

        # 生成 1000 组宽带信号，每组包含 4 个子带
        for i in range(1000):

            if (i + 1) % 50 == 0:  # 每 50 组打印一次进度
                print(f"SNR={snr}dB: 已完成 {i + 1}/1000 组宽带信号生成")

            # 生成宽带信号
            wideband_signal, f_k, mod_types = generate_wideband_signal(N, L, K)

            # 施加信道效应
            signal_with_effects = apply_channel_effects(wideband_signal, snr)

            # 提取子带 IQ 数据
            subband_iq = extract_subband_iq( signal_with_effects , f_k, snr)

            # 将调制类型转换为独热码标签
            one_hot_labels = np.array([modulation_to_one_hot[mod] for mod in mod_types])
            #print(one_hot_labels)
            # 将 K 个子带的数据和标签存储到数组中
            subband_data[i * K:(i + 1) * K] = subband_iq
            subband_label[i * K:(i + 1) * K] = one_hot_labels

        # 存储当前 SNR 下的数据和标签
        subband_dataset[snr] = subband_data
        subband_labels[snr] = subband_label
    # 保存数据集
    modulation_file_path = os.path.join(base_path2, f'K={K}_modulation_dataset.pkl')
    with open(modulation_file_path, 'wb') as file:
        pickle.dump(subband_dataset, file)
    print(f'调制分类数据集已保存为 {modulation_file_path}')

    # 保存调制标签集
    mod_labelset_file_path = os.path.join(base_path2, f'K={K}_mod_labelset.pkl')
    with open(mod_labelset_file_path, 'wb') as file:
        pickle.dump(subband_labels, file)
    print(f'调制标签集已保存为 {mod_labelset_file_path}')

    return subband_dataset, subband_labels


def bits_to_symbols(bits, bit_length):
    """ 将 N x bit_length 的比特序列转换为 0~2^bit_length-1 之间的整数 """
    symbols = np.zeros(bits.shape[0], dtype=int)
    for i in range(bit_length):
        symbols |= bits[:, i] << (bit_length - 1 - i)
    return symbols


def gray_code_mapping(modulation_type, N):
    """ 生成符合 RML2016 格雷码映射方式的 IQ 符号 """

    if modulation_type == 'BPSK':
        symbols = np.array([-1, 1])
        gray_map = {0: -1, 1: 1}
        bit_length = 1

    elif modulation_type == 'QPSK':
        # QPSK: 使用标准的灰码顺序 {0, 1, 3, 2}
        symbols = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)  # 相位顺序: π/4, 3π/4, 5π/4, 7π/4
        gray_map = {0: symbols[0], 1: symbols[1], 3: symbols[2], 2: symbols[3]}
        bit_length = 2

    elif modulation_type == 'QAM16':
        rml16_map = [
            2, 6, 14, 10,
            3, 7, 15, 11,
            1, 5, 13, 9,
            0, 4, 12, 8
        ]
        qam16_real = np.array([-3, -1, 1, 3]) / np.sqrt(10)
        qam16_imag = np.array([-3, -1, 1, 3]) / np.sqrt(10)
        symbols = (qam16_real[:, None] + 1j * qam16_imag).flatten()
        gray_map = {rml16_map[i]: symbols[i] for i in range(16)}
        bit_length = 4

    elif modulation_type == 'QAM64':
        rml64_map = [
            0, 32, 8, 40, 3, 35, 11, 43,
            48, 16, 56, 24, 51, 19, 59, 27,
            12, 44, 4, 36, 15, 47, 7, 39,
            60, 28, 52, 20, 63, 31, 55, 23,
            2, 34, 10, 42, 1, 33, 9, 41,
            50, 18, 58, 26, 49, 17, 57, 25,
            14, 46, 6, 38, 13, 45, 5, 37,
            62, 30, 54, 22, 61, 29, 53, 21
        ]
        qam64_real = np.array([-7, -5, -3, -1, 1, 3, 5, 7]) / np.sqrt(42)
        qam64_imag = np.array([-7, -5, -3, -1, 1, 3, 5, 7]) / np.sqrt(42)
        symbols = (qam64_real[:, None] + 1j * qam64_imag).flatten()
        gray_map = {rml64_map[i]: symbols[i] for i in range(64)}
        bit_length = 6

    else:
        raise ValueError(f"Unsupported modulation type: {modulation_type}")

    # 生成随机比特流
    bits = np.random.randint(0, 2, size=(N, bit_length))

    # 将比特转换为整数索引
    bit_values = bits_to_symbols(bits, bit_length)

    # 映射到 IQ 符号
    iq_data = np.array([gray_map[b] for b in bit_values])

    return iq_data


# --------------------  根升余弦滤波器设计  -------------------- #
def root_raised_cosine_filter(samples_per_symbol, span, alpha):
    """
    设计根升余弦滤波器
    :param samples_per_symbol: 每个符号的采样点数
    :param span: 滤波器符号跨度
    :param alpha: 滚降因子
    :return: 根升余弦滤波器系数
    """
    # 滤波器长度
    ntaps = span * samples_per_symbol + 1
    # 时间轴
    t = np.arange(-ntaps // 2, ntaps // 2 + 1) / samples_per_symbol
    # 根升余弦滤波器公式
    rrc_filter = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti == 0:
            rrc_filter[i] = 1 - alpha + (4 * alpha / np.pi)
        elif alpha != 0 and np.abs(ti) == 1 / (4 * alpha):
            rrc_filter[i] = (alpha / np.sqrt(2)) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha)) +
                                                    (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha)))
        else:
            rrc_filter[i] = (np.sin(np.pi * ti * (1 - alpha)) +
                             4 * alpha * ti * np.cos(np.pi * ti * (1 + alpha))) / \
                            (np.pi * ti * (1 - (4 * alpha * ti) ** 2))
    # 归一化
    rrc_filter /= np.sqrt(samples_per_symbol)
    return rrc_filter


# --------------------  脉冲成型  -------------------- #
def pulse_shaping(iq_data, samples_per_symbol, rrc_filter):
    """
    对 IQ 数据进行脉冲成型
    :param iq_data: 基带 IQ 数据
    :param samples_per_symbol: 每个符号的采样点数
    :param rrc_filter: 根升余弦滤波器系数
    :return: 成型后的信号
    """
    # 对 IQ 数据进行上采样
    upsampled = np.zeros(len(iq_data) * samples_per_symbol, dtype=np.complex64)
    upsampled[::samples_per_symbol] = iq_data

    # 对 I 和 Q 分量分别滤波
    filtered_i = lfilter(rrc_filter, 1, upsampled.real)
    filtered_q = lfilter(rrc_filter, 1, upsampled.imag)

    return filtered_i + 1j * filtered_q


# 生成宽带信号
def generate_wideband_signal(N, L, K):
    # 计算总长度
    total_length = 3840
    modulation_types = ['BPSK', 'QPSK', 'QAM16', 'QAM64']
    # 生成时间轴
    t = np.arange(0, total_length) / SAMPLE_RATE  # 时间轴，单位为秒

    carry_freqs = np.linspace(B_0 / 2, B - B_0 / 2, L)
    f_k = np.random.choice(carry_freqs, size=K, replace=False)  # 随机选择 K 个中心频率
    f_k = np.sort(f_k)  # 按升序排列 f_k

    # 随机选择 K 种调制类型（不重复）
    selected_modulations = np.random.choice(modulation_types, size=K, replace=False)


    wideband_signal = np.zeros(total_length, dtype=complex)

    # 设计根升余弦滤波器
    rrc_filter = root_raised_cosine_filter(SAMPLES_PER_SYMBOL, FILTER_SPAN, ALPHA)

    for i in range(K):
        # 生成基带信号
        iq_data = gray_code_mapping(selected_modulations[i], N)

        # 对基带信号进行脉冲成型
        shaped_signal = pulse_shaping(iq_data, SAMPLES_PER_SYMBOL, rrc_filter)

        # 调制到载波频率
        phase_offset = np.exp(1j * np.random.uniform(0, 2 * np.pi))
        carrier = np.exp(1j * 2 * np.pi * f_k[i] * t)
        wideband_signal += shaped_signal * carrier * phase_offset

    return wideband_signal, f_k, selected_modulations

# 施加信道效应
def apply_channel_effects(signal, snr_db):
    signal = signal.astype(np.complex128)
    original_signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = original_signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    signal += noise

    return signal
import scipy.signal as signal


# -------------------- 滤波器设计 -------------------- #
def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low')
    return b, a

def lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order)
    return signal.filtfilt(b, a, data)


# -------------------- 子带IQ提取（新方法） -------------------- #
def extract_subband_iq(wideband_signal, f_k, snr_db):
    subband_signals = []
    t = np.arange(len(wideband_signal)) / SAMPLE_RATE

    for center_freq in f_k:
        # Step 1: 搬移到基带
        baseband_signal = wideband_signal * np.exp(-1j * 2 * np.pi * center_freq * t)

        # Step 2: 低通滤波（子带带宽=2.5MHz，保守一点可以设成1.2*B_0）
        filtered_signal = lowpass_filter(baseband_signal, cutoff=1e6, fs=SAMPLE_RATE)

        # Step 3: 裁剪或重采样到256点
        if len(filtered_signal) < 256:
            padded_signal = np.zeros(256, dtype=np.complex128)
            padded_signal[:len(filtered_signal)] = filtered_signal
            filtered_signal = padded_signal
        else:
            filtered_signal = signal.resample_poly(filtered_signal, up=256, down=len(filtered_signal))

        # Step 4: 归一化
        energy = np.sum(np.abs(filtered_signal)**2)  
        filtered_signal /= (energy ** 0.5)

        subband_real = np.real(filtered_signal)
        subband_imag = np.imag(filtered_signal)
        subband_iq = np.stack([subband_real, subband_imag], axis=1)

        subband_signals.append(subband_iq)

    return np.stack(subband_signals, axis=0)

# 运行数据集生成
generate_dataset()
