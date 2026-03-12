import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, resample_poly
from scipy.signal.windows import hamming
import numpy as np
import pickle
import os

B = 75e6  # 总带宽
B_0 = 2.5e6  # 子带宽
N = 32  # 符号数量
P = 30  # 多陪集采样通道数
L = 30  # 子带数
K = 6  # 活跃子带数

# --------------------  升余弦脉冲成型参数  -------------------- #
SYMBOL_RATE = 1.25e6  # 符号速率 (1.25 MHz)
SAMPLE_RATE = 150e6  # 采样率 (150 MHz)
SAMPLES_PER_SYMBOL = int(SAMPLE_RATE / SYMBOL_RATE)  # 每个符号的采样点数
FILTER_SPAN = 10  # 滤波器符号跨度
ALPHA = 0.35  # 滚降因子


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


# --------------------  生成宽带信号  -------------------- #
def generate_wideband_signal(N, L, K):
    """ 生成宽带信号，每个频段内含不同调制类型 """
    modulations = ['BPSK', 'QPSK', 'QAM16', 'QAM64']
    selected_modulations = np.random.choice(modulations, K, replace=True)

    # 计算总长度
    total_length = 3840

    # 生成时间轴
    t = np.arange(0, total_length) / SAMPLE_RATE  # 时间轴，单位为秒
    freq = np.linspace(0, B, total_length)
    carry_freqs = np.linspace(B_0 / 2, B - B_0 / 2, L)

    f_k = np.random.choice(carry_freqs, size=K, replace=False)
    f_k = np.sort(f_k)  # 按升序排列 f_k


    wideband_signal = np.zeros(total_length, dtype=complex)
    shaped_signals = []

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
        # 保存成型后的信号
        shaped_signals.append(shaped_signal)

    return wideband_signal, shaped_signals, selected_modulations, freq, f_k, carry_freqs


# --------------------  信道效应  -------------------- #
def apply_channel_effects(signal, snr_db):
    """ 依次施加 Rician 衰落 -> 频率漂移 -> 相位偏移 -> AWGN，适用于调制分类 """
    signal = signal.astype(np.complex128)

    original_signal_power = np.mean(np.abs(signal)  ** 2)  # 计算原始信号功率

    #  计算 AWGN 噪声的功率**
    noise_power = original_signal_power / (10 ** (snr_db / 10))  # 计算噪声功率
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))  # 复数噪声

    # **6. 添加噪声**
    signal += noise

    return signal


# --------------------  多陪集采样  -------------------- #
def multicoset_sampling(signal, freq, L, P):
    """ 多陪集采样 """
    c = np.sort(np.random.choice(range(0, L), size=P, replace=False))
    c_t = np.reshape(c, (P, 1))
    y = np.zeros((P, len(signal) // L), dtype=complex)

    for i in range(P):
        y[i, :] = signal[c[i]:len(signal) // L * L:L]

    l = np.reshape(np.linspace(1, L, L), (1, L))
    A = np.exp(-1j * 2 * np.pi * c_t * (l - 1) / L)
    A_inverse = np.linalg.pinv(A)

    f = freq[:len(freq) // L]
    y_freq = np.exp(-1j * 2 * np.pi * c_t * f * (1 / B)) * (np.fft.fft(y) / y.shape[-1])
    x_freq = np.dot(A_inverse, y_freq)
    x_freq_energy = np.sum((np.abs(x_freq)  ** 2))
    x_norm = x_freq / (x_freq_energy  ** 0.5)

    return x_norm


# --------------------  主要数据生成函数  -------------------- #
def MP(N, L, K, P, snr_db):
    """
    生成带有信道效应的宽带信号，并进行多陪集采样
    """
    wideband_signal, shaped_signals, mod_types, freq, f_k, carry_freqs = generate_wideband_signal(N, L, K)
    wideband_signal_addnoise = apply_channel_effects(wideband_signal, snr_db)
    x_norm = multicoset_sampling(wideband_signal_addnoise, freq, L, P)

    # **给频段打标签**
    freq_label = np.zeros(L)

    # **创建调制类型独热码标签**
    mod_onehot_labels = np.zeros((K, 4))  # 4 modulation types: BPSK, QPSK, QAM16, QAM64
    mod_to_idx = {'BPSK': 0, 'QPSK': 1, 'QAM16': 2, 'QAM64': 3}

    for i in range(K):
        index = np.where(np.isclose(carry_freqs, f_k[i]))[0]
        if len(index) > 0:
            freq_label[index] = 1  # 标记 K 个活跃频段
            mod_idx = mod_to_idx[mod_types[i]]
            mod_onehot_labels[i, mod_idx] = 1  # 设置对应的调制类型标签
        else:
            print(f"⚠ 警告：未找到 f_k[{i}] = {f_k[i]} 在 carry_freqs 中的索引！")

    return x_norm, freq_label, wideband_signal_addnoise, mod_onehot_labels


# --------------------  生成数据集  -------------------- #
def generate_dataset():
    """
    生成数据集并分别保存频谱感知数据、调制分类数据及其标签
    """
    spectrum_nvecs_per_key =1200

    modulation_types = ['BPSK', 'QPSK', 'QAM16', 'QAM64']
    snr_vals = np.arange(-18, 20, 2)

    # 初始化数据存储结构
    spectrum_dataset = {}
    freq_labelset = {}
    wideband_signals = {}  # 存储添加信道效应的宽带信号
    mod_labelset = {}  # 存储调制类型的独热码标签

    for snr in snr_vals:
        print(f"Processing SNR: {snr}dB")

        # **初始化数据结构**
        spectrum_dataset[snr] = np.zeros([spectrum_nvecs_per_key, L, 128, 2])
        freq_labelset[snr] = np.zeros([spectrum_nvecs_per_key, L])
        wideband_signals[snr] = np.zeros([spectrum_nvecs_per_key, 3840], dtype=np.complex128)
        mod_labelset[snr] = np.zeros([spectrum_nvecs_per_key, K, 4])

        # **生成 1000 个频谱感知数据样本**
        for i in range(spectrum_nvecs_per_key):
            x_norm, freq_label, wideband_signal_addnoise, mod_onehot_labels = MP(N, L, K, P, snr)
            spectrum_dataset[snr][i, :, :, 0] = np.real(x_norm)
            spectrum_dataset[snr][i, :, :, 1] = np.imag(x_norm)
            freq_labelset[snr][i] = freq_label
            wideband_signals[snr][i] = wideband_signal_addnoise
            mod_labelset[snr][i] = mod_onehot_labels

    print("Dataset generation complete.")

    # **保存数据集**
    base_path1 = "/mnt/d/Myproject/jwsmc/data"
    os.makedirs(base_path1, exist_ok=True)

    # **保存频谱感知数据集**
    spectrum_file_path = os.path.join(base_path1, f'K={K}_spectrum_dataset.pkl')
    with open(spectrum_file_path, 'wb') as file:
        pickle.dump(spectrum_dataset, file)
    print(f'频谱感知数据集已保存为 {spectrum_file_path}')

    # **保存频率标签集**
    freq_labelset_file_path = os.path.join(base_path1, f'K={K}_freq_labelset.pkl')
    with open(freq_labelset_file_path, 'wb') as file:
        pickle.dump(freq_labelset, file)
    print(f'频率标签集已保存为 {freq_labelset_file_path}')

    # **保存宽带信号数据集**
    wideband_file_path = os.path.join(base_path1, f'K={K}_wideband_signals.pkl')
    with open(wideband_file_path, 'wb') as file:
        pickle.dump(wideband_signals, file)
    print(f'宽带信号数据集已保存为 {wideband_file_path}')

    #**保存调制类型标签集**
    mod_labelset_file_path = os.path.join(base_path1, f'K={K}_mod_labelset.pkl')
    with open(mod_labelset_file_path, 'wb') as file:
        pickle.dump(mod_labelset, file)
    print(f'调制类型标签集已保存为 {mod_labelset_file_path}')

    return spectrum_dataset, freq_labelset, wideband_signals, mod_labelset


# 运行数据集生成
generate_dataset()
