from math import erf, erfc, exp, log2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


matplotlib.use('TkAgg')  # 或者 'Qt5Agg'

# 基本物理常量
SPEED_OF_LIGHT = 3e8                  # 光速 (m/s)

# RF链路参数
RF_FREQUENCY = 2.4e9                  # RF载波频率 (Hz)
RF_PATH_LOSS_EXPONENT = 2             # RF路径损耗指数
RF_SHADOWING_STD = 0.05                # RF阴影衰落标准差 (dB)
RF_BANDWIDTH = 100e6                  # RF带宽 (Hz)，100MHz
RF_NOISE_POWER_DENSITY = 4e-21     # RF噪声功率谱密度 (W/Hz)
RF_TRANSMIT_POWER = 1                # RF发射功率 (W)

# FSO链路参数
FSO_WAVELENGTH = 1550e-9              # FSO波长 (m)，1550nm
FSO_BANDWIDTH = 1e9                   # FSO带宽 (Hz)，1GHz
FSO_NOISE_POWER_DENSITY = 4e-13     # FSO噪声功率谱密度 (W/Hz)
FSO_RECEIVER_AREA = 0.01              # FSO接收器面积 (m²)
FSO_TRANSMIT_POWER = 1                # FSO发射功率 (W)
FSO_ATMOSPHERIC_ATTENUATION = 100e-5  # FSO大气衰减系数 (1/m)
FSO_SHADOWING_STD = 0.005             # FSO阴影衰落标准差 (dB)
FSO_POINTING_ERROR_CONSTANT = 25     # FSO指向误差常数
FSO_REFRACTION_INDEX = 1e-12          # FSO折射率结构常数
FSO_DIVERGENCE_ANGLE = 1 * np.pi / 180  # FSO发散角 (rad)，1度

# 光束腰斑初始值 (m)
FSO_BEAM_WAIST_INITIAL = FSO_WAVELENGTH / (np.pi * np.tan(FSO_DIVERGENCE_ANGLE))

# 波数 (1/m)
FSO_WAVE_NUMBER = 2 * np.pi / FSO_WAVELENGTH

# 孔径半径 (m)
FSO_APERTURE_RADIUS = 8.0


def calculate_rf_channel_gain(distance, relative_velocity):
    """计算RF信道增益，包括路径损耗、阴影衰落"""

    effective_distance = distance + relative_velocity / 2
    
    # 路径损耗
    path_loss = np.where(
        effective_distance == 0,
        1.0,
        (SPEED_OF_LIGHT / (4 * np.pi * RF_FREQUENCY * effective_distance)) ** RF_PATH_LOSS_EXPONENT
    )
    
    # 阴影衰落
    shadowing = 10 ** (np.random.normal(0, RF_SHADOWING_STD) / 10)

    result = path_loss * shadowing
    return result


def calculate_rf_packet_error_rate(distance, packet_length_bits, relative_velocity):
    """计算RF链路的包错误率和信噪比"""

    noise_power = RF_BANDWIDTH * RF_NOISE_POWER_DENSITY
    channel_gain = calculate_rf_channel_gain(distance, relative_velocity)
    signal_noise_ratio = np.maximum((channel_gain * RF_TRANSMIT_POWER) / noise_power, 0)

    # Rayleigh fading + BPSK
    bit_error_rate = 0.5 * (1 - np.sqrt(signal_noise_ratio / (1 + signal_noise_ratio)))
    # bit_error_rate = 0.5 * np.exp(-signal_noise_ratio / 2)

    packet_error_rate = 1 - (1 - bit_error_rate) ** packet_length_bits

    return packet_error_rate, signal_noise_ratio


def calculate_fso_pointing_error_gain(distance, velocity_perpendicular):
    """计算FSO指向误差增益"""
    
    effective_distance = distance / 5 + 50
    
    # 计算光束参数
    coherence_length = (0.55 * FSO_REFRACTION_INDEX * (FSO_WAVE_NUMBER ** 2) * effective_distance) ** (-0.6)
    
    epsilon = 1 + (2 * FSO_BEAM_WAIST_INITIAL ** 2) / (coherence_length ** 2)
    beam_waist_at_distance = FSO_BEAM_WAIST_INITIAL * np.sqrt(
        1 + epsilon * ((FSO_WAVELENGTH * effective_distance) / (np.pi * FSO_BEAM_WAIST_INITIAL ** 2)) ** 2
    )
    
    # 计算指向误差
    jitter_radius = FSO_APERTURE_RADIUS * FSO_POINTING_ERROR_CONSTANT * np.abs(velocity_perpendicular ** 0.35) / 2
    
    normalized_aperture = (np.sqrt(np.pi) * FSO_APERTURE_RADIUS) / (np.sqrt(2) * beam_waist_at_distance)
    aperture_factor = erf(normalized_aperture) ** 2
    
    # 防止除零
    denominator = 2 * normalized_aperture * np.exp(-normalized_aperture ** 2) + 1e-5
    equivalent_beam_width = (beam_waist_at_distance ** 2 * np.sqrt(np.pi) * erf(normalized_aperture)) / denominator
    
    pointing_error_gain = aperture_factor * np.exp(-2 * (jitter_radius ** 2) / equivalent_beam_width)
    
    return pointing_error_gain


def calculate_fso_channel_gain(distance, velocity_parallel, velocity_perpendicular):
    """计算FSO信道增益，包括大气损耗、阴影衰落、指向误差和多径效应"""

    # 大气损耗
    effective_distance = distance + velocity_parallel / 2
    atmospheric_loss = np.where(
        effective_distance == 0,
        1.0,
        np.exp(-effective_distance * FSO_ATMOSPHERIC_ATTENUATION)
    )
    
    # 阴影衰落
    shadowing = 10 ** (np.random.normal(0, FSO_SHADOWING_STD) / 10)
    
    # 多径效应
    multipath_factor = 1 / (1 + (np.abs(velocity_parallel + 10) / 50) ** 0.5)
    
    # 指向误差
    pointing_error = calculate_fso_pointing_error_gain(distance, velocity_perpendicular)
    
    result = atmospheric_loss * shadowing * pointing_error * multipath_factor
    return result


def calculate_fso_packet_error_rate(distance, packet_length_bits, velocity_parallel, velocity_perpendicular):
    """计算FSO链路的包错误率和信噪比"""

    noise_power = FSO_BANDWIDTH * FSO_NOISE_POWER_DENSITY
    channel_gain = calculate_fso_channel_gain(distance, velocity_parallel, velocity_perpendicular)
    signal_noise_ratio = channel_gain * FSO_TRANSMIT_POWER * FSO_RECEIVER_AREA / noise_power
    
    # OOK调制误码率
    bit_error_rate = 0.5 * erfc(np.sqrt(signal_noise_ratio))
    packet_error_rate = 1 - (1 - bit_error_rate) ** packet_length_bits

    return packet_error_rate, signal_noise_ratio


def rf_parameter_testing(distance, velocity_parallel):
    # 链路稳定性表达式参数取值测试

    distance = (distance + 0.01) / 200
    velocity_parallel /= 40
    per = np.exp(-0.05 * distance * abs(velocity_parallel + 0.1) ** 2) * exp(- (2 + velocity_parallel / 10) * (distance ** 2))
    return per


def fso_parameter_testing(distance, velocity_parallel, velocity_perpendicular):
    # 链路稳定性表达式参数取值测试

    distance = (distance + 0.01) / 500
    velocity_parallel = velocity_parallel / 40
    velocity_perpendicular = velocity_perpendicular / 40
    per = exp(- 0.4 * distance) * exp(-0.01 * abs(velocity_parallel + 0.1)) * exp(9 * log2(distance ** 0.05) * velocity_perpendicular)
    return per


if __name__ == '__main__':

    packet = 128 * 5
    velocity_parallel = 20
    velocity_perpendicular = 10
    distance = 100
    test_range = list(range(-40, 40))
    test_rf = [rf_parameter_testing(distance, velocity_parallel) for velocity_parallel in test_range]
    # test_fso = [fso_parameter_testing(distance, velocity_parallel, velocity_perpendicular) for velocity_perpendicular in test_range]
    true_rf = [1 - calculate_rf_packet_error_rate(distance, packet, velocity_parallel)[0] for velocity_parallel in test_range]
    # true_fso = [1 - calculate_fso_packet_error_rate(distance, packet, velocity_parallel, velocity_perpendicular)[0] for velocity_perpendicular in test_range]

    plt.figure(figsize=(6, 4), dpi=100)  # 设置图像大小和dpi
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 也可以使用 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    plt.plot(test_range, true_rf, label='rf', color='#FF0000', linestyle='-',  alpha=1,  linewidth=1.5)
    # plt.plot(test_range, true_fso, label='fso', color='#00FF00', linestyle='-',  alpha=1,  linewidth=1.5)
    plt.plot(test_range, test_rf, label='test_rf', color='#AA0000', linestyle='-',  alpha=0.5,  linewidth=1.5)
    # plt.plot(test_range, test_fso, label='test_fso', color='#00AA00', linestyle='-',  alpha=0.5,  linewidth=1.5)

    plt.grid(True)
    plt.legend(loc='upper left', prop={'family': 'Microsoft YaHei'})
    plt.show()


