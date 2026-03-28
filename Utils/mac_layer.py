import random
from math import log2
import numpy as np
from Utils.physics_layer import calculate_fso_packet_error_rate, calculate_rf_packet_error_rate
from Utils.utils import vector_projection, calculate_angle_between_vectors

COM_DISTANCE_DIRECTIONAL = 500        # 定向通信最大距离 (m)
COM_DISTANCE_OMNI = 200               # 全向通信最大距离 (m)

# 传播时延计算
SPEED_OF_LIGHT = 3e8                  # 光速 (m/s)

# 排队时延参数
ARRIVAL_RATE_MAP = {
    "hc_base": 7.0,                   # HC_BASE协议到达率 (packets/s)
    "hci_base": 7.0,                  # HCI_BASE协议到达率 (packets/s)
    "hc_olsr": 5.0,                   # HC_OLSR协议到达率 (packets/s)
    "hci_olsr": 4.0,                  # HCI_OLSR协议到达率 (packets/s)
    "o_olsr": 6.0,                    # OC_OLSR协议到达率 (packets/s)
    "d_olsr": 4.0                     # DC_OLSR协议到达率 (packets/s)
}
DEFAULT_ARRIVAL_RATE = 4.0            # 默认到达率 (packets/s)

# CSMA/CA接入时延参数
SUCCESS_PROBABILITY_MAP = {
    "o_olsr": 0.40,                   # OC_OLSR成功概率
    "d_olsr": 0.45,                    # DC_OLSR成功概率
    "hc_base": 0.50,
    "hc_olsr": 0.55,
    "hci_base": 0.50,
    "hci_olsr": 0.65
}
DEFAULT_SUCCESS_PROBABILITY = 0.50    # 默认成功概率

MAX_ACCESS_ATTEMPTS = 10              # 最大接入尝试次数
BASE_PROCESSING_DELAY = 50e-6         # 基础处理时延 (s)
ACK_TIMEOUT = 0.0001                  # ACK超时时延 (s)
BACKOFF_BASE = 0.00001                # 退避时延基数 (s)

# 协议处理时延参数
DEFAULT_PROCESSING_DELAY = 5e-4       # 默认协议处理时延 (s)

# FSO指向调整时延参数
FSO_POINTING_DELAY_FACTOR = 1e5       # FSO指向调整时延因子 (rad/s)


def calculate_transmission_delay(packet_length_bits, signal_noise_ratio, bandwidth):
    """计算传输时延：数据包长度除以信道容量"""
    channel_capacity = bandwidth * log2(1 + signal_noise_ratio)
    return packet_length_bits / channel_capacity if channel_capacity > 0 else float('inf')


def calculate_propagation_delay(distance):
    """计算传播时延：距离除以光速"""
    distance = np.asarray(distance)
    result = distance / SPEED_OF_LIGHT
    return float(result) if result.shape == () else result


def calculate_queuing_delay(protocol_type, packet_length_bits, signal_noise_ratio):
    """基于Little定律计算排队时延"""
    arrival_rate = ARRIVAL_RATE_MAP.get(protocol_type, DEFAULT_ARRIVAL_RATE)
    
    packet_length_bits = np.asarray(packet_length_bits)
    signal_noise_ratio = np.asarray(signal_noise_ratio)
    
    service_rate = (signal_noise_ratio * log2(1 + signal_noise_ratio)) / (packet_length_bits * 8)
    utilization = np.where(service_rate > 0, np.minimum(arrival_rate / service_rate, 0.98), 0.98)

    high_utilization = utilization >= 0.98
    average_packets = np.where(high_utilization, 0, utilization / (1 - utilization))
    delay = np.where(high_utilization, 50 / arrival_rate, average_packets / arrival_rate)
    return float(delay) if delay.shape == () else delay


def calculate_access_delay(protocol_type):
    """计算CSMA/CA信道接入时延"""
    success_probability = SUCCESS_PROBABILITY_MAP.get(protocol_type, DEFAULT_SUCCESS_PROBABILITY)
    
    total_delay = BASE_PROCESSING_DELAY
    attempts = 0
    
    while attempts < MAX_ACCESS_ATTEMPTS:
        attempts += 1
        backoff = np.random.uniform(0, BACKOFF_BASE * (2 ** min(attempts, 10)))
        total_delay += backoff
        
        if np.random.rand() <= success_probability:
            break
        
        total_delay += ACK_TIMEOUT
    
    return total_delay


def calculate_processing_delay(protocol_type):
    """计算协议处理时延"""
    return DEFAULT_PROCESSING_DELAY


def calculate_mac_delay(receiver_node, sender_node, packet_length_bits, protocol_type):
    """
    计算MAC层总延迟
    返回: 总延迟(秒)，如果传输失败返回负值错误码
    错误码: -1=丢包, -2=超距
    """

    receiver_pos = np.asarray(receiver_node.position)
    sender_pos = np.asarray(sender_node.position)
    distance = np.linalg.norm(receiver_pos - sender_pos)
    
    # 根据协议类型选择通信参数
    if protocol_type == "o_olsr":
        max_distance = COM_DISTANCE_OMNI
        bandwidth = 100e6
        link_type = "rf"
    else:
        max_distance = COM_DISTANCE_DIRECTIONAL
        bandwidth = 1e9
        link_type = "fso"
    
    if distance > max_distance:
        return -2
    
    # 计算相对速度向量（节点远离时为正值）
    receiver_pos_before = np.asarray(receiver_node.position_before)
    sender_pos_before = np.asarray(sender_node.position_before)
    
    previous_distance_vector = receiver_pos_before - sender_pos_before
    current_distance_vector = receiver_pos - sender_pos
    velocity_vector = current_distance_vector - previous_distance_vector
    
    angle = calculate_angle_between_vectors(previous_distance_vector, current_distance_vector)
    velocity_parallel = vector_projection(velocity_vector, current_distance_vector)
    velocity_perpendicular = np.sqrt(max(np.linalg.norm(velocity_vector) ** 2 - velocity_parallel ** 2, 0))
    
    # 计算包错误率
    if link_type == "fso":
        packet_error_rate, signal_noise_ratio = calculate_fso_packet_error_rate(
            distance, packet_length_bits, velocity_parallel, velocity_perpendicular
        )
        total_delay = angle / FSO_POINTING_DELAY_FACTOR  # FSO指向调整时延
    else:
        packet_error_rate, signal_noise_ratio = calculate_rf_packet_error_rate(
            distance, packet_length_bits, velocity_parallel
        )
        total_delay = 0
    
    # 判断是否丢包
    if random.random() < packet_error_rate:
        return -1
    
    # 累加各组成部分时延
    total_delay += calculate_transmission_delay(packet_length_bits, signal_noise_ratio, bandwidth)
    total_delay += calculate_propagation_delay(distance)
    total_delay += calculate_queuing_delay(protocol_type, packet_length_bits, signal_noise_ratio)
    total_delay += calculate_access_delay(protocol_type)
    total_delay += calculate_processing_delay(protocol_type)
    
    return total_delay
