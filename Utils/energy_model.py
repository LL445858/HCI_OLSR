import math
import numpy as np

# 重力相关
DRONE_WEIGHT = 9.8                    # 无人机重量 (N)，假设质量为1kg
DRONE_MASS = 1.0                      # 无人机质量 (kg)

# 空气动力学参数
AIR_DENSITY = 1.225                   # 空气密度 (kg/m³)，标准海平面值
FUSELAGE_AREA = 0.008                 # 机身截面积 (m²)
DRAG_RATIO = 0.6                      # 阻力比，用于计算型阻功率

# 旋翼参数
ROTOR_RADIUS = 0.15                   # 旋翼半径 (m)
ROTOR_AREA = 0.4                      # 旋翼盘面积 (m²)
ROTOR_SPEED = 300                     # 旋翼转速 (rpm)
ROTOR_SOLIDITY = 0.05                 # 旋翼实度 (桨叶面积/桨盘面积)
PROFILE_DRAG_COEFFICIENT = 0.01       # 翼型阻力系数
INDUCED_POWER_FACTOR = 0.1            # 诱导功率因子，用于修正诱导功率

# 桨尖速度 (m/s)
TIP_SPEED = ROTOR_SPEED * ROTOR_RADIUS

# 功率计算系数 c1-c5
c1 = 3 / (TIP_SPEED ** 2)
c2 = AIR_DENSITY * FUSELAGE_AREA / (2 * DRONE_WEIGHT)
c3 = DRONE_MASS / DRONE_WEIGHT
c4 = AIR_DENSITY * ROTOR_AREA / DRONE_WEIGHT
c5 = 0.5 * DRAG_RATIO * AIR_DENSITY * ROTOR_SOLIDITY * ROTOR_AREA

# 型阻功率 (W)
PROFILE_POWER = (PROFILE_DRAG_COEFFICIENT * AIR_DENSITY * ROTOR_SOLIDITY * ROTOR_AREA *
                 (ROTOR_SPEED ** 3) * (ROTOR_RADIUS ** 3) / 8)

# 诱导功率 (W)
INDUCED_POWER = (1 + INDUCED_POWER_FACTOR) * (DRONE_WEIGHT ** 1.5) / ((2 * AIR_DENSITY * ROTOR_AREA) ** 0.5)

# FSO链路能耗参数
ENERGY_PER_BIT_FSO = 5e-11            # FSO每比特能耗系数 (J/bit/m²)
FSO_ELECTRONICS_ENERGY_PER_BIT = 1e-8 # FSO电子器件每比特能耗 (J/bit)
FSO_POINTING_ENERGY = 8e-4            # FSO指向能耗 (J)

# RF链路能耗参数
ENERGY_PER_BIT_RF = 1e-10             # RF每比特能耗系数 (J/bit/m²)
RF_ELECTRONICS_ENERGY_PER_BIT = 1e-8  # RF电子器件每比特能耗 (J/bit)


def calculate_straight_flight_power(velocity_current, velocity_previous):
    """计算水平直线飞行的功率消耗"""

    velocity_current = np.asarray(velocity_current)
    velocity_previous = np.asarray(velocity_previous)
    
    # 计算速度大小（向量化）
    velocity_magnitude = np.sqrt(velocity_current[..., 0] ** 2 + velocity_current[..., 1] ** 2)
    
    # 计算加速度大小（向量化）
    velocity_diff = velocity_current - velocity_previous
    acceleration_magnitude = np.sqrt(velocity_diff[..., 0] ** 2 + velocity_diff[..., 1] ** 2)
    
    # 计算阻力项（向量化）
    drag_term = c2 * (velocity_magnitude ** 2) + c3 * acceleration_magnitude * velocity_magnitude
    
    # 计算总功率（向量化）
    power = (PROFILE_POWER * (1 + c1 * (velocity_magnitude ** 2)) +
             INDUCED_POWER * np.sqrt(1 + drag_term ** 2) *
             np.sqrt(np.sqrt(1 + drag_term ** 2 + (c4 ** 2) * (velocity_magnitude ** 4)) - c4 * (velocity_magnitude ** 2)) +
             c5 * (velocity_magnitude ** 3))
    
    return float(power) if np.isscalar(power) or power.shape == () else power


def calculate_vertical_flight_power(velocity_current, velocity_previous, height_current, height_previous):
    """计算垂直飞行的功率消耗"""

    velocity_current = np.asarray(velocity_current)
    velocity_previous = np.asarray(velocity_previous)
    
    # 提取垂直速度分量
    vertical_velocity = velocity_current[..., 2] if velocity_current.ndim > 0 else velocity_current[2]
    previous_vertical_velocity = velocity_previous[..., 2] if velocity_previous.ndim > 0 else velocity_previous[2]
    
    # 计算垂直加速度
    vertical_acceleration = np.abs(vertical_velocity - previous_vertical_velocity)
    
    # 基础功率（型阻功率 + 诱导功率）
    base_power = PROFILE_POWER + INDUCED_POWER
    
    # 计算高度变化和动能变化
    height_change = np.asarray(height_current) - np.asarray(height_previous)
    kinetic_change = DRONE_MASS * (vertical_velocity ** 2 - previous_vertical_velocity ** 2) / 4
    
    # 使用向量化条件计算
    is_descending = np.asarray(height_current) < np.asarray(height_previous)

    # 计算有效重量和诱导速度
    effective_weight = np.where(is_descending,
                                DRONE_WEIGHT - DRONE_MASS * vertical_acceleration,
                                DRONE_WEIGHT + DRONE_MASS * vertical_acceleration)
    
    # 确保平方根内的值非负
    sqrt_term = vertical_velocity ** 2 + 2 * effective_weight / (AIR_DENSITY * ROTOR_AREA)
    sqrt_term = np.maximum(sqrt_term, 0)  # 防止负数导致NaN
    induced_velocity = np.sqrt(sqrt_term)
    
    # 计算总功率
    power = base_power + DRONE_WEIGHT * height_change / 2 - kinetic_change + effective_weight / 2 * induced_velocity
    
    return float(power) if np.isscalar(power) or power.shape == () else power


def calculate_mobile_energy(velocity_current, velocity_previous, height_current, height_previous):
    """计算无人机移动能耗（水平+垂直）"""

    straight_power = calculate_straight_flight_power(velocity_current, velocity_previous)
    vertical_power = calculate_vertical_flight_power(velocity_current, velocity_previous, height_current, height_previous)
    return np.abs(straight_power + vertical_power)


def calculate_communication_energy_energy(bits_sent, bits_received, distance, link_type):
    """计算通信能耗"""

    bits_sent = np.asarray(bits_sent)
    bits_received = np.asarray(bits_received)
    distance = np.asarray(distance)
    
    if link_type == 'fso':
        energy = (bits_sent * (distance ** 2) * ENERGY_PER_BIT_FSO +
                  (bits_sent + bits_received) * FSO_ELECTRONICS_ENERGY_PER_BIT +
                  FSO_POINTING_ENERGY)
    else:  # rf
        energy = (bits_sent * (distance ** 2) * ENERGY_PER_BIT_RF +
                  (bits_sent + bits_received) * RF_ELECTRONICS_ENERGY_PER_BIT)
    
    return float(energy) if np.isscalar(energy) or energy.shape == () else energy
