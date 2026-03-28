import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from Utils.ukf_predict import perform_ukf_prediction

matplotlib.use('TkAgg')

# 默认参数
DEFAULT_ALTITUDE = 300
MEASUREMENT_NOISE_STD = 0.25
SAFE_DISTANCE = 75


def initialize_drone_positions(num_drones, initial_altitude, drone_spacing):
    """初始化无人机位置，按网格排列"""
    grid_rows = int(np.floor(np.sqrt(num_drones)))
    grid_cols = num_drones // grid_rows
    remainder = num_drones % grid_rows
    
    positions = np.zeros((num_drones, 3))
    drone_index = 0
    
    # 网格布局
    for row in range(grid_rows):
        for col in range(grid_cols):
            positions[drone_index] = [col * drone_spacing, row * drone_spacing, initial_altitude]
            drone_index += 1
    
    # 剩余无人机
    for col in range(remainder):
        positions[drone_index] = [col * drone_spacing, grid_rows * drone_spacing, initial_altitude]
        drone_index += 1
    
    return positions


def apply_collision_avoidance(positions, velocities, safe_distance):
    """应用避碰机制，调整速度向量"""
    num_drones = len(positions)
    adjusted_velocities = velocities.copy()
    
    for i in range(num_drones):
        for j in range(num_drones):
            if i != j:
                distance_vector = positions[i] - positions[j]
                distance = np.linalg.norm(distance_vector)
                
                if safe_distance > distance > 0:
                    repulsion = distance_vector / distance
                    repulsion_strength = 0.5 * (safe_distance - distance) / safe_distance
                    adjusted_velocities[i] += repulsion * repulsion_strength
                    adjusted_velocities[j] -= repulsion * repulsion_strength
    
    return adjusted_velocities


def limit_velocity(velocity, max_speed):
    """限制速度不超过最大值"""
    speed = np.linalg.norm(velocity)
    if speed > max_speed and speed > 0:
        return (velocity / speed) * max_speed
    return velocity


def simulate_random_waypoint_motion(num_drones, time_steps, max_velocity):
    """随机路点移动模型 (Random Waypoint Model)"""

    initial_altitude = DEFAULT_ALTITUDE
    drone_spacing = 100
    safe_distance = SAFE_DISTANCE
    
    grid_rows = int(np.floor(np.sqrt(num_drones)))
    grid_cols = num_drones // grid_rows
    
    max_x = grid_rows * drone_spacing * 5
    max_y = (grid_cols + 1) * drone_spacing * 5
    max_z = initial_altitude * 2
    
    # 初始化状态
    true_positions = np.zeros((time_steps, num_drones, 3))
    true_velocities = np.zeros((time_steps, num_drones, 3))
    measurements = np.zeros((time_steps, num_drones, 3))
    
    true_positions[0] = initialize_drone_positions(num_drones, initial_altitude, drone_spacing)
    measurements[0] = true_positions[0] + np.random.normal(0, MEASUREMENT_NOISE_STD, (num_drones, 3))
    
    # 初始化路点
    waypoints = np.random.uniform([0, 0, 0], [max_x, max_y, max_z], size=(num_drones, 3))
    
    for t in tqdm(range(1, time_steps), desc="RWP Motion"):
        for i in range(num_drones):
            # 计算到路点的方向
            direction = waypoints[i] - true_positions[t - 1, i]
            distance = np.linalg.norm(direction)
            
            # 到达路点，生成新路点
            if distance < 1.0:
                waypoints[i] = np.random.uniform([0, 0, 0], [max_x, max_y, max_z])
                direction = waypoints[i] - true_positions[t - 1, i]
                distance = np.linalg.norm(direction)
            
            # 计算期望速度
            if distance > 0:
                desired_velocity = (direction / distance) * max_velocity
            else:
                desired_velocity = np.zeros(3)
            
            true_velocities[t, i] = desired_velocity
        
        # 避碰
        true_velocities[t] = apply_collision_avoidance(true_positions[t - 1], true_velocities[t], safe_distance)
        
        # 限速
        for i in range(num_drones):
            true_velocities[t, i] = limit_velocity(true_velocities[t, i], max_velocity)
        
        # 更新位置
        true_positions[t] = true_positions[t - 1] + true_velocities[t]
        
        # 边界限制
        true_positions[t, :, 0] = np.clip(true_positions[t, :, 0], 0, max_x)
        true_positions[t, :, 1] = np.clip(true_positions[t, :, 1], 0, max_y)
        true_positions[t, :, 2] = np.clip(true_positions[t, :, 2], 0, max_z)
        
        # 添加测量噪声
        measurements[t] = true_positions[t] + np.random.normal(0, MEASUREMENT_NOISE_STD, (num_drones, 3))
    
    return measurements


def simulate_gauss_markov_motion(num_drones, time_steps, max_velocity):
    """
    高斯-马尔可夫移动模型 (Gauss-Markov Model)
    
    速度具有时间相关性，适合模拟真实无人机运动
    """
    initial_altitude = DEFAULT_ALTITUDE
    drone_spacing = 150
    safe_distance = 50
    alpha = 0.75  # 惯性因子
    
    grid_rows = int(np.floor(np.sqrt(num_drones)))
    grid_cols = num_drones // grid_rows
    
    max_x = grid_rows * drone_spacing * 5
    max_y = (grid_cols + 1) * drone_spacing * 5
    max_z = initial_altitude * 2
    
    # 初始化状态
    true_positions = np.zeros((time_steps, num_drones, 3))
    true_velocities = np.zeros((time_steps, num_drones, 3))
    measurements = np.zeros((time_steps, num_drones, 3))
    
    true_positions[0] = initialize_drone_positions(num_drones, initial_altitude, drone_spacing)
    measurements[0] = true_positions[0] + np.random.normal(0, MEASUREMENT_NOISE_STD, (num_drones, 3))
    
    for t in tqdm(range(1, time_steps), desc="GM Motion"):
        # GM速度更新
        noise = np.random.normal(0, max_velocity, (num_drones, 3))
        true_velocities[t] = alpha * true_velocities[t - 1] + (1 - alpha) * noise
        
        # 避碰
        true_velocities[t] = apply_collision_avoidance(true_positions[t - 1], true_velocities[t], safe_distance)
        
        # 限速
        for i in range(num_drones):
            true_velocities[t, i] = limit_velocity(true_velocities[t, i], max_velocity)
        
        # 更新位置
        true_positions[t] = true_positions[t - 1] + true_velocities[t]
        
        # 边界限制
        true_positions[t, :, 0] = np.clip(true_positions[t, :, 0], 0, max_x)
        true_positions[t, :, 1] = np.clip(true_positions[t, :, 1], 0, max_y)
        true_positions[t, :, 2] = np.clip(true_positions[t, :, 2], 0, max_z)
        
        # 测量
        measurements[t] = true_positions[t] + np.random.normal(0, MEASUREMENT_NOISE_STD, (num_drones, 3))
    
    return measurements


def simulate_pursuit_motion(num_drones, time_steps, max_velocity):
    """追随移动模型 (Pursuit Mobility Model)"""

    initial_altitude = DEFAULT_ALTITUDE
    drone_spacing = 150
    safe_distance = 80
    
    grid_rows = int(np.floor(np.sqrt(num_drones)))
    grid_cols = num_drones // grid_rows
    
    max_x = grid_rows * drone_spacing * 100
    max_y = (grid_cols + 1) * drone_spacing * 100
    max_z = initial_altitude * 2
    
    # 初始化状态
    true_positions = np.zeros((time_steps, num_drones, 3))
    true_velocities = np.zeros((time_steps, num_drones, 3))
    measurements = np.zeros((time_steps, num_drones, 3))
    target_state = np.zeros((time_steps, 6))  # [x, y, z, vx, vy, vz]
    
    # 初始化目标位置
    target_state[0] = [grid_rows * drone_spacing / 2, grid_cols * drone_spacing / 2, initial_altitude, 0, 0, 0]
    
    true_positions[0] = initialize_drone_positions(num_drones, initial_altitude, drone_spacing)
    measurements[0] = true_positions[0] + np.random.normal(0, MEASUREMENT_NOISE_STD, (num_drones, 3))
    
    for t in tqdm(range(1, time_steps), desc="Pursuit Motion\t"):
        # 更新目标状态
        acceleration = np.random.normal(0, max_velocity / 2, 3)
        target_state[t, 3:] = np.clip(target_state[t - 1, 3:] + acceleration, -max_velocity, max_velocity)
        target_state[t, :3] = target_state[t - 1, :3] + target_state[t, 3:]
        target_state[t, :3] = np.clip(target_state[t, :3], [0, 0, 0], [max_x, max_y, max_z])
        
        # 无人机追随目标
        for i in range(num_drones):
            direction = (target_state[t, :3] - true_positions[t - 1, i]) * np.random.normal(1, 0.5, 3)
            distance = np.linalg.norm(direction)
            
            if distance > 0:
                speed = np.random.normal(distance / 400, max_velocity / 6, 3)
                true_velocities[t, i] = true_velocities[t - 1, i] + direction / distance * speed
            
            # 避碰
            for j in range(num_drones):
                if i != j:
                    distance_to_other = np.linalg.norm(true_positions[t - 1, i] - true_positions[t - 1, j])
                    if distance_to_other < safe_distance:
                        repulsion = (true_positions[t - 1, i] - true_positions[t - 1, j]) / distance_to_other
                        true_velocities[t, i] += repulsion * (safe_distance - distance_to_other) / safe_distance / 2
                        true_velocities[t, j] -= repulsion * (safe_distance - distance_to_other) / safe_distance / 2
            
            # 限速
            true_velocities[t, i] = np.clip(true_velocities[t, i], -max_velocity, max_velocity)
            
            # 更新位置
            true_positions[t, i] = true_positions[t - 1, i] + true_velocities[t, i]
            measurements[t, i] = true_positions[t, i] + np.random.normal(0, MEASUREMENT_NOISE_STD, 3)
    
    return measurements


def predict_with_mmn(positions):
    """
    使用MMN（简单线性预测）模型预测位置
    """
    time_steps, num_drones, _ = positions.shape
    predictions = np.zeros_like(positions)
    predictions[0] = np.zeros_like(positions[0])
    
    if time_steps < 2:
        return predictions
    
    predictions[1] = positions[0].copy()
    
    for t in range(2, time_steps):
        predictions[t] = positions[t - 1] + (positions[t - 1] - positions[t - 2])
    
    return predictions


def write_motion_data_to_files(num_drones, time_steps, max_velocity, true_path, predict_path):
    """ 生成运动数据并写入文件"""

    # 生成真实运动数据（使用追随模型）
    measurements = simulate_pursuit_motion(num_drones, time_steps + 1, max_velocity)
    
    # 写入真实位置
    with open(true_path, 'w') as f:
        for t in range(len(measurements)):
            for i in range(len(measurements[t])):
                position = measurements[t][i]
                f.write(f'{position[0]:.6f} {position[1]:.6f} {position[2]:.6f}\n')
    
    # UKF预测
    ukf_predictions = perform_ukf_prediction(measurements, 1)
    with open(predict_path, 'w') as f:
        for t in range(len(ukf_predictions)):
            for i in range(len(ukf_predictions[t])):
                position = ukf_predictions[t][i]
                f.write(f'{position[0]:.6f} {position[1]:.6f} {position[2]:.6f}\n')
    
    # MMN预测
    mmn_predictions = predict_with_mmn(measurements)
    with open(predict_path.replace('.txt', '_mmn.txt'), 'w') as f:
        for t in range(len(mmn_predictions)):
            for i in range(len(mmn_predictions[t])):
                position = mmn_predictions[t][i]
                f.write(f'{position[0]:.6f} {position[1]:.6f} {position[2]:.6f}\n')


def plot_drone_trajectories(num_drones, true_positions, target_positions=None):
    """绘制无人机运动轨迹"""

    fig = plt.figure(figsize=(10, 8), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制每个无人机的轨迹
    for i in range(num_drones):
        ax.plot(true_positions[:, i, 0], true_positions[:, i, 1], true_positions[:, i, 2], 
                linewidth=0.5, alpha=0.7)
    
    # 绘制目标轨迹（如果提供）
    if target_positions is not None:
        ax.plot(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2], 
                'r--', linewidth=2, label='Target')
    
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title('Drone Cluster Movement Trajectories', fontsize=14)
    
    # 设置等比例坐标轴
    max_range = np.array([
        true_positions[:, :, 0].max() - true_positions[:, :, 0].min(),
        true_positions[:, :, 1].max() - true_positions[:, :, 1].min(),
        true_positions[:, :, 2].max() - true_positions[:, :, 2].min()
    ]).max() / 2.0
    
    mid_x = (true_positions[:, :, 0].max() + true_positions[:, :, 0].min()) * 0.5
    mid_y = (true_positions[:, :, 1].max() + true_positions[:, :, 1].min()) * 0.5
    mid_z = (true_positions[:, :, 2].max() + true_positions[:, :, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.rcParams['axes.unicode_minus'] = False
    ax.legend()
    plt.tight_layout()
    plt.show()
