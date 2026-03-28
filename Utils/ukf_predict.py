import numpy as np
from tqdm import tqdm


def compute_unscented_transform(state_mean, state_covariance, kappa):
    """计算无迹变换（Unscented Transform）"""

    state_dim = state_mean.size
    lambda_ = kappa - state_dim
    num_sigma_points = 2 * state_dim + 1
    
    sigma_points = np.zeros((num_sigma_points, state_dim))
    weights_mean = np.zeros(num_sigma_points)
    weights_covariance = np.zeros(num_sigma_points)
    
    # 中心点
    sigma_points[0] = state_mean
    weights_mean[0] = lambda_ / (state_dim + lambda_)
    weights_covariance[0] = lambda_ / (state_dim + lambda_)
    
    # 计算矩阵平方根
    covariance_scaled = (state_dim + lambda_) * state_covariance + np.eye(state_dim) * 1e-10
    matrix_sqrt = np.linalg.cholesky(covariance_scaled)
    
    # 生成对称的Sigma点
    weight = 1 / (2 * (state_dim + lambda_))
    for i in range(state_dim):
        sigma_points[i + 1] = state_mean + matrix_sqrt[i]
        sigma_points[state_dim + i + 1] = state_mean - matrix_sqrt[i]
        weights_mean[i + 1] = weight
        weights_covariance[i + 1] = weight
        weights_mean[state_dim + i + 1] = weight
        weights_covariance[state_dim + i + 1] = weight
    
    return sigma_points, weights_mean, weights_covariance


def predict_state(sigma_points, weights_mean, weights_covariance, process_function, process_noise_covariance):
    """预测步骤：通过过程模型传播Sigma点"""

    state_dim = sigma_points.shape[1]
    num_sigma_points = sigma_points.shape[0]
    
    # 传播Sigma点通过过程模型
    predicted_sigma_points = np.array([process_function(point) for point in sigma_points])
    
    # 计算预测状态的均值
    predicted_state = np.dot(weights_mean, predicted_sigma_points)
    
    # 计算预测状态的协方差
    predicted_covariance = np.zeros((state_dim, state_dim))
    for i in range(num_sigma_points):
        state_diff = predicted_sigma_points[i] - predicted_state
        predicted_covariance += weights_covariance[i] * np.outer(state_diff, state_diff)
    predicted_covariance += process_noise_covariance
    
    return predicted_state, predicted_covariance, predicted_sigma_points


def update_state(predicted_sigma_points, weights_mean, weights_covariance, 
                 measurement, measurement_function, measurement_noise_covariance,
                 predicted_state, predicted_covariance):
    """
    更新步骤：融合测量值修正预测状态
    """
    state_dim = predicted_sigma_points.shape[1]
    measurement_dim = measurement.size
    num_sigma_points = predicted_sigma_points.shape[0]
    
    # 传播Sigma点通过测量模型
    measurement_sigma_points = np.array([measurement_function(point) for point in predicted_sigma_points])
    
    # 计算预测的测量值
    predicted_measurement = np.dot(weights_mean, measurement_sigma_points)
    
    # 计算测量协方差和互协方差
    measurement_covariance = np.zeros((measurement_dim, measurement_dim))
    cross_covariance = np.zeros((state_dim, measurement_dim))
    
    for i in range(num_sigma_points):
        measurement_diff = measurement_sigma_points[i] - predicted_measurement
        state_diff = predicted_sigma_points[i] - predicted_state
        measurement_covariance += weights_covariance[i] * np.outer(measurement_diff, measurement_diff)
        cross_covariance += weights_covariance[i] * np.outer(state_diff, measurement_diff)
    
    measurement_covariance += measurement_noise_covariance
    
    # 计算卡尔曼增益
    kalman_gain = np.dot(cross_covariance, np.linalg.inv(measurement_covariance))
    
    # 更新状态估计
    innovation = measurement - predicted_measurement
    updated_state = predicted_state + np.dot(kalman_gain, innovation)
    updated_covariance = predicted_covariance - np.dot(kalman_gain, 
                                                       np.dot(measurement_covariance, kalman_gain.T))
    
    return updated_state, updated_covariance


def constant_velocity_motion_model(state, time_step=1.0):
    """
    恒定速度运动模型
    """
    transition_matrix = np.array([
        [1, 0, 0, time_step, 0, 0],
        [0, 1, 0, 0, time_step, 0],
        [0, 0, 1, 0, 0, time_step],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])
    return transition_matrix @ state


def position_measurement_model(state):
    """
    位置测量模型：从状态向量中提取位置
    """
    return state[:3]


def perform_ukf_prediction(measurements, measurement_interval):
    """执行无迹卡尔曼滤波预测"""

    # 噪声参数
    process_noise_std = 5.0
    measurement_noise_std = 0.5
    kappa = 3  # 无迹变换缩放参数
    
    time_steps, num_drones, _ = measurements.shape
    predicted_states = np.zeros((time_steps, num_drones, 3))
    
    for drone_id in tqdm(range(num_drones), desc = "UKF predict\t"):
        # 初始化状态 [位置, 速度]
        initial_position = measurements[0, drone_id]
        initial_velocity = [0, 0, 0]
        state = np.hstack((initial_position, initial_velocity))
        
        # 初始化协方差
        state_covariance = np.eye(6)
        
        process_noise_covariance = np.eye(6) * measurement_interval * process_noise_std
        measurement_noise_covariance = np.eye(3) * measurement_noise_std
        
        for time_step in range(time_steps):
            # 无迹变换
            sigma_points, weights_mean, weights_covariance = compute_unscented_transform(
                state, state_covariance, kappa
            )
            
            # 预测步骤
            predicted_state, predicted_covariance, predicted_sigma_points = predict_state(
                sigma_points, weights_mean, weights_covariance,
                constant_velocity_motion_model, process_noise_covariance
            )
            
            # 更新步骤（仅在测量时刻）
            if (time_step % measurement_interval) == 0:
                current_measurement = measurements[time_step, drone_id]
                updated_state, updated_covariance = update_state(
                    predicted_sigma_points, weights_mean, weights_covariance,
                    current_measurement, position_measurement_model, measurement_noise_covariance,
                    predicted_state, predicted_covariance
                )
                state, state_covariance = updated_state, updated_covariance
            else:
                state, state_covariance = predicted_state, predicted_covariance
            
            # 保存预测的位置
            predicted_states[time_step, drone_id] = state[:3]
    
    return predicted_states
