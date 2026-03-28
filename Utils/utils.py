import heapq
import numpy as np

# RF链路稳定性计算常量
RF_FACTOR_VELOCITY = 0.05             # RF链路速度影响因子
RF_FACTOR_DISTANCE_NEAR = 0.5         # RF近距离因子
RF_FACTOR_DISTANCE_FAR = 20           # RF远距离因子
RF_DISTANCE_THRESHOLD_NEAR = 0.8         # RF近距离阈值

# FSO链路稳定性计算常量
FSO_FACTOR_VELOCITY_PARALLEL = 0.05      # FSO径向速度影响因子
FSO_FACTOR_VELOCITY_PERPENDICULAR = 15   # FSO切向速度影响因子
FSO_FACTOR_DISTANCE_NEAR = 0.5             # FSO近距离因子
FSO_FACTOR_DISTANCE_FAR = 20              # FSO远距离因子
FSO_FACTOR_VELOCITY_POINT = 0.05          # FSO距离影响指向误差因子
FSO_DISTANCE_THRESHOLD_NEAR = 0.9         # FSO近距离阈值

# 通信系统常量
MAX_VELOCITY = 40                     # 最大速度 (m/s)
COM_DISTANCE_OMNI = 200               # 全向通信最大距离 (m)
COM_DISTANCE_DIRECTIONAL = 500        # 定向通信最大距离 (m)

# 链路稳定性计算中间常量
STABILITY_NORMALIZATION_MIN = 0       # 稳定性最小值
STABILITY_NORMALIZATION_MAX = 1       # 稳定性最大值
STABILITY_OFFSET = 0.005              # 稳定性偏移量（避免0值）
STABILITY_SCALE = 0.99                # 稳定性缩放因子
VELOCITY_OFFSET = 0.1                 # 速度偏移量


def load_all_positions(position_file_path, node_count, simulation_steps):
    """一次性从文件中读取所有位置信息"""

    try:
        with open(position_file_path, 'r') as file:
            lines = file.readlines()
        
        # 验证数据完整性
        expected_lines = node_count * (simulation_steps + 1)
        if len(lines) < expected_lines:
            raise ValueError(f"位置文件数据不完整: 期望 {expected_lines} 行, 实际 {len(lines)} 行")
        
        # 预分配NumPy数组以提高性能
        positions = np.zeros((simulation_steps + 1, node_count, 3), dtype=np.float64)
        
        # 解析数据
        for t in range(simulation_steps + 1):
            for i in range(node_count):
                line_idx = t * node_count + i
                try:
                    coords = lines[line_idx].strip().split()
                    if len(coords) != 3:
                        raise ValueError(f"第 {line_idx + 1} 行数据格式错误: 期望3个坐标值")
                    positions[t, i, :] = [float(coord) for coord in coords]
                except (IndexError, ValueError) as e:
                    raise ValueError(f"解析第 {line_idx + 1} 行时出错: {str(e)}")
        
        return positions
    
    except FileNotFoundError:
        raise FileNotFoundError(f"位置文件不存在: {position_file_path}")
    except Exception as e:
        raise RuntimeError(f"读取位置文件时发生错误: {str(e)}")


def calculate_distance_matrix(positions_t):
    """计算给定时间步所有节点之间的两两距离矩阵"""

    diff = positions_t[:, np.newaxis, :] - positions_t[np.newaxis, :, :]
    distance_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    return distance_matrix


def get_distance_from_matrix(distance_matrix, node_id_a, node_id_b):
    """从距离矩阵中获取两个节点之间的距离"""
    return distance_matrix[node_id_a - 1, node_id_b - 1]


def dijkstra(cost_matrix, source_id):
    """使用Dijkstra算法计算从源节点到所有其他节点的最短路径，返回下一跳路由表"""

    source_index = source_id - 1
    node_count = len(cost_matrix)
    distances = [float('inf')] * node_count
    next_hops = {i + 1: float('inf') for i in range(node_count)}
    next_hops[source_id] = source_id
    distances[source_index] = 0
    
    priority_queue = [(0, source_index)]
    
    while priority_queue:
        current_dist, current_node = heapq.heappop(priority_queue)
        if current_dist > distances[current_node]:
            continue
        
        for neighbor in range(node_count):
            if cost_matrix[current_node][neighbor] != float('inf'):
                new_distance = current_dist + cost_matrix[current_node][neighbor]
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    if current_node == source_index:
                        next_hops[neighbor + 1] = neighbor + 1
                    else:
                        next_hops[neighbor + 1] = next_hops[current_node + 1]
                    heapq.heappush(priority_queue, (new_distance, neighbor))
    
    return next_hops


def calculate_angle_between_vectors(vector_a, vector_b):
    """计算两个向量之间的夹角（弧度）"""

    vector_a = np.asarray(vector_a)
    vector_b = np.asarray(vector_b)
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    cos_theta = np.clip(dot_product / (norm_a * norm_b), -1.0, 1.0)
    return abs(np.arccos(cos_theta))


def vector_projection(vector_a, vector_b):
    """计算向量A在向量B方向上的投影，保留方向信息"""

    vector_a = np.asarray(vector_a)
    vector_b = np.asarray(vector_b)
    dot_product = np.dot(vector_a, vector_b)
    norm_b_squared = np.dot(vector_b, vector_b)
    
    if norm_b_squared == 0:
        return 0.0
    
    projection = (dot_product / norm_b_squared) * vector_b
    direction = 1 if calculate_angle_between_vectors(vector_a, vector_b) <= np.pi / 2 else -1
    return direction * np.linalg.norm(projection)


def calculate_rf_link_stability(node1, node2, max_velocity):
    """计算RF链路稳定性，考虑距离和相对速度的影响"""

    pos1 = np.asarray(node1.position)
    pos2 = np.asarray(node2.position)
    pos1_pred = np.asarray(node1.position_predict)
    pos2_pred = np.asarray(node2.position_predict)

    distance_vector = pos1 - pos2
    predicted_distance_vector = pos1_pred - pos2_pred
    velocity_vector = predicted_distance_vector - distance_vector

    velocity_parallel = vector_projection(velocity_vector, distance_vector) / MAX_VELOCITY
    normalized_distance = np.linalg.norm(predicted_distance_vector) / COM_DISTANCE_OMNI

    distance_factor = RF_FACTOR_DISTANCE_NEAR if normalized_distance < RF_DISTANCE_THRESHOLD_NEAR else RF_FACTOR_DISTANCE_FAR

    if max_velocity > 0:
        stability = (np.exp(-RF_FACTOR_VELOCITY * normalized_distance * (velocity_parallel + VELOCITY_OFFSET) ** 2) *
                     np.exp(-(distance_factor + velocity_parallel / 10) * (normalized_distance ** 2)))
    else:
        stability = np.exp(-distance_factor * (normalized_distance ** 2))

    return min(max(stability, STABILITY_NORMALIZATION_MIN), STABILITY_NORMALIZATION_MAX) * STABILITY_SCALE + STABILITY_OFFSET


def calculate_fso_link_stability(node1, node2, max_velocity):
    """计算FSO链路稳定性，考虑距离、径向速度和切向速度的影响"""

    pos1 = np.asarray(node1.position)
    pos2 = np.asarray(node2.position)
    pos1_pred = np.asarray(node1.position_predict)
    pos2_pred = np.asarray(node2.position_predict)
    
    distance_vector = pos1 - pos2
    predicted_distance_vector = pos1_pred - pos2_pred
    velocity_vector = predicted_distance_vector - distance_vector

    distance_norm = np.linalg.norm(predicted_distance_vector) + 0.01
    velocity_parallel = vector_projection(velocity_vector, distance_vector)
    velocity_perpendicular = np.sqrt(max(np.linalg.norm(velocity_vector) ** 2 - velocity_parallel ** 2, 0))

    normalized_distance = distance_norm / COM_DISTANCE_DIRECTIONAL
    normalized_velocity_parallel = velocity_parallel / MAX_VELOCITY
    normalized_velocity_perpendicular = velocity_perpendicular / MAX_VELOCITY

    distance_factor = FSO_FACTOR_DISTANCE_NEAR if normalized_distance < FSO_DISTANCE_THRESHOLD_NEAR else FSO_FACTOR_DISTANCE_FAR

    if max_velocity > 0:
        stability = (np.exp(-FSO_FACTOR_VELOCITY_PARALLEL * abs(normalized_velocity_parallel)) *
                     np.exp(-distance_factor * normalized_distance) *
                     np.exp(FSO_FACTOR_VELOCITY_PERPENDICULAR * np.log2(normalized_distance ** FSO_FACTOR_VELOCITY_POINT) *
                            normalized_velocity_perpendicular))
    else:
        stability = np.exp(-distance_factor * normalized_distance)

    return min(max(stability, STABILITY_NORMALIZATION_MIN), STABILITY_NORMALIZATION_MAX) * STABILITY_SCALE + STABILITY_OFFSET
