import random
from tqdm import tqdm

from BASE.node import Node
from Utils.energy_model import calculate_mobile_energy, calculate_communication_energy_energy
from Utils.mac_layer import calculate_mac_delay
from Utils.physics_layer import calculate_rf_packet_error_rate
from Utils.utils import vector_projection, load_all_positions, calculate_distance_matrix, get_distance_from_matrix

COM_DISTANCE_DIRECTIONAL = 500


def initialize_nodes(node_count, initial_energy, positions):
    """从预加载的位置数据初始化节点列表"""
    nodes = []
    for i in range(node_count):
        position = positions[0, i, :].tolist()
        nodes.append(Node(i + 1, node_count, position, initial_energy))
    return nodes


def update_node_states(nodes, current_time, positions):
    """更新所有节点的位置、速度和能耗状态"""
    node_count = len(nodes)
    
    for i in range(node_count):
        if not nodes[i].alive:
            continue
        
        nodes[i].position_before = nodes[i].position
        nodes[i].position = positions[current_time, i, :].tolist()
        
        velocity = [a - b for a, b in zip(nodes[i].position, nodes[i].position_before)]
        mobile_energy = calculate_mobile_energy(
            velocity, nodes[i].speed, nodes[i].position_before[2], nodes[i].position[2]
        )
        
        nodes[i].energy_rate -= mobile_energy
        nodes[i].mobile_energy += mobile_energy
        nodes[i].speed = velocity
        
        if nodes[i].energy_rate < 0:
            nodes[i].alive = False
    
    return nodes


def broadcast_hello_messages(nodes, distance_matrix):
    """广播Hello消息以发现邻居"""
    hello_message_size = 4 * 3
    
    for i in range(len(nodes)):
        if not nodes[i].alive:
            continue
        
        nodes[i].cost_hello += hello_message_size
        communication_energy = calculate_communication_energy_energy(hello_message_size * 8, 0, COM_DISTANCE_DIRECTIONAL, 'rf')
        nodes[i].energy_rate -= communication_energy
        nodes[i].communication_energy += communication_energy
        
        if nodes[i].energy_rate <= 0:
            nodes[i].alive = False
            continue
        
        for j in range(len(nodes)):
            if i != j and distance_matrix[i, j] < COM_DISTANCE_DIRECTIONAL and nodes[j].alive:
                forward_hello_message(nodes[i], nodes[i], nodes[j], nodes, hello_message_size, distance_matrix)
        
        nodes[i].hello_sequence += 1
    
    return nodes


def forward_hello_message(source_node, sender_node, receiver_node, all_nodes, message_size, distance_matrix):
    """递归转发Hello消息"""
    if (source_node.node_id == receiver_node.node_id or not receiver_node.alive or
            source_node.hello_sequence <= receiver_node.hello_receive_sequence[source_node.node_id]):
        return
    
    relative_velocity = [a - b for a, b in zip(receiver_node.speed, sender_node.speed)]
    distance_vector = [a - b for a, b in zip(receiver_node.position, sender_node.position)]
    velocity_parallel = vector_projection(relative_velocity, distance_vector)
    
    distance = get_distance_from_matrix(distance_matrix, receiver_node.node_id, sender_node.node_id)
    packet_error_rate, _ = calculate_rf_packet_error_rate(
        distance, message_size * 8, velocity_parallel
    )
    
    if random.random() <= packet_error_rate:
        return
    
    receiver_node.receive_hello_message(source_node, message_size)
    receiver_node.hello_receive_sequence[source_node.node_id] = source_node.hello_sequence
    
    if not receiver_node.alive:
        return
    
    receiver_node.cost_hello += message_size
    communication_energy = calculate_communication_energy_energy(message_size * 8, 0, COM_DISTANCE_DIRECTIONAL, 'rf')
    receiver_node.energy_rate -= communication_energy
    receiver_node.communication_energy += communication_energy
    
    if receiver_node.energy_rate <= 0:
        receiver_node.alive = False
        return
    
    for next_node in all_nodes:
        if (next_node.node_id != receiver_node.node_id and 
                get_distance_from_matrix(distance_matrix, receiver_node.node_id, next_node.node_id) < COM_DISTANCE_DIRECTIONAL and next_node.alive):
            forward_hello_message(source_node, receiver_node, next_node, all_nodes, message_size, distance_matrix)


def update_routing_tables(nodes, distance_matrix):
    """更新所有节点的路由表"""
    for node in nodes:
        if node.alive:
            node.update_route(distance_matrix)
    return nodes


def transmit_service_message(sender, receiver, accumulated_delay, accumulated_cost, message_size, 
                             all_nodes, route_path, max_velocity, distance_matrix):
    """递归转发业务消息"""
    if sender.route_table[receiver.node_id] == float('inf'):
        return 0, False, False
    
    if receiver.node_id == sender.node_id:
        return accumulated_delay, accumulated_cost, route_path
    
    next_hop_id = sender.route_table[receiver.node_id]
    if not all_nodes[next_hop_id - 1].alive:
        return -4, False, False
    
    if next_hop_id in route_path:
        return -3, False, False
    
    next_hop = all_nodes[next_hop_id - 1]
    distance = get_distance_from_matrix(distance_matrix, sender.node_id, next_hop.node_id)
    
    sender.cost_message += message_size
    accumulated_cost += message_size
    communication_energy = calculate_communication_energy_energy(message_size * 8, 0, distance, 'fso')
    sender.energy_rate -= communication_energy
    sender.communication_energy += communication_energy
    
    if sender.energy_rate <= 0:
        sender.alive = False
    
    delay = calculate_mac_delay(next_hop, sender, message_size * 8, 'hc_base')
    
    if delay <= 0:
        return delay, False, False
    
    accumulated_delay += delay
    route_path.append(next_hop.node_id)
    
    communication_energy = calculate_communication_energy_energy(0, message_size * 8, distance, 'fso')
    next_hop.energy_rate -= communication_energy
    next_hop.communication_energy += communication_energy
    
    if next_hop.energy_rate <= 0:
        next_hop.alive = False
        return -4, False, False
    
    return transmit_service_message(next_hop, receiver, accumulated_delay, accumulated_cost, 
                                   message_size, all_nodes, route_path, max_velocity, distance_matrix)


def run_base_protocol(node_count, simulation_steps, max_velocity, senders, receivers, send_times,
                     packet_sizes, initial_energy, position_file_path):
    """运行BASE协议仿真"""
    alive_history = []
    success_counts = []
    delay_history = []
    failed_stats = [0] * 5
    all_route_lengths = []
    
    # 一次性加载所有位置数据
    positions = load_all_positions(position_file_path, node_count, simulation_steps)
    
    # 初始化节点
    nodes = initialize_nodes(node_count, initial_energy, positions)
    
    # 预计算所有时间步的距离矩阵
    distance_matrices = []
    for t in range(simulation_steps + 1):
        dist_matrix = calculate_distance_matrix(positions[t])
        distance_matrices.append(dist_matrix)
    
    for current_time in tqdm(range(1, simulation_steps + 1), desc="BASE\t"):
        # 使用预计算的距离矩阵
        current_distance_matrix = distance_matrices[current_time]
        
        nodes = update_node_states(nodes, current_time, positions)
        nodes = broadcast_hello_messages(nodes, current_distance_matrix)
        nodes = update_routing_tables(nodes, current_distance_matrix)
        
        # 处理业务消息
        for i in range(len(senders)):
            if not (send_times[i] <= current_time <= send_times[i] + 9):
                continue
            
            sender_id, receiver_id = senders[i], receivers[i]
            
            if current_time == send_times[i]:
                success_counts.append(0)
                delay_history.append(0)
                
                # 计算路由长度
                current_id, target_id = sender_id, receiver_id
                current_route = []
                while (current_id != target_id and
                       nodes[current_id - 1].route_table[target_id] not in current_route and
                       nodes[current_id - 1].route_table[target_id] != float('inf')):
                    current_route.append(nodes[current_id - 1].route_table[target_id])
                    current_id = nodes[current_id - 1].route_table[target_id]
                
                if current_id == target_id and current_route:
                    all_route_lengths.append(len(current_route))
            
            packet_length = 128 * packet_sizes[i]
            delay, _, length = transmit_service_message(
                nodes[sender_id - 1], nodes[receiver_id - 1], 0, 0,
                packet_length, nodes, [sender_id], max_velocity, current_distance_matrix
            )
            
            if delay > 0:
                success_counts[i] += 1
                delay_history[i] += delay / len(length)
            else:
                failed_stats[-delay] += 1
        
        # 统计存活节点
        alive_count = sum(1 for node in nodes if node.alive)
        alive_history.append(alive_count)
        
        # 存在节点死亡，立即结束
        if alive_count < node_count:
            break
    
    # 计算平均延迟（所有成功消息的总时延 / 成功消息总数）
    total_success = sum(success_counts)
    avg_delay = sum(delay_history) / total_success if total_success > 0 else 0
    
    # 计算失败原因比例
    total_failures = sum(failed_stats)
    failure_percentages = [round(f / total_failures * 100, 2) if total_failures > 0 else 0 for f in failed_stats]
    
    # 计算控制开销
    control_overhead = sum(node.cost_hello for node in nodes) / (simulation_steps * node_count)
    
    return (alive_history, success_counts, avg_delay,
            control_overhead, sum(node.communication_energy for node in nodes),
            sum(node.mobile_energy for node in nodes), failure_percentages, 0, all_route_lengths)
