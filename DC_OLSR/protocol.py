import random

import numpy as np
from tqdm import tqdm

from DC_OLSR.node import Node
from Utils.energy_model import calculate_mobile_energy, calculate_communication_energy_energy
from Utils.mac_layer import calculate_mac_delay
from Utils.physics_layer import calculate_fso_packet_error_rate
from Utils.utils import vector_projection, load_all_positions, calculate_distance_matrix, get_distance_from_matrix


HELLO_INTERVAL = 2
TC_INTERVAL = 5
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
        nodes[i].update_neighbors()
        
        if nodes[i].energy_rate < 0:
            nodes[i].alive = False
    
    return nodes


def broadcast_hello_messages(nodes, distance_matrix):
    """广播Hello消息以发现邻居（FSO定向通信）"""
    for i in range(len(nodes)):
        if not nodes[i].alive:
            continue
        
        hello_message_size = 4 * (8 + len(nodes[i].one_hop_symmetric_neighbors) + 
                                  len(nodes[i].one_hop_asymmetric_neighbors))
        
        for j in range(len(nodes)):
            if i != j and distance_matrix[i, j] < COM_DISTANCE_DIRECTIONAL and nodes[j].alive:
                nodes[i].cost_hello += hello_message_size
                
                communication_energy = calculate_communication_energy_energy(
                    hello_message_size * 8, 0, COM_DISTANCE_DIRECTIONAL, 'fso'
                )
                nodes[i].energy_rate -= communication_energy
                nodes[i].communication_energy += communication_energy
                
                if nodes[i].energy_rate <= 0:
                    nodes[i].alive = False
                    continue
                
                relative_velocity = [a - b for a, b in zip(nodes[j].speed, nodes[i].speed)]
                distance_vector = [a - b for a, b in zip(nodes[j].position, nodes[i].position)]
                velocity_parallel = vector_projection(relative_velocity, distance_vector)
                velocity_perpendicular = np.sqrt(max(np.linalg.norm(relative_velocity) ** 2 - velocity_parallel ** 2, 0))
                
                packet_error_rate, _ = calculate_fso_packet_error_rate(
                    distance_matrix[i, j], hello_message_size * 8, 
                    velocity_parallel, velocity_perpendicular
                )
                
                if random.random() > packet_error_rate:
                    nodes[j].receive_hello_message(nodes[i], hello_message_size)
    
    return nodes


def broadcast_tc_messages(nodes, max_velocity, distance_matrix):
    """MPR节点广播TC消息以传播拓扑信息"""
    for i in range(len(nodes)):
        if not nodes[i].alive or len(nodes[i].mpr_selector_set) == 0:
            continue
        
        tc_message_size = 4 * (5 + len(nodes[i].mpr_selector_set))
        nodes[i].tc_success.append(0)
        
        for j in range(len(nodes)):
            if i != j and distance_matrix[i, j] < COM_DISTANCE_DIRECTIONAL and nodes[j].alive:
                nodes[i].cost_tc += tc_message_size
                
                communication_energy = calculate_communication_energy_energy(
                    tc_message_size * 8, 0, COM_DISTANCE_DIRECTIONAL, 'fso'
                )
                nodes[i].energy_rate -= communication_energy
                nodes[i].communication_energy += communication_energy
                
                if nodes[i].energy_rate <= 0:
                    nodes[i].alive = False
                    continue
                
                forward_tc_message(nodes[i], nodes[i], nodes[j], nodes, tc_message_size, max_velocity, distance_matrix)
        
        nodes[i].tc_sequence += 1
    
    return nodes


def forward_tc_message(source_node, sender_node, receiver_node, all_nodes, message_size, max_velocity, distance_matrix):
    """MPR转发TC消息"""
    if (source_node.tc_sequence <= receiver_node.tc_receive_sequence[source_node.node_id] or
            source_node.node_id == receiver_node.node_id or not receiver_node.alive):
        return
    
    relative_velocity = [a - b for a, b in zip(receiver_node.speed, sender_node.speed)]
    distance_vector = [a - b for a, b in zip(receiver_node.position, sender_node.position)]
    velocity_parallel = vector_projection(relative_velocity, distance_vector)
    velocity_perpendicular = np.sqrt(max(np.linalg.norm(relative_velocity) ** 2 - velocity_parallel ** 2, 0))
    
    distance = get_distance_from_matrix(distance_matrix, receiver_node.node_id, sender_node.node_id)
    packet_error_rate, _ = calculate_fso_packet_error_rate(
        distance, message_size * 8, 
        velocity_parallel, velocity_perpendicular
    )
    
    if random.random() <= packet_error_rate:
        return
    
    receiver_node.receive_tc_message(source_node, message_size)
    receiver_node.tc_receive_sequence[source_node.node_id] = source_node.tc_sequence
    
    if receiver_node.node_id not in sender_node.mpr_set or not receiver_node.alive:
        source_node.tc_success[-1] += 1
        return
    
    receiver_node.cost_tc += message_size
    communication_energy = calculate_communication_energy_energy(
        message_size * 8, 0, COM_DISTANCE_DIRECTIONAL, 'fso'
    )
    receiver_node.energy_rate -= communication_energy
    receiver_node.communication_energy += communication_energy
    
    if receiver_node.energy_rate <= 0:
        receiver_node.alive = False
        return
    
    for next_node in all_nodes:
        if (next_node.node_id != receiver_node.node_id and 
                get_distance_from_matrix(distance_matrix, receiver_node.node_id, next_node.node_id) < COM_DISTANCE_DIRECTIONAL and next_node.alive):
            forward_tc_message(source_node, receiver_node, next_node, all_nodes, message_size, max_velocity, distance_matrix)
    
    source_node.tc_success[-1] += 1


def update_routing_tables(nodes):
    """更新所有节点的路由表"""
    for node in nodes:
        if node.alive and node.topology_changed:
            node.update_route()
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
    
    delay = calculate_mac_delay(next_hop, sender, message_size * 8, 'd_olsr')
    
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


def run_dc_olsr_protocol(node_count, simulation_steps, max_velocity, senders, receivers, send_times,
                        packet_sizes, initial_energy, position_file_path):
    """运行DC_OLSR协议仿真（定向OLSR）"""
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
    
    for current_time in tqdm(range(1, simulation_steps + 1), desc="DC_OLSR\t"):
        # 使用预计算的距离矩阵
        current_distance_matrix = distance_matrices[current_time]
        
        nodes = update_node_states(nodes, current_time, positions)
        
        if current_time % HELLO_INTERVAL == 1:
            nodes = broadcast_hello_messages(nodes, current_distance_matrix)
        if current_time % TC_INTERVAL == 1:
            nodes = broadcast_tc_messages(nodes, max_velocity, current_distance_matrix)
        
        nodes = update_routing_tables(nodes)
        
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
        
        if alive_count < node_count:
            break
    
    # 计算控制开销
    control_overhead = sum(node.cost_hello for node in nodes) + sum(node.cost_tc for node in nodes)
    
    # 计算平均延迟（所有成功消息的总时延 / 成功消息总数）
    total_success = sum(success_counts)
    avg_delay = sum(delay_history) / total_success if total_success > 0 else 0
    
    # 计算失败原因比例
    total_failures = sum(failed_stats)
    failure_percentages = [round(f / total_failures * 100, 2) if total_failures > 0 else 0 for f in failed_stats]
    
    # 计算TC消息成功率
    tc_success_rate = np.average([np.average(node.tc_success) for node in nodes if node.tc_success]) / node_count
    
    return (alive_history, success_counts, avg_delay,
            control_overhead / (simulation_steps * node_count), sum(node.communication_energy for node in nodes),
            sum(node.mobile_energy for node in nodes), failure_percentages,
            tc_success_rate, all_route_lengths)
