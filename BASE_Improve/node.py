#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from Utils.energy_model import calculate_communication_energy_energy
from Utils.utils import dijkstra

COM_DISTANCE_DIRECTIONAL = 500


class Node:
    """I_BASE协议节点类 - 带位置预测的改进基础广播路由"""
    
    def __init__(self, node_id, node_count, position, initial_energy):
        self.node_id = node_id
        self.position = position
        self.position_before = [0, 0, 0]
        self.position_predict = [0, 0, 0]
        self.initial_energy = initial_energy
        self.energy_rate = initial_energy
        
        # 能耗统计
        self.cost_hello = 0
        self.cost_message = 0
        self.communication_energy = 0
        self.mobile_energy = 0
        
        # 状态管理
        self.alive = True
        self.speed = [0, 0, 0]
        
        # Hello消息序列管理
        self.hello_sequence = 1
        self.hello_receive_sequence = {i: 0 for i in range(1, node_count + 1)}
        
        # 网络拓扑信息
        self.position_table = {self.node_id: self.position}
        self.position_prediction_table = {self.node_id: self.position_predict}
        self.topology_table = [[float('inf')] * node_count for _ in range(node_count)]
        for i in range(node_count):
            self.topology_table[i][i] = 0
        self.route_table = {}

    def receive_hello_message(self, source_node, message_size):
        """接收Hello消息，更新位置表和预测位置表"""
        self.communication_energy += calculate_communication_energy_energy(0, message_size, 0, 'rf')
        self.energy_rate -= calculate_communication_energy_energy(0, message_size, 0, 'rf')
        self.position_table[source_node.node_id] = source_node.position
        self.position_prediction_table[source_node.node_id] = source_node.position_predict

    def update_route(self, distance_matrix, prediction_matrix):
        """基于距离矩阵和预测矩阵更新路由表，使用预测位置验证链路有效性"""
        self.position_table[self.node_id] = self.position
        known_nodes = list(self.position_table.keys())
        
        for i, node_a in enumerate(known_nodes):
            for node_b in known_nodes[i + 1:]:
                current_distance = distance_matrix[node_a - 1, node_b - 1]
                
                if current_distance <= COM_DISTANCE_DIRECTIONAL:
                    predicted_distance = prediction_matrix[node_a - 1, node_b - 1]
                    
                    if predicted_distance <= COM_DISTANCE_DIRECTIONAL:
                        self.topology_table[node_a - 1][node_b - 1] = 1
                        self.topology_table[node_b - 1][node_a - 1] = 1
                    else:
                        self.topology_table[node_a - 1][node_b - 1] = float('inf')
                        self.topology_table[node_b - 1][node_a - 1] = float('inf')
                else:
                    self.topology_table[node_a - 1][node_b - 1] = float('inf')
                    self.topology_table[node_b - 1][node_a - 1] = float('inf')
        
        self.route_table = dijkstra(self.topology_table, self.node_id)
