#!/usr/bin/env python
# -*- coding: utf-8 -*-

import heapq
import numpy as np
from Utils.energy_model import calculate_communication_energy_energy
from Utils.utils import dijkstra, calculate_rf_link_stability, calculate_fso_link_stability

# 超时时间常量
TIME_TOPOLOGY = 6
TIME_MPR_S_SET = 6
TIME_ASYMMETRIC_NEIGHBORS = 4
TIME_SYMMETRIC_NEIGHBORS = 6
TIME_DIRECTION_NEIGHBORS = 6
TIME_TWO_SYMMETRIC_NEIGHBORS = 6


class Node:
    """HCI_OLSR协议节点类 - 带位置预测的混合OLSR路由"""
    
    def __init__(self, node_id, node_count, position, initial_energy):
        self.node_id = node_id
        self.position = position
        self.position_before = [0, 0, 0]
        self.position_predict = [0, 0, 0]
        self.node_num = node_count
        self.energy_rate = initial_energy
        
        # 能耗统计
        self.cost_hello = 0
        self.cost_tc = 0
        self.cost_message = 0
        self.communication_energy = 0
        self.mobile_energy = 0
        
        # 状态管理
        self.alive = True
        self.speed = [0, 0, 0]
        self.topology_changed = False
        self.mpr_s_change = False
        
        # 消息序列管理
        self.tc_sequence = 1
        self.hello_sequence = 1
        self.hello_receive_sequence = {i: 0 for i in range(1, node_count + 1)}
        self.tc_receive_sequence = {i: 0 for i in range(1, node_count + 1)}
        
        # TC消息成功率统计
        self.tc_success = []
        
        # 邻居管理
        self.one_omnidirectional_asymmetry_neighbors = {}  # 一跳全向非对称邻居
        self.one_omnidirectional_symmetry_neighbors = {}   # 一跳全向对称邻居
        self.one_directional_asymmetry_neighbors = {}      # 一跳定向非对称邻居
        self.one_directional_symmetry_neighbors = {}       # 一跳定向对称邻居
        self.two_omnidirectional_symmetry_neighbors = {}   # 两跳对称邻居
        self.two_omnidirectional_symmetry_neighbors_time = {}  # 两跳邻居剩余时间
        
        # MPR管理
        self.mpr_nodes = []  # MPR集合
        self.mpr_s_nodes = {}  # MPR选择集
        
        # 拓扑和路由
        self.topology_table = [[float('inf')] * node_count for _ in range(node_count)]
        self.topology_table_time = [[0] * node_count for _ in range(node_count)]
        for i in range(node_count):
            self.topology_table[i][i] = 0
        self.route_table = {}

    def update_neighbors(self):
        """更新邻居状态，处理超时"""
        self.topology_changed = False
        self.mpr_s_change = False
        
        self._remove_asymmetry_neighbors()
        self._remove_symmetry_neighbors()
        self._remove_directional_asymmetry_neighbors()
        self._remove_directional_symmetry_neighbors()
        self._remove_two_symmetry_neighbors()
        self._remove_mpr_s()
        self._remove_topology_table()

    def _remove_asymmetry_neighbors(self):
        """移除超时的一跳全向非对称邻居"""
        to_remove = [node for node, time in self.one_omnidirectional_asymmetry_neighbors.items() if time <= 1]
        for node in to_remove:
            del self.one_omnidirectional_asymmetry_neighbors[node]
        
        for node in self.one_omnidirectional_asymmetry_neighbors:
            self.one_omnidirectional_asymmetry_neighbors[node] -= 1

    def _remove_symmetry_neighbors(self):
        """移除超时的一跳全向对称邻居，降级为非对称邻居"""
        to_remove = [node for node, time in self.one_omnidirectional_symmetry_neighbors.items() if time <= 1]
        
        for node in to_remove:
            del self.one_omnidirectional_symmetry_neighbors[node]
            self.topology_table_time[node - 1][self.node_id - 1] = 1
            self.topology_table_time[self.node_id - 1][node - 1] = 1
            self.one_omnidirectional_asymmetry_neighbors[node] = TIME_ASYMMETRIC_NEIGHBORS
            
            if node in self.mpr_nodes:
                self.mpr_nodes.remove(node)
            
            # 从两跳邻居中移除
            to_remove_2 = []
            for n_2, n_1_list in self.two_omnidirectional_symmetry_neighbors.items():
                if node in n_1_list:
                    to_remove_2.append(n_2)
            
            for node_2 in to_remove_2:
                self.topology_table_time[node - 1][node_2 - 1] = 1
                self.topology_table_time[node_2 - 1][node - 1] = 1
                index = self.two_omnidirectional_symmetry_neighbors[node_2].index(node)
                self.two_omnidirectional_symmetry_neighbors[node_2].remove(node)
                del self.two_omnidirectional_symmetry_neighbors_time[node_2][index]
                
                if not self.two_omnidirectional_symmetry_neighbors[node_2]:
                    del self.two_omnidirectional_symmetry_neighbors[node_2]
                    del self.two_omnidirectional_symmetry_neighbors_time[node_2]

    def _remove_directional_asymmetry_neighbors(self):
        """移除超时的一跳定向非对称邻居"""
        to_remove = [node for node, time in self.one_directional_asymmetry_neighbors.items() if time <= 1]
        for node in to_remove:
            del self.one_directional_asymmetry_neighbors[node]
        
        for node in self.one_directional_asymmetry_neighbors:
            self.one_directional_asymmetry_neighbors[node] -= 1

    def _remove_directional_symmetry_neighbors(self):
        """移除超时的一跳定向对称邻居，降级为非对称邻居"""
        to_remove = [node for node, time in self.one_directional_symmetry_neighbors.items() if time <= 1]
        
        for node in to_remove:
            del self.one_directional_symmetry_neighbors[node]
            self.topology_table_time[node - 1][self.node_id - 1] = 1
            self.topology_table_time[self.node_id - 1][node - 1] = 1
            self.one_directional_asymmetry_neighbors[node] = TIME_ASYMMETRIC_NEIGHBORS

    def _remove_two_symmetry_neighbors(self):
        """移除超时的两跳对称邻居"""
        to_remove = {}
        for node, time_list in self.two_omnidirectional_symmetry_neighbors_time.items():
            for one_hop_node in range(len(time_list)):
                if time_list[one_hop_node] > 1:
                    self.two_omnidirectional_symmetry_neighbors_time[node][one_hop_node] -= 1
                elif node in to_remove:
                    to_remove[node].append(one_hop_node)
                else:
                    to_remove[node] = [one_hop_node]
        
        for node, index_list in to_remove.items():
            for i, index in enumerate(sorted(index_list, reverse=True)):
                self.topology_table_time[node - 1][self.two_omnidirectional_symmetry_neighbors[node][index - i] - 1] = 1
                self.topology_table_time[self.two_omnidirectional_symmetry_neighbors[node][index - i] - 1][node - 1] = 1
                del self.two_omnidirectional_symmetry_neighbors[node][index - i]
                del self.two_omnidirectional_symmetry_neighbors_time[node][index - i]
            
            if len(self.two_omnidirectional_symmetry_neighbors[node]) == 0:
                del self.two_omnidirectional_symmetry_neighbors[node]
                del self.two_omnidirectional_symmetry_neighbors_time[node]

    def _remove_mpr_s(self):
        """移除超时的MPR选择器"""
        to_remove = [node for node, time in self.mpr_s_nodes.items() if time <= 1]
        for node in to_remove:
            del self.mpr_s_nodes[node]
            self.mpr_s_change = True
        
        for node in self.mpr_s_nodes:
            self.mpr_s_nodes[node] -= 1

    def _remove_topology_table(self):
        """更新拓扑表超时"""
        for i in range(len(self.topology_table_time)):
            for j in range(len(self.topology_table_time)):
                if i != j and self.topology_table_time[i][j] > 1:
                    self.topology_table_time[i][j] -= 1
                elif i != j and self.topology_table_time[i][j] <= 1:
                    self.topology_table[i][j] = float('inf')
                    self.topology_changed = True

    def _topology_table_change(self, other_node, link_stability, is_active):
        """
        更新拓扑表
        使用FSO链路稳定值的-np.log作为链路成本
        """
        cost = -np.log(link_stability) if is_active else float('inf')
        self.topology_table[self.node_id - 1][other_node.node_id - 1] = cost
        self.topology_table_time[self.node_id - 1][other_node.node_id - 1] = TIME_TOPOLOGY if is_active else 0
        self.topology_table[other_node.node_id - 1][self.node_id - 1] = cost
        self.topology_table_time[other_node.node_id - 1][self.node_id - 1] = TIME_TOPOLOGY if is_active else 0

    def _mpr_selection(self, all_nodes, max_velocity, initial_energy):
        """MPR选择算法,考虑覆盖度、MPR_S长度、移动稳定性和能量"""

        self.mpr_nodes = []
        normalized_velocity = max_velocity / 40
        
        if not self.one_omnidirectional_symmetry_neighbors or not self.two_omnidirectional_symmetry_neighbors:
            return
        
        one_hop_set = set(self.one_omnidirectional_symmetry_neighbors.keys())
        two_hop_set = set(self.two_omnidirectional_symmetry_neighbors.keys())
        selected_one_hop = set()
        
        # 第一步：选择唯一覆盖两跳邻居的一跳节点
        for node_2 in list(two_hop_set):
            n_1_list = self.two_omnidirectional_symmetry_neighbors[node_2]
            if len(n_1_list) == 1 and n_1_list[0] not in selected_one_hop:
                self.mpr_nodes.append(n_1_list[0])
                selected_one_hop.add(n_1_list[0])
                two_hop_set.remove(node_2)
        
        # 移除已被覆盖的两跳邻居
        for node_2 in list(two_hop_set):
            if bool(set(self.two_omnidirectional_symmetry_neighbors[node_2]) & selected_one_hop):
                two_hop_set.remove(node_2)
        
        remaining_one_hop = one_hop_set - selected_one_hop
        node_scores = {item: 0 for item in remaining_one_hop}
        
        if two_hop_set:
            # 覆盖度评分
            for node_2 in two_hop_set:
                for n in self.two_omnidirectional_symmetry_neighbors[node_2]:
                    node_scores[n] += 1
            
            max_cover = max(node_scores.values())
            for n_1 in node_scores.keys():
                node_scores[n_1] /= (max_cover / (0.5 - normalized_velocity / 10))
            
            # MPR_S长度评分
            mps_s_lengths = [len(all_nodes[n_1 - 1].mpr_s_nodes) for n_1 in node_scores.keys()]
            max_mps_s = max(mps_s_lengths) if mps_s_lengths else 0
            if max_mps_s != 0:
                for n_1 in node_scores.keys():
                    node_scores[n_1] += np.exp(-len(all_nodes[n_1 - 1].mpr_s_nodes) / max_mps_s) * (0.1 - normalized_velocity / 10)
            
            # 移动稳定性评分
            for n_1 in node_scores.keys():
                n_ls2 = 0
                ls2 = 0
                ls0 = calculate_rf_link_stability(all_nodes[n_1 - 1], self, max_velocity)
                for n_2 in two_hop_set:
                    if n_2 in all_nodes[n_1 - 1].one_omnidirectional_symmetry_neighbors:
                        ls2 += calculate_rf_link_stability(all_nodes[n_2 - 1], all_nodes[n_1 - 1], max_velocity)
                        n_ls2 += 1
                ls2 = ls2 / n_ls2 if n_ls2 else 0
                ls = ls0 * ls2
                node_scores[n_1] += ls * (0.4 + normalized_velocity / 5)
            
            # 能量评分
            for n_1 in node_scores.keys():
                energy_ratio = all_nodes[n_1 - 1].energy_rate / initial_energy
                if energy_ratio > 0.5:
                    e = 1
                elif 0.2 < energy_ratio < 0.5:
                    e = energy_ratio
                else:
                    e = energy_ratio / 5
                node_scores[n_1] *= e
            
            # 贪心选择
            heap = [(-count, node_1) for node_1, count in node_scores.items()]
            heapq.heapify(heap)
            while two_hop_set:
                _, max_mpr = heapq.heappop(heap)
                self.mpr_nodes.append(max_mpr)
                to_remove = {node_2 for node_2 in two_hop_set if max_mpr in self.two_omnidirectional_symmetry_neighbors[node_2]}
                two_hop_set -= to_remove

    def _update_neighbors_table(self, other_node, max_velocity, hello_message_size):
        """更新邻居表（全向Hello消息）"""
        # 清理定向邻居
        if other_node.node_id in self.one_directional_symmetry_neighbors:
            del self.one_directional_symmetry_neighbors[other_node.node_id]
        if other_node.node_id in self.one_directional_asymmetry_neighbors:
            del self.one_directional_asymmetry_neighbors[other_node.node_id]
        
        # 能耗计算
        communication_energy = calculate_communication_energy_energy(0, hello_message_size * 8, 0, 'rf')
        self.communication_energy += communication_energy
        self.energy_rate -= communication_energy
        if self.energy_rate <= 0:
            self.alive = False
            return False
        
        topology_changed = False
        
        # 邻居关系更新逻辑
        if other_node.node_id in self.one_omnidirectional_asymmetry_neighbors:
            if (self.node_id in other_node.one_omnidirectional_asymmetry_neighbors or
                    self.node_id in other_node.one_omnidirectional_symmetry_neighbors):
                del self.one_omnidirectional_asymmetry_neighbors[other_node.node_id]
                self.one_omnidirectional_symmetry_neighbors[other_node.node_id] = TIME_SYMMETRIC_NEIGHBORS
                if other_node.node_id in self.two_omnidirectional_symmetry_neighbors:
                    del self.two_omnidirectional_symmetry_neighbors[other_node.node_id]
                    del self.two_omnidirectional_symmetry_neighbors_time[other_node.node_id]
                topology_changed = True
                self._topology_table_change(other_node, calculate_fso_link_stability(self, other_node, max_velocity), True)
            else:
                self.one_omnidirectional_asymmetry_neighbors[other_node.node_id] = TIME_ASYMMETRIC_NEIGHBORS
        
        elif other_node.node_id in self.one_omnidirectional_symmetry_neighbors:
            if (self.node_id in other_node.one_omnidirectional_asymmetry_neighbors or
                    self.node_id in other_node.one_omnidirectional_symmetry_neighbors):
                self.one_omnidirectional_symmetry_neighbors[other_node.node_id] = TIME_SYMMETRIC_NEIGHBORS
                self._topology_table_change(other_node, calculate_fso_link_stability(self, other_node, max_velocity), True)
            else:
                del self.one_omnidirectional_symmetry_neighbors[other_node.node_id]
                self.one_omnidirectional_asymmetry_neighbors[other_node.node_id] = TIME_ASYMMETRIC_NEIGHBORS
                topology_changed = True
                self._topology_table_change(other_node, 0, False)
        else:
            if (self.node_id in other_node.one_omnidirectional_asymmetry_neighbors or
                    self.node_id in other_node.one_omnidirectional_symmetry_neighbors):
                self.one_omnidirectional_symmetry_neighbors[other_node.node_id] = TIME_SYMMETRIC_NEIGHBORS
                if other_node.node_id in self.two_omnidirectional_symmetry_neighbors:
                    del self.two_omnidirectional_symmetry_neighbors[other_node.node_id]
                    del self.two_omnidirectional_symmetry_neighbors_time[other_node.node_id]
                topology_changed = True
                self._topology_table_change(other_node, calculate_fso_link_stability(self, other_node, max_velocity), True)
            else:
                self.one_omnidirectional_asymmetry_neighbors[other_node.node_id] = TIME_ASYMMETRIC_NEIGHBORS
        
        # 更新两跳邻居
        if other_node.node_id in self.one_omnidirectional_symmetry_neighbors:
            for node in other_node.one_omnidirectional_symmetry_neighbors:
                if node != self.node_id and node not in self.one_omnidirectional_symmetry_neighbors:
                    if node not in self.two_omnidirectional_symmetry_neighbors:
                        self.two_omnidirectional_symmetry_neighbors[node] = [other_node.node_id]
                        self.two_omnidirectional_symmetry_neighbors_time[node] = [TIME_TWO_SYMMETRIC_NEIGHBORS]
                        topology_changed = True
                    elif other_node.node_id in self.two_omnidirectional_symmetry_neighbors[node]:
                        index = self.two_omnidirectional_symmetry_neighbors[node].index(other_node.node_id)
                        self.two_omnidirectional_symmetry_neighbors_time[node][index] = TIME_TWO_SYMMETRIC_NEIGHBORS

                    self.topology_table[node - 1][other_node.node_id - 1] = other_node.topology_table[node - 1][other_node.node_id - 1]
                    self.topology_table_time[node - 1][other_node.node_id - 1] = TIME_TOPOLOGY
                    self.topology_table[other_node.node_id - 1][node - 1] = other_node.topology_table[other_node.node_id - 1][node - 1]
                    self.topology_table_time[other_node.node_id - 1][node - 1] = TIME_TOPOLOGY
        
        # 更新MPR选择集
        if self.node_id in other_node.mpr_nodes:
            if other_node.node_id in self.mpr_nodes:
                self.mpr_s_change = True
            self.mpr_s_nodes[other_node.node_id] = TIME_MPR_S_SET
        
        return topology_changed

    def receive_hello_message(self, other_node, all_nodes, max_velocity, hello_message_size, initial_energy):
        """接收全向Hello消息"""
        if self._update_neighbors_table(other_node, max_velocity, hello_message_size):
            self._mpr_selection(all_nodes, max_velocity, initial_energy)

    def _update_directional_neighbors_table(self, other_node, max_velocity, hello_message_size):
        """更新定向邻居表（定向Hello消息）"""
        communication_energy = calculate_communication_energy_energy(0, hello_message_size * 8, 0, 'rf')
        self.communication_energy += communication_energy
        self.energy_rate -= communication_energy
        if self.energy_rate <= 0:
            self.alive = False
            return False
        
        topology_changed = False
        
        if other_node.node_id in self.one_directional_asymmetry_neighbors:
            if (self.node_id in other_node.one_directional_asymmetry_neighbors or
                    self.node_id in other_node.one_directional_symmetry_neighbors):
                del self.one_directional_asymmetry_neighbors[other_node.node_id]
                self.one_directional_symmetry_neighbors[other_node.node_id] = TIME_DIRECTION_NEIGHBORS
                self._topology_table_change(other_node, calculate_fso_link_stability(self, other_node, max_velocity), True)
                topology_changed = True
            else:
                self.one_directional_asymmetry_neighbors[other_node.node_id] = TIME_DIRECTION_NEIGHBORS
        
        elif other_node.node_id in self.one_directional_symmetry_neighbors:
            if (self.node_id in other_node.one_directional_asymmetry_neighbors or
                    self.node_id in other_node.one_directional_symmetry_neighbors):
                self.one_directional_symmetry_neighbors[other_node.node_id] = TIME_DIRECTION_NEIGHBORS
                self._topology_table_change(other_node, calculate_fso_link_stability(self, other_node, max_velocity), True)
            else:
                del self.one_directional_symmetry_neighbors[other_node.node_id]
                self.one_directional_asymmetry_neighbors[other_node.node_id] = TIME_DIRECTION_NEIGHBORS
                self._topology_table_change(other_node, 0, False)
                topology_changed = True
        else:
            if (self.node_id in other_node.one_directional_asymmetry_neighbors or
                    self.node_id in other_node.one_directional_symmetry_neighbors):
                self.one_directional_symmetry_neighbors[other_node.node_id] = TIME_DIRECTION_NEIGHBORS
                self._topology_table_change(other_node, calculate_fso_link_stability(self, other_node, max_velocity), True)
                topology_changed = True
            else:
                self.one_directional_asymmetry_neighbors[other_node.node_id] = TIME_DIRECTION_NEIGHBORS
        
        return topology_changed

    def receive_hello_message_d(self, other_node, max_velocity, hello_message_size):
        """接收定向Hello消息"""
        if (other_node.node_id not in self.one_omnidirectional_symmetry_neighbors and
                other_node.node_id not in self.one_omnidirectional_asymmetry_neighbors):
            self._update_directional_neighbors_table(other_node, max_velocity, hello_message_size)

    def receive_tc_message(self, other_node, tc_message_size):
        """
        接收TC消息
        """
        if other_node.node_id == self.node_id:
            return
        
        self.tc_receive_sequence[other_node.node_id] = other_node.tc_sequence
        
        communication_energy = calculate_communication_energy_energy(0, tc_message_size * 8, 0, 'rf')
        self.communication_energy += communication_energy
        self.energy_rate -= communication_energy
        if self.energy_rate <= 0:
            self.alive = False
            return
        
        # 更新拓扑信息
        for node in other_node.mpr_s_nodes:
            if self.topology_table[other_node.node_id - 1][node - 1] == float('inf'):
                self.topology_changed = True
            self.topology_table[other_node.node_id - 1][node - 1] = other_node.topology_table[other_node.node_id - 1][node - 1]
            self.topology_table_time[other_node.node_id - 1][node - 1] = TIME_TOPOLOGY
            
            if self.topology_table[node - 1][other_node.node_id - 1] == float('inf'):
                self.topology_changed = True
            self.topology_table[node - 1][other_node.node_id - 1] = other_node.topology_table[node - 1][other_node.node_id - 1]
            self.topology_table_time[node - 1][other_node.node_id - 1] = TIME_TOPOLOGY

        for node in other_node.one_directional_symmetry_neighbors:
            if self.topology_table[other_node.node_id - 1][node - 1] == float('inf'):
                self.topology_changed = True
            self.topology_table[other_node.node_id - 1][node - 1] = other_node.topology_table[other_node.node_id - 1][node - 1]
            self.topology_table_time[other_node.node_id - 1][node - 1] = TIME_TOPOLOGY
            
            if self.topology_table[node - 1][other_node.node_id - 1] == float('inf'):
                self.topology_changed = True
            self.topology_table[node - 1][other_node.node_id - 1] = other_node.topology_table[node - 1][other_node.node_id - 1]
            self.topology_table_time[node - 1][other_node.node_id - 1] = TIME_TOPOLOGY

    def update_route(self):
        """更新路由表"""
        self.route_table = dijkstra(self.topology_table, self.node_id)
        self.topology_changed = False
