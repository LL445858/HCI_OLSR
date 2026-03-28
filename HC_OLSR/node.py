#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Utils.energy_model import calculate_communication_energy_energy
from Utils.utils import dijkstra

# 超时时间常量
TIMEOUT_TOPOLOGY = 6
TIMEOUT_MPR_SELECTOR = 6
TIMEOUT_ASYMMETRIC_NEIGHBOR = 4
TIMEOUT_SYMMETRIC_NEIGHBOR = 6
TIMEOUT_TWO_HOP_NEIGHBOR = 6


class Node:
    """HC_OLSR协议节点类 - 混合OLSR路由（RF控制 + FSO业务）"""
    
    def __init__(self, node_id, node_count, position, initial_energy):
        self.node_id = node_id
        self.position = position
        self.position_before = [0, 0, 0]
        self.node_count = node_count
        self.energy_rate = initial_energy
        
        # 开销统计
        self.cost_hello = 0
        self.cost_tc = 0
        self.cost_message = 0
        self.communication_energy = 0
        self.mobile_energy = 0
        
        # 状态管理
        self.alive = True
        self.speed = [0, 0, 0]
        self.topology_changed = False
        self.mpr_selector_changed = False
        
        # TC消息管理
        self.tc_sequence = 1
        self.tc_success = []
        self.tc_receive_sequence = {i: 0 for i in range(1, node_count + 1)}
        
        # 邻居管理
        self.one_hop_asymmetric_neighbors = {}  # 一跳非对称邻居
        self.one_hop_symmetric_neighbors = {}   # 一跳对称邻居
        self.two_hop_neighbors = {}             # 两跳邻居
        self.two_hop_neighbors_time = {}        # 两跳邻居超时时间
        
        # MPR管理
        self.mpr_set = []                       # MPR集合
        self.mpr_selector_set = {}              # MPR选择集
        
        # 拓扑和路由
        self.topology_table = [[float('inf')] * node_count for _ in range(node_count)]
        self.topology_table_time = [[0] * node_count for _ in range(node_count)]
        for i in range(node_count):
            self.topology_table[i][i] = 0
        self.route_table = {}

    def update_neighbors(self):
        """更新邻居状态，处理超时"""
        self.topology_changed = False
        self.mpr_selector_changed = False
        
        self._remove_asymmetric_neighbors()
        self._remove_symmetric_neighbors()
        self._remove_two_hop_neighbors()
        self._remove_mpr_selectors()
        self._update_topology_table_timeouts()

    def _remove_asymmetric_neighbors(self):
        """移除超时的一跳非对称邻居"""
        to_remove = [node for node, time in self.one_hop_asymmetric_neighbors.items() if time <= 1]
        for node in to_remove:
            del self.one_hop_asymmetric_neighbors[node]
        
        for node in self.one_hop_asymmetric_neighbors:
            self.one_hop_asymmetric_neighbors[node] -= 1

    def _remove_symmetric_neighbors(self):
        """移除超时的一跳对称邻居，降级为非对称邻居"""
        to_remove = [node for node, time in self.one_hop_symmetric_neighbors.items() if time <= 1]
        
        for node in to_remove:
            del self.one_hop_symmetric_neighbors[node]
            self.topology_table_time[node - 1][self.node_id - 1] = 1
            self.topology_table_time[self.node_id - 1][node - 1] = 1
            self.one_hop_asymmetric_neighbors[node] = TIMEOUT_ASYMMETRIC_NEIGHBOR
            
            if node in self.mpr_set:
                self.mpr_set.remove(node)
            
            # 从两跳邻居中移除
            for two_hop_node, one_hop_list in list(self.two_hop_neighbors.items()):
                if node in one_hop_list:
                    self.topology_table_time[node - 1][two_hop_node - 1] = 1
                    self.topology_table_time[two_hop_node - 1][node - 1] = 1
                    index = self.two_hop_neighbors[two_hop_node].index(node)
                    del self.two_hop_neighbors[two_hop_node][index]
                    del self.two_hop_neighbors_time[two_hop_node][index]
                    
                    if not self.two_hop_neighbors[two_hop_node]:
                        del self.two_hop_neighbors[two_hop_node]
                        del self.two_hop_neighbors_time[two_hop_node]
        
        for node in self.one_hop_symmetric_neighbors:
            self.one_hop_symmetric_neighbors[node] -= 1

    def _remove_two_hop_neighbors(self):
        """移除超时的两跳邻居"""
        to_remove = {}
        for node, time_list in self.two_hop_neighbors_time.items():
            for i, time in enumerate(time_list):
                if time <= 1:
                    to_remove.setdefault(node, []).append(i)
        
        for node, index_list in to_remove.items():
            for i, index in enumerate(sorted(index_list, reverse=True)):
                one_hop_node = self.two_hop_neighbors[node][index]
                self.topology_table_time[node - 1][one_hop_node - 1] = 1
                self.topology_table_time[one_hop_node - 1][node - 1] = 1
                del self.two_hop_neighbors[node][index]
                del self.two_hop_neighbors_time[node][index]
            
            if not self.two_hop_neighbors[node]:
                del self.two_hop_neighbors[node]
                del self.two_hop_neighbors_time[node]
        
        for node in self.two_hop_neighbors_time:
            for i in range(len(self.two_hop_neighbors_time[node])):
                self.two_hop_neighbors_time[node][i] -= 1

    def _remove_mpr_selectors(self):
        """移除超时的MPR选择器"""
        to_remove = [node for node, time in self.mpr_selector_set.items() if time <= 1]
        for node in to_remove:
            del self.mpr_selector_set[node]
            self.mpr_selector_changed = True
        
        for node in self.mpr_selector_set:
            self.mpr_selector_set[node] -= 1

    def _update_topology_table_timeouts(self):
        """更新拓扑表超时"""
        for i in range(len(self.topology_table_time)):
            for j in range(len(self.topology_table_time)):
                if i != j and self.topology_table_time[i][j] > 1:
                    self.topology_table_time[i][j] -= 1
                elif i != j and self.topology_table_time[i][j] <= 1:
                    self.topology_table[i][j] = float('inf')
                    self.topology_changed = True

    def receive_hello_message(self, source_node, message_size):
        """接收Hello消息，更新邻居关系"""
        self.communication_energy += calculate_communication_energy_energy(0, message_size * 8, 0, 'rf')
        self.energy_rate -= calculate_communication_energy_energy(0, message_size * 8, 0, 'rf')
        
        if self.energy_rate <= 0:
            self.alive = False
            return
        
        topology_changed = self._update_neighbor_relationship(source_node)
        
        if topology_changed:
            self._select_mpr()

    def _update_neighbor_relationship(self, source_node):
        """更新与源节点的邻居关系"""
        def update_topology_table(flag):
            self.topology_table[self.node_id - 1][source_node.node_id - 1] = 1 if flag else float('inf')
            self.topology_table_time[self.node_id - 1][source_node.node_id - 1] = TIMEOUT_TOPOLOGY if flag else 0
            self.topology_table[source_node.node_id - 1][self.node_id - 1] = 1 if flag else float('inf')
            self.topology_table_time[source_node.node_id - 1][self.node_id - 1] = TIMEOUT_TOPOLOGY if flag else 0
        
        is_source_asymmetric = self.node_id in source_node.one_hop_asymmetric_neighbors
        is_source_symmetric = self.node_id in source_node.one_hop_symmetric_neighbors
        
        if source_node.node_id in self.one_hop_asymmetric_neighbors:
            if is_source_asymmetric or is_source_symmetric:
                del self.one_hop_asymmetric_neighbors[source_node.node_id]
                self.one_hop_symmetric_neighbors[source_node.node_id] = TIMEOUT_SYMMETRIC_NEIGHBOR
                if source_node.node_id in self.two_hop_neighbors:
                    del self.two_hop_neighbors[source_node.node_id]
                    del self.two_hop_neighbors_time[source_node.node_id]
                self.topology_changed = True
                update_topology_table(True)
            else:
                self.one_hop_asymmetric_neighbors[source_node.node_id] = TIMEOUT_ASYMMETRIC_NEIGHBOR
        
        elif source_node.node_id in self.one_hop_symmetric_neighbors:
            if is_source_asymmetric or is_source_symmetric:
                self.one_hop_symmetric_neighbors[source_node.node_id] = TIMEOUT_SYMMETRIC_NEIGHBOR
                update_topology_table(True)
            else:
                del self.one_hop_symmetric_neighbors[source_node.node_id]
                self.one_hop_asymmetric_neighbors[source_node.node_id] = TIMEOUT_ASYMMETRIC_NEIGHBOR
                self.topology_changed = True
                update_topology_table(False)
        
        else:
            if is_source_asymmetric or is_source_symmetric:
                self.one_hop_symmetric_neighbors[source_node.node_id] = TIMEOUT_SYMMETRIC_NEIGHBOR
                if source_node.node_id in self.two_hop_neighbors:
                    del self.two_hop_neighbors[source_node.node_id]
                    del self.two_hop_neighbors_time[source_node.node_id]
                self.topology_changed = True
                update_topology_table(True)
            else:
                self.one_hop_asymmetric_neighbors[source_node.node_id] = TIMEOUT_ASYMMETRIC_NEIGHBOR
        
        # 更新两跳邻居
        if source_node.node_id in self.one_hop_symmetric_neighbors:
            for neighbor in source_node.one_hop_symmetric_neighbors:
                if neighbor != self.node_id and neighbor not in self.one_hop_symmetric_neighbors:
                    if neighbor not in self.two_hop_neighbors:
                        self.two_hop_neighbors[neighbor] = [source_node.node_id]
                        self.two_hop_neighbors_time[neighbor] = [TIMEOUT_TWO_HOP_NEIGHBOR]
                        self.topology_changed = True
                    elif source_node.node_id in self.two_hop_neighbors[neighbor]:
                        index = self.two_hop_neighbors[neighbor].index(source_node.node_id)
                        self.two_hop_neighbors_time[neighbor][index] = TIMEOUT_TWO_HOP_NEIGHBOR
                    
                    self.topology_table[neighbor - 1][source_node.node_id - 1] = 1
                    self.topology_table_time[neighbor - 1][source_node.node_id - 1] = TIMEOUT_TOPOLOGY
        
        # 更新MPR选择集
        if self.node_id in source_node.mpr_set:
            if source_node.node_id in self.mpr_set:
                self.mpr_selector_changed = True
            self.mpr_selector_set[source_node.node_id] = TIMEOUT_MPR_SELECTOR
        
        return self.topology_changed

    def _select_mpr(self):
        """选择MPR节点以覆盖所有两跳邻居"""
        self.mpr_set = []
        if not self.one_hop_symmetric_neighbors or not self.two_hop_neighbors:
            return
        
        one_hop_set = set(self.one_hop_symmetric_neighbors.keys())
        two_hop_set = set(self.two_hop_neighbors.keys())
        selected_one_hop = set()
        
        # 第一步：选择唯一覆盖两跳邻居的一跳节点
        for two_hop_node in list(two_hop_set):
            one_hop_list = self.two_hop_neighbors[two_hop_node]
            if len(one_hop_list) == 1 and one_hop_list[0] not in selected_one_hop:
                self.mpr_set.append(one_hop_list[0])
                selected_one_hop.add(one_hop_list[0])
                two_hop_set.remove(two_hop_node)
        
        # 移除已被覆盖的两跳邻居
        for two_hop_node in list(two_hop_set):
            if any(n in selected_one_hop for n in self.two_hop_neighbors[two_hop_node]):
                two_hop_set.remove(two_hop_node)
        
        remaining_one_hop = one_hop_set - selected_one_hop
        coverage_count = {node: 0 for node in remaining_one_hop}
        
        for two_hop_node in two_hop_set:
            for one_hop in self.two_hop_neighbors[two_hop_node]:
                if one_hop in coverage_count:
                    coverage_count[one_hop] += 1
        
        # 贪心选择覆盖最多未覆盖两跳邻居的节点
        while two_hop_set:
            max_coverage_node = max(coverage_count, key=coverage_count.get)
            self.mpr_set.append(max_coverage_node)
            
            to_remove = {node for node in two_hop_set if max_coverage_node in self.two_hop_neighbors[node]}
            two_hop_set -= to_remove
            del coverage_count[max_coverage_node]

    def receive_tc_message(self, source_node, message_size):
        """接收TC消息，更新拓扑表"""
        if source_node.node_id == self.node_id:
            return
        
        self.tc_receive_sequence[source_node.node_id] = source_node.tc_sequence
        self.communication_energy += calculate_communication_energy_energy(0, message_size * 8, 0, 'rf')
        self.energy_rate -= calculate_communication_energy_energy(0, message_size * 8, 0, 'rf')
        
        if self.energy_rate <= 0:
            self.alive = False
            return
        
        for mpr_selector in source_node.mpr_selector_set:
            if self.topology_table[source_node.node_id - 1][mpr_selector - 1] != 1:
                self.topology_table[source_node.node_id - 1][mpr_selector - 1] = 1
                self.topology_changed = True
            self.topology_table_time[source_node.node_id - 1][mpr_selector - 1] = TIMEOUT_TOPOLOGY
            
            if self.topology_table[mpr_selector - 1][source_node.node_id - 1] != 1:
                self.topology_table[mpr_selector - 1][source_node.node_id - 1] = 1
                self.topology_changed = True
            self.topology_table_time[mpr_selector - 1][source_node.node_id - 1] = TIMEOUT_TOPOLOGY

    def update_route(self):
        """使用Dijkstra算法更新路由表"""
        self.route_table = dijkstra(self.topology_table, self.node_id)
        self.topology_changed = False
