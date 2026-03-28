#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')  # 或者 'Qt5Agg'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑字体
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', family='Microsoft YaHei')  # 设置全局


def plot_static():
    plt.figure(figsize=(6, 4), dpi=180)  # 设置图像大小和dpi
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    x = [i * 5 for i in range(9)]

    y0 = [88.65, 87.50, 85.74, 83.00, 80.00, 71.56, 60.48, 51.61, 41.78]
    y5 = [90.20, 89.88, 88.58, 86.30, 82.66, 74.87, 65.49, 56.65, 48.20]
    y1 = [93.76, 92.67, 90.91, 89.04, 86.77, 84.25, 80.96, 73.45, 60.74]
    y2 = [87.92, 83.55, 79.37, 76.66, 71.91, 64.56, 54.92, 45.63, 34.03]
    y3 = [77.34, 74.56, 70.00, 67.39, 62.78, 55.03, 44.28, 36.49, 26.18]
    y4 = [91.02, 89.25, 87.75, 84.66, 78.73, 68.67, 58.10, 49.65, 37.95]
    plt.ylabel('分组递交率 (%)')

    plt.plot(x, y1, marker='*', ls='-', color='#0C79F7', alpha=0.9, label='HCI-OLSR', linewidth=0.8, markersize=5)
    plt.plot(x, y2, marker='s', ls='-', color='#E65576', alpha=0.9, label='HC-OLSR', linewidth=0.8, markersize=4)
    plt.plot(x, y3, marker='o', ls='-', color='#AC49F5', alpha=0.9, label='OC-OLSR', linewidth=0.8, markersize=5)
    plt.plot(x, y5, marker='^', ls='-', color='#F9A01E', alpha=0.9, label='HCI_BASE', linewidth=0.8, markersize=5)
    plt.plot(x, y0, marker='h', ls='-', color='#61EFDE', alpha=0.9, label='HC_BASE', linewidth=0.8, markersize=5)
    plt.plot(x, y4, marker='x', ls='-', color='#64BB5C', alpha=0.9, label='DC-OLSR', linewidth=0.8, markersize=4)

    plt.xlabel('节点最大速度 (m/s)')
    plt.xticks(range(int(min(x)), int(max(x)) + 1, 5))
    plt.yticks(range(0, 5001, 500))
    plt.legend(loc='lower left', fontsize=7.5)
    plt.grid(True)
    plt.show()


def plot_control_cost():
    # 创建两个垂直排列的子图，共享x轴
    fig, (ax1, ax2) = plt.subplots(
        3, 1,
        figsize=(6, 8),  # 增加总高度
        dpi=150,
        sharex=True,
        gridspec_kw={'height_ratios': [1, 2, 3]}  # 关键参数
    )

    x = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    y0 = [150, 157, 165, 167, 171, 185, 203, 217, 229]
    y5 = [192, 199, 208, 212, 216, 230, 245, 257, 269]
    y1 = [242, 248, 255, 259, 265, 276, 287, 295, 306]
    y2 = [58.9, 60.1, 63.1, 65.3, 67.7, 73.3, 78.5, 83.1, 89.9]
    y3 = [65.9, 67.4, 70.7, 75.1, 80.2, 89.1, 98.1, 110.4, 120.6]
    y4 = [4200, 4280, 4360, 4440, 4520, 4535, 4550, 4760, 5040]

    ax1.plot(x, y4, marker='x', ls='-', color='#64BB5C', label='DC-OLSR    ', linewidth=1, markersize=5, alpha=0.9)
    ax1.legend(loc='upper left', fontsize=7.8)
    ax1.grid(True)
    ax1.set_yticks(range(4000, 6001, 1000))

    ax2.plot(x, y5, marker='^', ls='-', color='#F9A01E', label='HCI_BASE', linewidth=1, markersize=5, alpha=0.9)
    ax2.plot(x, y0, marker='h', ls='-', color='#61EFDE', label='HC_BASE', linewidth=1, markersize=5, alpha=0.9)
    ax2.legend(loc='upper left', fontsize=7.8)
    ax2.grid(True)
    ax2.set_yticks(range(4000, 6001, 1000))

    ax2.plot(x, y1, marker='*', ls='-', color='#0C79F7', label='HCI-OLSR', linewidth=1, markersize=5, alpha=0.9)
    ax2.plot(x, y2, marker='s', ls='-', color='#E65576', label='HC-OLSR', linewidth=1, markersize=5, alpha=0.9)
    ax2.plot(x, y3, marker='o', ls='-', color='#AC49F5', label='OC-OLSR', linewidth=1, markersize=5, alpha=0.9)
    ax2.legend(loc='upper left', fontsize=7.8)
    ax2.grid(True)
    ax2.set_yticks(range(0, 401, 100))

    plt.xlabel('节点最大速度(m/s)')
    plt.ylabel('控制开销(MB)')

    plt.xticks(x)
    plt.show()


def plot_polar_graph():
    # 参数设置
    I0 = 1.0  # 主瓣处最大光强
    theta_1e = np.deg2rad(5)  # 1/e^2半角宽度，5度转换为弧度

    # 生成极坐标下的角度数据，从0到2π
    theta = np.linspace(0, 2 * np.pi, 400)

    # 计算主瓣偏离角度：对于θ>π时，采用2π-θ
    deviation = np.where(theta > np.pi, 2 * np.pi - theta, theta)

    # 计算高斯光束的辐射方向图
    I_gaussian = I0 * np.exp(-2 * (deviation ** 2) / (theta_1e ** 2))

    # 全向天线（各方向均相同）辐射图：常数曲线
    I_omni = 0.5 * np.ones_like(theta)

    # 绘制极坐标图
    plt.figure(figsize=(4, 4), dpi=120)
    ax = plt.subplot(111, projection='polar')
    ax.plot(theta, I_gaussian, label='FSO')
    ax.plot(theta, I_omni, label='RF', linestyle='-.')
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.show()


def plot_fail_reason():
    # 新标签和颜色配置（显示四种失败原因）
    labels = ['路由不可达', '信道丢包', '节点超距', '路由环路']
    colors = ['#0C79F7', '#E65576', '#64BB5C', '#AC49F5']

    protocol_names = {
        '协议F': 'pre_baseline',
        '协议E': 'baseline',
        '协议D': 'D_OLSR',
        '协议C': 'O_OLSR',
        '协议B': 'HC_OLSR',
        '协议A': 'HCI_OLSR'
    }

    original_data = {
        0: {
            '协议A': [0, 100, 0, 0],
            '协议B': [97, 3, 0, 0],
            '协议C': [2, 98, 0, 0],
            '协议D': [0, 100, 0, 0],
            '协议E': [0, 100, 0, 0],
            '协议F': [0, 100, 0, 0]

        },
        10: {
            '协议A': [2, 97, 1, 0],
            '协议B': [80, 17, 0, 3],
            '协议C': [3, 10, 85, 2],
            '协议D': [0, 95, 4, 1],
            '协议E': [0, 97, 3, 0],
            '协议F': [0, 100, 0, 0]
        },
        20: {
            '协议A': [16, 82, 1, 1],
            '协议B': [72, 20, 0, 8],
            '协议C': [4, 12, 81, 3],
            '协议D': [0, 60, 39, 1],
            '协议E': [2, 94, 4, 0],
            '协议F': [2, 98, 0, 0]
        },
        30: {
            '协议A': [43, 54, 2, 1],
            '协议B': [70, 21, 0, 9],
            '协议C': [16, 13, 69, 2],
            '协议D': [2, 47, 50, 1],
            '协议E': [6, 78, 14, 2],
            '协议F': [11, 87, 0, 2]
        },
        40: {
            '协议A': [60, 31, 8, 1],
            '协议B': [82, 13, 0, 5],
            '协议C': [44, 12, 42, 2],
            '协议D': [4, 43, 52, 1],
            '协议E': [11, 63, 24, 2],
            '协议F': [26, 72, 0, 2]
        }
        # 其他速度值可自行添加
    }

    def process_data(original):
        return {
            speed: {
                protocol_names[proto]: values
                for proto, values in protocols.items()
            }
            for speed, protocols in original.items()
        }

    data = process_data(original_data)

    speeds = sorted(data.keys())
    protocols = list(protocol_names.values())
    num_protocols = len(protocols)
    height = 0.1
    spacing = 0.05
    proto_positions = (np.arange(num_protocols) - (num_protocols - 1) / 2) * (height + spacing)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)

    for speed_idx, speed in enumerate(speeds):
        for proto_idx, proto in enumerate(protocols):
            values = data[speed][proto]
            y = speed_idx + proto_positions[proto_idx]
            left = 0

            for i in range(4):
                ax.barh(y, values[i], height, left=left,
                        color=colors[i], edgecolor='white')
                left += values[i]

            # 标签简写显示
            short_label = {
                'HCI_OLSR': 'HCI-OLSR',
                'HC_OLSR': 'HC-OLSR',
                'O_OLSR': 'OC-OLSR',
                'D_OLSR': 'DC-OLSR',
                'baseline': 'HC-BASE',
                'pre_baseline': 'HCI-BASE'
            }[proto]

            ax.text(105, y, short_label, va='center', ha='left', fontsize=7, fontweight='light')

    y_base = np.arange(len(speeds))
    ax.set_yticks(y_base)
    ax.set_yticklabels([f"{s}" for s in speeds])
    ax.set_ylabel('节点最大速度(m/s)', fontsize=13)
    ax.set_xlabel('各个失败原因占比 (%)', fontsize=13)
    ax.set_xlim(0, 115)

    title_handle = plt.Rectangle((0, 0), 0, 0, fc="none", edgecolor="none")
    color_handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    new_labels = ["失败原因:"] + labels
    new_handles = [title_handle] + color_handles

    ax.legend(new_handles, new_labels,
              loc='upper center',
              bbox_to_anchor=(0.47, 1.06),
              ncol=5,
              frameon=False, fontsize=9.8)

    plt.tight_layout()
    plt.show()

