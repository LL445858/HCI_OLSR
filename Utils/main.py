import sys
import random
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from BASE.protocol import run_base_protocol
from BASE_Improve.protocol import run_i_base_protocol
from OC_OLSR.protocol import run_oc_olsr_protocol
from DC_OLSR.protocol import run_dc_olsr_protocol
from HC_OLSR.protocol import run_hc_olsr_protocol
from HCI_OLSR.protocol import run_hci_olsr_protocol
from Utils.node_mobility import write_motion_data_to_files


def run_protocol(protocol_func, args, protocol_name):
    result = protocol_func(*args)
    return protocol_name, result


def main(node_count, simulation_steps, max_velocity, message_count, energy_config):
    formatted_time = datetime.now().strftime('%Y_%m_%d_%H_%M')
    position_file_path = r'../text/position/true/' + f'{formatted_time}.txt'
    prediction_file_path = position_file_path.replace('true', 'predict')
    
    senders, receivers, send_times, packet_sizes = [], [], [], []
    for i in range(message_count):
        sender = random.randint(1, node_count)
        receiver = random.randint(1, node_count)
        while receiver == sender:
            receiver = random.randint(1, node_count)
        senders.append(sender)
        receivers.append(receiver)
        send_time = random.randint(11, simulation_steps - 11)
        send_times.append(send_time)
        packet_sizes.append(5)
    
    send_times = sorted(send_times)
    initial_energy = 3.6e5 if energy_config else 1e100

    write_motion_data_to_files(node_count, simulation_steps, max_velocity, position_file_path, prediction_file_path)
    common_args = (node_count, simulation_steps, max_velocity, senders, receivers, send_times, 
                   packet_sizes, initial_energy, position_file_path)

    results = {}
    with ProcessPoolExecutor(max_workers=6) as executor:
        future_to_protocol = {
            executor.submit(run_protocol, run_base_protocol, common_args, "BASE"): "BASE",
            executor.submit(run_protocol, run_i_base_protocol, common_args + (prediction_file_path,), "BASE_Improve"): "BASE_Improve",
            executor.submit(run_protocol, run_hci_olsr_protocol, common_args + (prediction_file_path,), "HCI-OLSR"): "HCI-OLSR",
            executor.submit(run_protocol, run_hc_olsr_protocol, common_args, "HC-OLSR"): "HC-OLSR",
            executor.submit(run_protocol, run_oc_olsr_protocol, common_args, "OC-OLSR"): "OC-OLSR",
            executor.submit(run_protocol, run_dc_olsr_protocol, common_args, "DC-OLSR"): "DC-OLSR"
        }

        for future in as_completed(future_to_protocol):
            protocol_name, result = future.result()
            results[protocol_name] = result

    for protocol_name, result in results.items():
        if result is not None:
            (alive_history, success_counts, avg_delay, control_overhead,
             communication_energy, mobile_energy, failure_stats,
             tc_success_rate, route_lengths) = result

            print(
                f'\n\n{protocol_name}\n'
                f'节点生存时长: {len(alive_history)}秒; '
                f'平均分组递交率: {round(np.average(success_counts) * 10, 2)}%; '
                f'平均时延: {avg_delay * np.average(route_lengths):.2f}ms; '
                f'TC消息覆盖率: {tc_success_rate*100:.2f}%;\n'
                f'平均路由长度: {np.average(route_lengths) if route_lengths else 0:.2f}; '
                f'控制消息开销: {control_overhead / 1e3:.2f}KB/s; '
                f'平均通信能耗: {communication_energy / (node_count * simulation_steps):.2e}J/s; '
                f'平均移动能耗: {mobile_energy/ (node_count * simulation_steps) :.2e}J/s;\n'
                f'失败原因占比统计: 未发现可达路由: {failure_stats[0]}% '
                f'丢包: {failure_stats[1]}%；'
                f'节点超距: {failure_stats[2]}%；'
                f'路由回环: {failure_stats[3]}%；'
                f'中间路由节点无能量: {failure_stats[4]}%\n'

            )
        else:
            print(f"\n{protocol_name}：执行失败")


if __name__ == '__main__':
    max_velocity = 40
    node_counts = 300
    simulation_steps = 200
    energy_config = False
    formatted_time = datetime.now().strftime('%Y_%m_%d_%H_%M')
    output_path = r'../text/' + f'result_{formatted_time}.txt'
    sys.stdout = open(output_path, 'w', encoding='utf-8')

    for velocity in range(10, max_velocity + 1, 5):
        message_count = int((simulation_steps - 20) * node_counts * 2)
        print(f'\n-----当前节点数为{node_counts},当前速度为{velocity}-----')
        main(node_counts, simulation_steps, velocity, message_count, energy_config=energy_config)
