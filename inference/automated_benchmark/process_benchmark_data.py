import csv
import os
import numpy as np
from enum_types import Server, Task
from constants import MICROSECONDS_LENGTH, MILLISECONDS_LENGTH, SECONDS_LENGTH, MINUTES_LENGTH
from constants import NUMBER_OF_MICROSEC_IN_SECOND, NUMBER_OF_MS_IN_SECOND, NUMBER_OF_SECONDS_IN_MINUTE
from constants import RAW_DIR, PROCESSED_DIR, PLOTS_DIR, TOO_MANY_REQUEST_ERROR
import typer

def calculate_num_of_tokens(task, server):
    input_tokens = 100
    expected_output_tokens_class = 4
    expected_output_tokens_summ = 50
    actual_output_tokens = 0

    if task == Task.CLASSIFICATION.value:
        actual_output_tokens = expected_output_tokens_class
        if server == Server.RAY.value:
            actual_output_tokens += input_tokens
    else:
        actual_output_tokens = expected_output_tokens_summ
        if server == Server.RAY.value:
            actual_output_tokens += input_tokens

    return input_tokens, actual_output_tokens

def save_data_for_final_table(csv_file_path, task, data, instance_cost):
    headers = ["hardware", "server", "rps", "latency_with_deviation", "throughput_with_deviation", "duration_with_deviation", 
               "input_tokens", "output_tokens", "cost"]
    input_tokens, output_tokens = calculate_num_of_tokens(task, data[1])
    number_of_seconds = 3600
    number_of_tokens = 1000
    cost = instance_cost / (data[2] * number_of_seconds * (input_tokens + output_tokens)) * number_of_tokens
    data.extend([input_tokens, output_tokens, "{:.10f}".format(cost)])

    write_header = not os.path.exists(csv_file_path) or os.path.getsize(csv_file_path) == 0

    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(headers)
        writer.writerow(data)
        
def save_data_for_final_plot(csv_file_path, requests, latencies, throughputs, durations):
    headers = ["number of requests", "latency", "throughput", "duration"]
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for data_row in zip(requests, latencies, throughputs, durations):
            writer.writerow(data_row)

def convert_to_seconds(time):
    if 'ms' in time:
        return float(time[:-MILLISECONDS_LENGTH]) / NUMBER_OF_MS_IN_SECOND
    elif 'µs' in time:
        return float(time[:-MICROSECONDS_LENGTH]) / NUMBER_OF_MICROSEC_IN_SECOND
    elif 'm' in time and 's' in time:
        return float(time[:-MINUTES_LENGTH]) * NUMBER_OF_SECONDS_IN_MINUTE
    else:
        return float(time[:-SECONDS_LENGTH])

def get_metrics(huggingface_rep: str, task: str, hardware: str, server: str, instance_cost: str):
    instance_cost = float(instance_cost)
    model_name = huggingface_rep.split('/')[1]
    rate = 0
    with open(f"{RAW_DIR}/{model_name}/{hardware}/{server}.txt") as f:
        path_processed = f"{PROCESSED_DIR}/{model_name}.csv"
        path_plot = f"{PLOTS_DIR}/{model_name}/{hardware}/{server}.csv"
        benchmark_logs = f.readlines()
        result_dict = {}
        max_total_request = 0
        raws = [i.split() for i in benchmark_logs]
        for raw in raws:
            if len(raw) > 0:
                if raw[0] == TOO_MANY_REQUEST_ERROR: 
                    break
                if raw[0] == 'Requests':
                    pos_of_total_request_value = 4
                    pos_of_throughput_value = 6
                    pos_of_rate_value = 5

                    total_request = int(raw[pos_of_total_request_value][:-1])
                    if result_dict.get(total_request) is None:
                        result_dict[total_request] = {'latency': [], 
                                                      'throughput': [convert_to_seconds(raw[pos_of_throughput_value])],
                                                      'count': 1, 
                                                      'duration': []}
                    else:
                        result_dict[total_request]['count'] += 1
                        result_dict[total_request]['throughput'].append(convert_to_seconds(raw[pos_of_throughput_value]))
                    max_total_request = total_request
                    rate = float(raw[pos_of_rate_value][:-1])
                if raw[0] == 'Duration':
                    pos_of_duration_value = 4
                    result_dict[max_total_request]['duration'].append(convert_to_seconds(raw[pos_of_duration_value]))
                if raw[0] == 'Latencies':
                    pos_of_latency_value = 11
                    result_dict[max_total_request]['latency'].append(convert_to_seconds(raw[pos_of_latency_value]))
        
        keys_to_modify = ['latency', 'duration', 'throughput']

        for num_req in result_dict.keys():
            for key in keys_to_modify:
                mean_value = np.mean(result_dict[num_req][key])
                std_deviation = np.std(result_dict[num_req][key])

                formatted_mean = "{:.3f}".format(mean_value)
                formatted_std_dev = "{:.3f}".format(std_deviation)

                result_dict[num_req][f"{key}_with_deviation"] = f"{formatted_mean}±{formatted_std_dev}"
                result_dict[num_req][key] = mean_value

        requests = list(result_dict.keys())
        latencies = [result_dict[num_req]['latency'] for num_req in result_dict]
        throughputs = [result_dict[num_req]['throughput'] for num_req in result_dict]
        durations = [result_dict[num_req]['duration'] for num_req in result_dict]

        save_data_for_final_table(path_processed, task, [hardware, server, rate, result_dict[max_total_request]['latency_with_deviation'], 
                                                        result_dict[max_total_request]['throughput_with_deviation'], 
                                                        result_dict[max_total_request]['duration_with_deviation']], 
                                                        instance_cost)
        save_data_for_final_plot(path_plot, requests, latencies, throughputs, durations)

if __name__ == '__main__':
    typer.run(get_metrics)