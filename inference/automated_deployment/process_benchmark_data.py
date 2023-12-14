import csv
import os
import numpy as np
from enum_types import Server, Task
from constants import MICROSECONDS_LENGTH, MILLISECONDS_LENGTH, SECONDS_LENGTH, MINUTES_LENGTH
from constants import NUMBER_OF_MICROSEC_IN_SECOND, NUMBER_OF_MS_IN_SECOND, NUMBER_OF_SECONDS_IN_MINUTE
from constants import TOO_MANY_REQUEST_ERROR
import typer
from utils import load_json
from constants import CONFIG_FILE_PATH

def save_data_for_final_table(csv_file_path, data):
    headers = ["model", "server", "rps", "latency_with_deviation", "throughput_with_deviation", "duration_with_deviation"]

    write_header = not os.path.exists(csv_file_path) or os.path.getsize(csv_file_path) == 0

    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(headers)
        writer.writerow(data)

def convert_to_seconds(time):
    if 'ms' in time:
        return float(time[:-MILLISECONDS_LENGTH]) / NUMBER_OF_MS_IN_SECOND
    elif 'µs' in time:
        return float(time[:-MICROSECONDS_LENGTH]) / NUMBER_OF_MICROSEC_IN_SECOND
    elif 'm' in time and 's' in time:
        return float(time[:-MINUTES_LENGTH]) * NUMBER_OF_SECONDS_IN_MINUTE
    else:
        return float(time[:-SECONDS_LENGTH])

def get_metrics(raw_results_path: str, processed_results_path: str):
    rate = 0
    config = load_json(CONFIG_FILE_PATH)
    server = config["server"]
    model = config["model_name"]
    with open(raw_results_path) as f:
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

        save_data_for_final_table(processed_results_path, [model, server, rate, result_dict[max_total_request]['latency_with_deviation'], 
                                                        result_dict[max_total_request]['throughput_with_deviation'], 
                                                        result_dict[max_total_request]['duration_with_deviation']])

if __name__ == '__main__':

    typer.run(get_metrics)