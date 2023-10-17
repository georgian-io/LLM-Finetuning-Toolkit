import csv
import os
import sys
    

def calculate_num_of_tokens(task, server):
    input = 100
    if task == "classification":
        if server == "vllm" or server == "tgi":
            output = 4
        else:
            output = input + 4
    else:
        if server == "vllm" or server == "tgi":
            output = 60
        else:
            output = input + 60
    return input, output

def save_data_for_final_table(csv_file_path, task, data, instance_cost):
    headers = ["compute", "server", "rps", "latency", "throughput", "duration", "input_tokens", "output_tokens", "cost"]
    input, output = calculate_num_of_tokens(task, data[1]) 
    cost = instance_cost / (data[2] * 3600 * (input + output)) * 1000
    data.extend([input, output, "{:.10f}".format(cost)]) 
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
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
        return float(time[:-3]) / 1000
    elif 'µs' in time:
        return float(time[:-3]) / 1000000
    else:
        return float(time[:-2])

def get_metrics(model_name, task, compute, server, instance_cost):
    f = open(f"./benchmark_results/raw/{model_name}/{compute}/{server}.txt")
    path_processed = f"./benchmark_results/processed/{model_name}.csv"
    path_plot = f"./benchmark_results/plots/{model_name}/{compute}/{server}.csv"
    text = f.readlines()
    result_dict = {}
    max_rps = 0
    text_list = [i.split() for i in text]
    for li in text_list:
        if len(li) > 0:
            if (li[0] == 'Error' and len(li) > 2) or li[0] == "429": 
                break
            if li[0] == 'Requests':
                num = int(li[4][:-1])
                if result_dict.get(num) is None:
                    result_dict[num] = {'latency': 0, 'throughput': float(li[6]),
                                        'count': 1, 'duration': 0}
                else:
                    result_dict[num]['count'] += 1
                    result_dict[num]['throughput'] += float(li[6])
                max_rps = num
            if li[0] == 'Duration':
                result_dict[max_rps]['duration'] += convert_to_seconds(li[4])
            if li[0] == 'Latencies':
                result_dict[max_rps]['latency'] += convert_to_seconds(li[11])
    keys_to_modify = ['latency', 'duration', 'throughput']

    for num_req in result_dict.keys():
        for key in keys_to_modify:
            result_dict[num_req][key] = result_dict[num_req][key] / result_dict[num_req]['count']

    requests = list(result_dict.keys())
    latencies = [result_dict[num_req]['latency'] for num_req in result_dict.keys()]
    throughputs = [result_dict[num_req]['throughput'] for num_req in result_dict.keys()]
    durations = [result_dict[num_req]['duration'] for num_req in result_dict.keys()]

    save_data_for_final_table(path_processed, task, [compute, server, max_rps, result_dict[max_rps]['latency'], 
                                                     result_dict[max_rps]['throughput'], 
                                                     result_dict[max_rps]['duration']], instance_cost)
    save_data_for_final_plot(path_plot, requests, latencies, throughputs, durations)

if __name__ == '__main__':
    
    model_name = sys.argv[1].split('/')[1]
    task = sys.argv[2]
    compute = sys.argv[3]
    server = sys.argv[4]
    instance_cost = float(sys.argv[5])
    get_metrics(model_name, task, compute, server, instance_cost)