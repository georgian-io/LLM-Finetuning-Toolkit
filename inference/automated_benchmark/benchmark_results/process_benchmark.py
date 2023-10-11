import csv
import os
import sys
    
def save_data_for_final_table(csv_file_path, task, data, instance_cost):
    print(csv_file_path, task, data, instance_cost)
    headers = ["compute", "server", "rps", "latency", "throughput", "cost"]
    num_of_tokens = 120 if task == "summarization" else 101
    data.append(instance_cost / (data[2] * 3600 * num_of_tokens) * 1000) # calculating price per 1K tokens 
    if not os.path.isfile(csv_file_path) or os.path.getsize(csv_file_path) == 0:
        write_mode = 'a'  
        with open(csv_file_path, mode=write_mode, newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
    else:
        write_mode = 'a'  

    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)
        
def save_data_for_final_plot(csv_file_path, requests, latencies, throughputs):
    headers = ["number of requests", "latency", "throughput"]
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for data_row in zip(requests, latencies, throughputs):
            writer.writerow(data_row)


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
            if li[0] == 'Error' and len(li) > 2: 
                break
            if li[0] == 'Requests':
                num = int(li[4][:-1])
                if result_dict.get(num) is None:
                    result_dict[num] = [[0, float(li[6])], 1]
                else:
                    result_dict[num][1] += 1
                    result_dict[num][0][1] += float(li[6])
                max_rps = num
            if li[0] == 'Latencies':
                if 'ms' in li[11]:
                    result_dict[max_rps][0][0] += float(li[11][:-3]) / 1000
                elif 'µs' in li[11]:
                    result_dict[max_rps][0][0] += float(li[11][:-3]) / 1000000
                else:
                    result_dict[max_rps][0][0] += float(li[11][:-2])

    result_dict_ready = {key: [result_dict[key][0][0] / result_dict[key][1], result_dict[key][0][1] / result_dict[key][1]] for key in result_dict.keys()}
    requests = list(result_dict_ready.keys())
    latencies = [result_dict_ready[val][0] for val in result_dict_ready.keys()]
    throughputs = [result_dict_ready[val][1] for val in result_dict_ready.keys()]
   
    save_data_for_final_table(path_processed, task, [compute, server, max_rps, latencies[-1], 
                                                     throughputs[-1]], instance_cost)
    save_data_for_final_plot(path_plot, requests, latencies, throughputs)

if __name__ == '__main__':
    
    model_name = sys.argv[1].split('/')[1]
    task = sys.argv[2]
    compute = sys.argv[3]
    server = sys.argv[4]
    instance_cost = float(sys.argv[5])
    get_metrics(model_name, task, compute, server, instance_cost)