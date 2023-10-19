import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

NAMES = {"tgi":"Text Generation Inference", "vllm": "vLLM", "ray": "Ray"}

def build_plot(title, csv_paths_with_titles):
    
    plot_file_name = '_'.join(title.lower().split())
    blue_color = "#012970"
    everything_else = "#BA9D46"

    colors = ["#F2BE22", "red", "green", "white"]
    
    plt.figure(facecolor=blue_color)
    ax = plt.axes()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    for i in range(len(csv_paths_with_titles)):
        dataframe = pd.read_csv(csv_paths_with_titles[i][0])
        x_data = dataframe['number of requests']
        y_data = dataframe['latency']
        ax.scatter(x_data, y_data, marker='o', color=colors[i], alpha=1, edgecolor='white', linewidth=1.5, zorder=2, label=csv_paths_with_titles[i][1])
        ax.plot(x_data, y_data, linestyle='-', color=colors[i], zorder=1)

    ax.set_facecolor(blue_color)
    plt.grid(axis='y', linestyle='-', color=everything_else, alpha=0.7)
    ax.yaxis.set_label_coords(-0.09, 0.2)
    ax.xaxis.set_label_coords(-0.09, -0.11)

    ax.tick_params(axis='x', colors=everything_else)
    ax.tick_params(axis='y', colors=everything_else)
    ax.spines['bottom'].set_color(everything_else)
    ax.spines['left'].set_color(everything_else)
    title_font = {'fontname': 'Arial', 'color': colors[0], 'size': 12}

    plt.title(title, fontdict=title_font, fontweight='bold', pad=18)
    plt.xlabel("Requests per second", fontdict=title_font, color=everything_else)
    plt.ylabel("Latency (s)", fontdict=title_font, color=everything_else)
    
    plt.legend(loc='best')
    plt.savefig(f"./{plot_file_name}.png")
    plt.show()

def list_files_in_folder(folder_path):
    filenames = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    csv_filenames = [f for f in filenames if f.endswith('.csv')]
    result = [(os.path.join(folder_path, f), NAMES[f[:-4]]) for f in csv_filenames]
    
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name")
    parser.add_argument("--compute")
    parser.add_argument("--title")

    args = parser.parse_args()
    folder_path = f"./{args.model_name}/{args.compute}"
    files_and_titles = list_files_in_folder(folder_path)

    build_plot(args.title, files_and_titles)
    print(list_files_in_folder(folder_path))