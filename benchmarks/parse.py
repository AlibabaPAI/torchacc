import argparse
import csv
import os
import re
import matplotlib.pyplot as plt
from adjustText import adjust_text

def parse_log_files(directory):
    data = {}

    pattern = r'\[.*? - INFO\] \[BENCHMARK\] throughput: ([\d.]+) samples/s, max memory usage: ([\d.]+) GB'

    for filename in os.listdir(directory):
        if filename.endswith('.log'):
            backend, model = filename[:-4].split('_')
            if backend not in data:
                data[backend] = {}
            if model not in data[backend]:
                data[backend][model] = {'throughput': [], 'memory_usage': []}

            with open(os.path.join(directory, filename), 'r') as file:
                for line in file:
                    match = re.search(pattern, line)
                    if match:
                        throughput = float(match.group(1))
                        memory_usage = float(match.group(2))
                        data[backend][model]['throughput'].append(throughput)
                        data[backend][model]['memory_usage'].append(memory_usage)

    return data

def write_to_csv(data, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['backend', 'model', 'throughput', 'max_memory_usage'])
        for backend in data:
            models = sorted(data[backend].keys())
            for model in models:
                throughput = data[backend][model]['throughput'][-1]
                memory_usage = data[backend][model]['memory_usage'][-1]
                writer.writerow([backend, model, throughput, memory_usage])

def plot_data(data, png_file):
    models = set()
    backends = sorted(list(data.keys()))

    for backend in data:
        models.update(data[backend].keys())
    models = sorted(models)

    fig, axs = plt.subplots(2, 1, figsize=(15, 24), sharex=False, gridspec_kw={'height_ratios': [1.5, 1.5], 'hspace': 0.2})

    bar_width = 0.15
    index = range(len(models))

    for i, backend in enumerate(backends):
        throughputs = [data[backend][model]['throughput'][-1] for model in models]
        axs[0].bar([pos + bar_width * i for pos in index], throughputs, bar_width, label=backend)
    
    axs[0].set_title('Throughput', fontsize=22)
    axs[0].set_ylabel('Throughput (samples/s)', fontsize=20)
    axs[0].tick_params(axis='y', labelsize=18)
    axs[0].set_xticks([pos + bar_width * (len(backends) - 1) / 2 for pos in index])
    axs[0].set_xticklabels(models, rotation=0, ha='center', fontsize=18)
    axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=18)

    if 'cuda' in backends:
        texts = []
        cuda_throughputs = [data['cuda'][model]['throughput'][-1] for model in models]
        for i, backend in enumerate(backends):
            if backend != 'cuda':
                throughputs = [data[backend][model]['throughput'][-1] for model in models]
                for j, (throughput, cuda_throughput) in enumerate(zip(throughputs, cuda_throughputs)):
                    speedup = throughput / cuda_throughput
                    text = axs[0].text(index[j] + bar_width * i, throughput + 0.01, f'{speedup:.2f}x', ha='center', va='bottom', fontsize=16)
                    texts.append(text)
            else:
                throughputs = [data[backend][model]['throughput'][-1] for model in models]
                for j in range(len(models)):
                    text = axs[0].text(index[j] + bar_width * i, throughputs[j] + 0.01, f'{1.0:.2f}x', ha='center', va='bottom', fontsize=16)
                    texts.append(text)
        adjust_text(texts, ax=axs[0])

    texts = []
    for i, backend in enumerate(backends):
        memory_usages = [data[backend][model]['memory_usage'][-1] for model in models]
        bars = axs[1].bar([pos + bar_width * i for pos in index], memory_usages, bar_width, label=backend)
        
        for bar, memory_usage in zip(bars, memory_usages):
            height = bar.get_height()
            text = axs[1].text(bar.get_x() + bar.get_width() / 2, height + 0.05, f'{memory_usage:.2f} GB', ha='center', va='bottom', fontsize=16)
            texts.append(text)

    axs[1].set_title('Max Memory Usage', fontsize=22)
    axs[1].set_ylabel('Memory Usage (GB)', fontsize=20)
    axs[1].tick_params(axis='y', labelsize=18)
    axs[1].set_xticks([pos + bar_width * (len(backends) - 1) / 2 for pos in index])
    axs[1].set_xticklabels(models, rotation=0, ha='center', fontsize=18)
    axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=18)

    adjust_text(texts, ax=axs[1])

    fig.savefig(png_file, bbox_inches='tight')

    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse log files and plot throughput and memory usage.')
    parser.add_argument('directory', type=str, help='Directory containing log files')
    parser.add_argument('--png-file', type=str, default='./output.png', help='PNG file to save the plots')
    parser.add_argument('--csv-file', type=str, default='./output.csv', help='CSV file to save the parsed data')
    args = parser.parse_args()

    data = parse_log_files(args.directory)
    plot_data(data, args.png_file)
    write_to_csv(data, args.csv_file)