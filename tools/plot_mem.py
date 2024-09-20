import argparse
import re

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

BUFFER_ASSIGNMENT_SUFFIX = "buffer-assignment.txt"
BUFFER_LIVE_RANGE_KEYWORD = "BufferLiveRange:"
TENSOR_PATTERN = r"value: <\d+ ([^@>]+)@0> \(size\=(\d+),offset=(\d+)\):"
ALLOCATION_PATTERN = r"allocation (\d+): size (\d+)"
LIVE_RANGE_PATTERN = r"\s*(\S+):(\d+)-(\d+)\s*\n?"


def parse_args() -> argparse.Namespace:
    '''
    Parses command-line arguments and check validity.

    Returns:
      args (argparse.Namespace): Parsed command line arguments
    '''
    parser = argparse.ArgumentParser(description='''
        Plot tensor lifecycle and buffer usage, when given a buffer assignment file from XLA.
        After dumping the buffer assignment file with "--xla_dump_hlo_as_text" flag, you can use this script to visualize the buffer usage during the execution.
        The script will plot the lifecycle of each tensor, and the buffer usage over time.
        The peak memory usage will be annotated on the plot.
        
        For example:\n
        
            python plot-mem.py --input=module_0000.SyncTensorsGraph.105381.sm_8.0_gpu_after_optimizations-buffer-assignment.txt
            
        The script will analyze the buffer assignment file and generate a plot named "tensor_lifecycle.png".
    ''')
    parser.add_argument('--input',
                        type=str,
                        help='The file to be parsed',
                        required=True)
    args = parser.parse_args()
    args.input = args.input.strip()

    if not args.input.endswith(BUFFER_ASSIGNMENT_SUFFIX):
        raise ValueError(
            f"The input file should end with '{BUFFER_ASSIGNMENT_SUFFIX}'.")

    return args


def read_file(file_name: str) -> str:
    '''
    Read the file and return the content.

    Args:
      file_name (str): Path to the file to be read
      
    Returns:
      str: Content of the file
    '''
    with open(file_name, 'r', encoding='utf-8') as file:
        return file.read()


def get_all_tensors(file_content: str) -> dict[str, dict]:
    '''
    Extract information about all tensors from the file content. The pattern is:
    value: <89591 custom-call.87.0{2} @0> (size=33554432,offset=0): bf16[16384,8,128]{2,1,0}
    
    Args:
      file_content (str): Content of the file
    
    Returns:
      dict[str, dict]: Dictionary of tensors, where the key is the tensor name and the value is a dictionary containing the size, offset, and allocation_id
    '''
    pattern = TENSOR_PATTERN  # (\w+\[\d+(?:,\d+)*\])"
    allocation_pattern = ALLOCATION_PATTERN
    results = {}
    offset_lb = 0
    offset_ub = 0
    allocation_id = 0

    lines = file_content.splitlines()
    for line in lines:
        allocation_matches = re.search(allocation_pattern, line)
        if allocation_matches:
            allocation_id, size = allocation_matches.groups()
            offset_lb = offset_ub
            offset_ub = offset_lb + int(size)
        matches = re.findall(pattern, line)
        for match in matches:
            tensor_name, size, offset = match
            results[tensor_name.strip()] = {
                'size': size,
                'offset': int(offset) + offset_lb,
                'allocation_id': allocation_id
            }
    print(f"[INFO] Total {len(results)} tensors are found.")
    return results


def get_live_range(file_content: str, tensors: dict) -> tuple[dict, set]:
    '''
    Retrieves the live ranges for the tensors from the file content. Live range pattern is:
    BufferLiveRange:
        reduce-scatter.8199.1{}:2037-2091
        reduce-scatter.8708.1{}:2145-2198

    Args:
      file_content (str): Content of the file
      tensors (dict): Dictionary of tensors, where the key is the tensor name and the value is a dictionary containing the size, offset, and allocation_id
    
    Returns:
      dict: Dictionary of tensors with live range information added
      diff (set): Set of tensors that are not matched with live range pattern
    '''
    keyword = BUFFER_LIVE_RANGE_KEYWORD
    keyword_pos = file_content.find(keyword)
    if keyword_pos == -1:
        raise ValueError(f"{keyword} not found.")

    file_content = file_content[keyword_pos + len(keyword):]
    pattern = LIVE_RANGE_PATTERN
    matches = re.findall(pattern, file_content)
    i = 0
    matched_name = set()
    for match in matches:
        name, live_range_start, live_range_end = match
        if name not in tensors:
            name = name[:-2]
        if name not in tensors:
            # raise ValueError(f"Tenosr name is not exist in value list. Tensor name: {name}.")
            continue
        if name in tensors:
            tensors[name]['start'] = live_range_start
            tensors[name]['end'] = live_range_end
        matched_name.add(name)
        i += 1
    print(f"[INFO] Total {i} tensors' live range are found.")
    diff = set(tensors.keys()) - matched_name
    return tensors, diff


def get_allocation_group_by_size(allocations: dict) -> dict:
    '''
    Groups allocations by their sizes.

    Args:
      allocations (dict): Dictionary of allocations, where the key is the allocation ID and the value is a dictionary containing the size and start time
    
    Returns:
      dict: Dictionary of allocations grouped by their sizes, the key is the index and the value is a dictionary containing the size, start time and count
    '''
    allocations_group_by_size = {}
    index = 0
    sorted_allocations = sorted(allocations.items(),
                                key=lambda item: int(item[0]))

    prev_size = None

    for allocation_id, allocation_info in sorted_allocations:
        size = int(allocation_info['size'])
        start = int(allocation_info['start'])

        if size != prev_size:
            index += 1
            allocations_group_by_size[index] = {
                'size': size,
                'start': start,
                'count': 1
            }
        else:
            allocations_group_by_size[index]['count'] += 1

        prev_size = size

    return allocations_group_by_size

def plot_tensor_lifecycle(tensors: dict, allocations: dict):
    '''
    Plots the lifecycle of tensors and buffer usage over time.

    Args:
        tensors (dict): Dictionary containing tensor information.
        allocations (dict): Dictionary containing allocation information.
    '''
    fig, ax = plt.subplots(figsize=(12, 6))
    max_time = 0
    max_buffer_size = 0
    buffer_usage = {}

    for tensor_name, tensor_info in tensors.items():
        try:
            size = int(tensor_info['size'])
            offset = int(tensor_info['offset'])
            start = int(tensor_info['start'])
            end = int(tensor_info['end'])
            rect = patches.Rectangle((start, offset),
                                     end - start,
                                     size,
                                     fill=True,
                                     facecolor='skyblue',
                                     edgecolor='skyblue',
                                     alpha=0.7)
            ax.add_patch(rect)

            for t in range(start, end + 1):
                if t not in buffer_usage:
                    buffer_usage[t] = 0
                buffer_usage[t] += size

            max_time = max(max_time, end)
            max_buffer_size = max(max_buffer_size, offset + size)
        except:
            # print(f"Error: {tensor_name} has invalid info: {tensor_info}")
            pass

    allocations_group_by_size = get_allocation_group_by_size(allocations)
    for index, allocation_info in sorted(allocations_group_by_size.items(),
                                         key=lambda x: x[0]):
        size = allocation_info['size']
        start = allocation_info['start']
        count = allocation_info['count']
        # only show allocations larger than 512MB
        if size * count < 512 * 1024 * 1024:
            continue
        rect = patches.Rectangle((0, start),
                                 max_time,
                                 size * count,
                                 fill=False,
                                 edgecolor='red',
                                 linewidth=1.5,
                                 alpha=1.0)
        ax.add_patch(rect)
        ax.text(
            1.02,
            start + size * count / 2,
            f"Count: {count}\nSize: {size} bytes ({size/1024/1024/1024:.2f} GB)\nTotal: {size * count} bytes ({size * count/1024/1024/1024:.2f} GB)",
            fontsize=6,
            ha='left',
            va='center',
            transform=ax.get_yaxis_transform(),
        )

    peak_time = max(buffer_usage, key=buffer_usage.get)
    peak_usage = buffer_usage[peak_time]

    times = sorted(buffer_usage.keys())
    usages = [buffer_usage[t] for t in times]
    ax.plot(times, usages, color='Slateblue', linewidth=2, alpha=0.6)

    ax.annotate(
        f"Time: {peak_time}\nActual Usage: {peak_usage} bytes ({peak_usage/1024/1024/1024:.2f} GB)",
        xy=(peak_time, peak_usage),
        xytext=(peak_time, peak_usage + max(usages) * 0.1),
        arrowprops=dict(facecolor='black', shrink=0.05),
        fontsize=8,
        ha='center',
        va='bottom')

    ax.set_xlim(0, max_time)
    ax.set_ylim(0, max(max_buffer_size, peak_usage * 1.4))
    ax.set_xlabel('Time')
    ax.set_ylabel('Buffer Size (bytes)')
    ax.set_title('Tensor Lifecycle and Buffer Usage')

    y_ticks = ax.get_yticks()
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{y /1024 / 1024 / 1024 :.2f} GB' for y in y_ticks])

    ax.axhline(y=max_buffer_size, color='red', linestyle='--', alpha=0.7)
    ax.text(1.01,
            max_buffer_size,
            f'Allocated Size: {max_buffer_size/1024/1024/1024:.2f} GB',
            transform=ax.get_yaxis_transform(),
            ha='left',
            va='center',
            color='red',
            fontsize=8)

    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig('tensor_lifecycle.png', dpi=1200)
    print("[INFO] tensor_lifecycle.png saved successfully.")


def get_all_allocation(file_content: str) -> tuple:
    '''
    Extracts information about all allocations from the file content.
    allocation 1: size 525338624,

    Args:
        file_content (str): Content of the buffer assignment file.

    Returns:
        tuple: Dictionaries containing allocation information and total allocation size.
    '''
    pattern = ALLOCATION_PATTERN

    matches = re.findall(pattern, file_content)
    results = {}
    start = 0
    for match in matches:
        allocation_id, size = match
        results[allocation_id] = {'size': size, 'start': start}
        start += int(size)

    total = 0
    for key, value in results.items():
        total += int(value['size'])
    return results, total


if __name__ == '__main__':
    args = parse_args()
    content = read_file(args.input)

    tensors = get_all_tensors(content)
    tensors, diff = get_live_range(content, tensors)
    allocations, total = get_all_allocation(content)
    print(
        f"[INFO] Total allocation size: {total} bytes ({total/1024/1024/1024:.2f} GB)"
    )

    if diff:
        print(f"[WARNING] {len(diff)} tensors' live range are not found.")

    plot_tensor_lifecycle(tensors, allocations)
