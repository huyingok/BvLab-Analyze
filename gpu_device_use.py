# -*- coding: utf-8 -*-
import tifffile
import os
# 必须在任何 import torch 之前执行
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import torch
import subprocess
import psutil
import multiprocessing as mp
import argparse


def max_safe_workers(mem_ratio=0.8, single_mem_MB=None):
    """
    返回当前机器上“最稳妥”的进程池大小。
    mem_ratio: 允许使用多少比例的物理内存（Default 80%）
    single_mem_MB: 你预估每个进程峰值内存（MB）。
                   如果留 None，会按 1 GB 估一个保底值。
    """
    # 1. CPU 限制
    logical = mp.cpu_count()               # 逻辑核
    physical = psutil.cpu_count(logical=False) or logical//2  # 物理核

    # 2. 内存限制
    mem = psutil.virtual_memory()
    avail_MB = mem.available / 1024 / 1024
    if single_mem_MB is None:
        single_mem_MB = 1024               # 默认按 1 GB 估
    mem_limit = int(avail_MB * mem_ratio / single_mem_MB)

    # 3. I/O 保守值（机械盘建议 ≤物理核，SSD 可放宽到逻辑核）
    io_limit = physical

    # 最终推荐
    best = min(logical, mem_limit, io_limit)
    info = dict(
        physical_cores=physical,
        logical_cores=logical,
        available_RAM_MB=int(avail_MB),
        mem_limited_workers=mem_limit,
        io_limited_workers=io_limit,
        recommended_pool_size=best
    )
    return info


def get_cpu_usage(interval=1):
    """
    获取CPU的使用情况
    :param interval: 采样间隔时间（Second）
    :return: CPU使用率（Percent）
    """
    cpu_usage = psutil.cpu_percent(interval=interval)
    return cpu_usage


def get_detailed_cpu_usage(interval=1):
    """
    获取每个CPU核心的使用情况
    :param interval: 采样间隔时间（Second）
    :return: 每个CPU核心的使用率（Percent）
    """
    cpu_usages = psutil.cpu_percent(interval=interval, percpu=True)
    return cpu_usages


def get_gpu_usage():
    # 调用 nvidia-smi 命令并获取输出
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
        capture_output=True, text=True)
    # 解析输出
    gpu_usages = [line.split(',') for line in result.stdout.strip().split('\n')]
    return gpu_usages


def is_gpu_occupied(gpu_id, threshold=30):
    """
    判断指定的GPU是否被占用
    :param gpu_id: GPU的ID
    :param threshold: 占用阈值（Percent）
    :return: True表示被占用，False表示未被占用
    """
    gpu_usages = get_gpu_usage()
    if gpu_id >= len(gpu_usages):
        raise ValueError(f"GPU ID {gpu_id} does not exist.")

    gpu_usage = gpu_usages[gpu_id]
    gpu_utilization = int(gpu_usage[0].strip())
    memory_used = int(gpu_usage[1].strip())
    memory_total = int(gpu_usage[2].strip())

    gpu_utilization2 = memory_used / memory_total * 100

    print(f"GPU {gpu_id}: "
          f"Utilization {gpu_utilization:.2f}%, "
          f"Usage {gpu_utilization2:.2f}%, "
          f"Used VRAM {memory_used}/{memory_total} MB")

    gpu_utilization = (gpu_utilization + gpu_utilization2) / 2

    return gpu_utilization > threshold


def get_gpu_utilization(threshold=50):
    # 检测是否有可用的GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # 获取可用的GPUQuantity
        num_gpus = torch.cuda.device_count()
        print(f"Available GPU Quantity: {num_gpus}")
        print(f"GPU {0}: {torch.cuda.get_device_name(0)}")
        return 0
        # for i in range(num_gpus):
        #     print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        #
        #     # 判断GPU 0是否被占用
        #     if is_gpu_occupied(i, threshold=threshold):
        #         print("GPU 0 is in use")
        #     else:
        #         print("GPU 0 is not in use")
        #         return i
        # torch.cuda.empty_cache()
    else:
        print("There is no available GPU, so the calculation has to be done using the CPU")

    return None


if __name__ == "__main__":
    # is_gpu_occupied(0)

    # Example：获取每个CPU核心的使用情况
    # cpu_usages = get_detailed_cpu_usage(interval=1)
    # for i, usage in enumerate(cpu_usages):
    #     print(f"CPU核心 {i}: 使用率 {usage}%")

    # cpu_usage = get_cpu_usage(interval=1)
    # print(f"CPU使用率: {cpu_usage}%")

    # torch.cuda.empty_cache()

    # parser = argparse.ArgumentParser()
    # parser.add_argument("-m", "--mem-per-worker", type=int,
    #                     help="预估单进程内存占用（MB）")
    # args = parser.parse_args()
    #
    # info = max_safe_workers(single_mem_MB=args.mem_per_worker)
    # for k, v in info.items():
    #     print(f"{k:25s}: {v}")

    pool_size = max_safe_workers(single_mem_MB=800)['recommended_pool_size']
    print(pool_size)
