import torch
import time
import math
import sys
import os
import torch.distributed as dist
from torch import nn, tensor
import torch.multiprocessing as mp

def cleanup():
    dist.destroy_process_group()

def cuda_if_available(rank: int) -> str:
    if torch.cuda.is_available():
        return f"cuda:{rank}"
    return "cpu"

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def render_duration(duration: float) -> str:
    if duration < 1e-3:
        return f"{duration * 1e6:.2f}us"
    if duration < 1:
        return f"{duration * 1e3:.2f}ms"
    return f"{duration:.2f}s"

def distributed_demo(rank, world_size):
    print(f"rank = {rank}, world_size = {world_size}")
    setup(rank, world_size)

    ### All-reduce 全局聚合与同步 (reduce_scatter + all_gather)
    dist.barrier() # 等待所有进程走到这
    data = torch.randint(0, 10, (3,))
    print(f"rank {rank} data (before all-reduce): {data}")
    dist.all_reduce(data, async_op=False)
    print(f"rank {rank} data (after all-reduce): {data}")
    dist.barrier()

    ### Reduce-scatter 聚合与切片分发
    dist.barrier()
    input = torch.arange(world_size,dtype=torch.float32) + rank
    output = torch.empty(1) # Allocate output
    print(f"Rank {rank} [before reduce-scatter]: input = {input}, output = {output}", flush=True)
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    print(f"Rank {rank} [after reduce-scatter]: input = {input}, output = {output}", flush=True)

    ### All-gather 全收集与拼接,让每个进程将手中的切片广播给集群内的所有人。
    dist.barrier()
    input = output
    output = torch.empty(world_size)  # Allocate output
    print(f"Rank {rank} [before all-gather]: input = {input}, output = {output}", flush=True)
    dist.all_gather_into_tensor(output_tensor=output, input_tensor=input,async_op=False)
    print(f"Rank {rank} [after all-gather]: input = {input}, output = {output}", flush=True)

    cleanup()

def all_reduce(rank: int, world_size: int, num_elements: int):
    setup(rank, world_size)

    data = torch.randn(num_elements, device=cuda_if_available(rank))

    #Warmup
    dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)
    torch.cpu.synchronize() # wait for cpu/cuda kernel to finish
    dist.barrier() # wait for all processes to get here

    start_time = time.time()
    dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)
    torch.cpu.synchronize() # wait for cpu/cuda kernel to finish
#    dist.barrier() # wait for all processes to get here
    end_time = time.time()

    duration = end_time - start_time
    print(f"[all_reduce] Rank {rank}: all_reduce(world_size={world_size},  \
        num_elements={num_elements}) took {render_duration(duration)}", flush=True)

    # Measure the effective bandwidth
    dist.barrier()
    size_bytes = data.element_size() * data.numel()
    total_traffic_bytes = size_bytes * 2 * (world_size - 1)  # How much needs to be sent (no 2x here)
    total_duration = world_size * duration
    bandwidth = total_traffic_bytes / total_duration
    print(f"[reduce_scatter] Rank {rank}: all_reduce measured bandwidth = {round(bandwidth / 1024**3)} GB/s", flush=True)
    cleanup()


def reduce_scatter(rank: int, world_size: int, num_elements: int):
    setup(rank, world_size)  
    # PyTorch 文档规定 reduce-scatter 的 input 总大小应该是 output 的 world_size 倍
    input = torch.randn((num_elements), 
                        dtype=torch.float32,
                        device=cuda_if_available(rank)
                        )  # Each rank has a matrix

    output = torch.empty((num_elements // world_size),
                        dtype=torch.float32,
                        device=cuda_if_available(rank)
                        )

    # Warmup
    dist.reduce_scatter_tensor(output=output, 
                               input=input,
                               op=dist.ReduceOp.SUM, 
                               async_op=False)
    torch.cpu.synchronize()  # Wait for cpu kernels to finish

    dist.barrier()            # Wait for all the processes to get here
    # Perform reduce-scatter
    start_time = time.time()
    dist.reduce_scatter_tensor(output=output,
                               input=input, 
                               op=dist.ReduceOp.SUM, 
                               async_op=False)
    torch.cpu.synchronize()  # Wait for CPU kernels to finish

    end_time = time.time()
    duration = end_time - start_time
    print(f"[reduce_scatter] Rank {rank}: reduce_scatter(world_size={world_size}, \
    num_elements={num_elements}) took {render_duration(duration)}", flush=True) 

    # Measure the effective bandwidth
    dist.barrier()
    data_bytes = input.element_size() * input.numel()  # How much data in the input

    sent_bytes_per_rank = (
        data_bytes * (world_size-1) / world_size
    )

    bandwidth = sent_bytes_per_rank / duration
    print(f"[reduce_scatter] Rank {rank}: reduce_scatter measured bandwidth = {round(bandwidth / 1024**3)} GB/s", flush=True)
    # Notes:
    # - all-reduce = reduce-scatter + all-gather
    # - all-reduce moves 2x the data in 2x the time compared to reduce-scatter, so similar bandwidth
    cleanup()


def benchmarking(rank, world_size):
    # How fast does communication happen?
    # All-reduce
   # all_reduce(rank,world_size=4, num_elements=100 * 1024**2)

    reduce_scatter(rank,
                   world_size=world_size, 
                   num_elements=100 * 1024**2)


if __name__ == "__main__":
    world_size = 4
    # mp.spawn(fn=func, args=(a, b), nprocs=N)
    # func(rank, a, b):
    mp.spawn(fn=benchmarking, 
             args=(world_size,), 
             nprocs=world_size, 
             join=True
             )


