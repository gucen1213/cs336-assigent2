import os
import torch
import time
import math
import statistics
import torch.multiprocessing as mp
import torch.distributed as dist

def cleanup():
    dist.destroy_process_group()

def cuda_if_available(rank: int) -> str:
    if torch.cuda.is_available():
        return f"cuda:{rank}"
    return "cpu"

def render_duration(duration: float) -> str:
    if duration < 1e-3:
        return f"{duration * 1e6:.2f}us"
    if duration < 1:
        return f"{duration * 1e3:.2f}ms"
    return f"{duration:.2f}s"

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    else:
        torch.cpu.synchronize()

def all_reduce(rank, world_size,num_elements):
    setup(rank, world_size)
    device = torch.device(cuda_if_available(rank))

    data = torch.randn(num_elements, device=cuda_if_available(rank))
    #Warmup
    for _ in range(5):
        dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)
        synchronize(device)
        dist.barrier() # wait for all processes to get here

    durations = []
    for _ in range(30):
        data.fill_(1)
        dist.barrier()
        synchronize(device) # 让所有进程都准备好；barrier 本身不计入耗时

        start_time = time.perf_counter()
        dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)
        synchronize(device) # 确保通信实际完成

        end_time = time.perf_counter()
        durations.append(end_time - start_time)

    times = torch.tensor(durations,
                        dtype=torch.float64,
                        device=device)

    dist.reduce(times, dst=0, op=dist.ReduceOp.MAX) # 对每一轮，取所有 rank 耗时中的最大值。目标进程是0

    if rank == 0:
        samples = times.cpu().tolist()
        size_bytes = data.numel() * data.element_size()
        print(
            f"device={device.type},"
            f"world_size={world_size},"
            f"bytes_per_rank={size_bytes},"
            f"mean={render_duration(statistics.mean(samples))}, "
            f"median={render_duration(statistics.median(samples))}",
            flush=True,
        )

    cleanup()

def benchmarking(rank: int, world_size):
    all_reduce(rank,
               world_size=world_size,
               num_elements=1 * 1024**2 // 4 #1M
    )

if __name__ == "__main__":
    world_size = 2
    mp.spawn(fn=benchmarking, 
            args=(world_size,), 
            nprocs=world_size, 
            join=True
            )

