"""CS336 naive DDP benchmark. Run from an environment with cs336_basics installed.

Default: 1 node, 2 GPUs, XL, global batch 4, sequence length 512, FP32 + AdamW.
CPU/tiny is only for checking the script; it is not the assignment measurement.
"""

import argparse
import json
import math
import os
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics.nn_utils import cross_entropy,clip_gradient
from cs336_basics.optimizer import AdamW


@dataclass(frozen=True)
class ModelConfigXL:
    vocab_size: int = 10_000
    d_model: int = 2560
    d_ff: int = 10_240
    num_layers: int = 32
    num_heads: int = 32
    context_length: int = 512
    rope_theta: float = 10_000.0


def instantiate_model(config, device):
    # 官方接口接收各个超参数，不能直接写 BasicsTransformerLM(config)。
    from cs336_basics.model import BasicsTransformerLM

        # 直接在目标设备(GPU)初始化，避免额外保存一份巨大的 CPU 模型。
    with torch.device(device):
        model = BasicsTransformerLM(**asdict(config))

    return model


def get_train_data(config, local_batch, device, rank, seed):
    # 每个 rank 使用不同的随机序列；x/y 相差一个 token。
    generator = torch.Generator(device=device).manual_seed(seed + 1000 + rank)
    tokens = torch.randint(
        config.vocab_size,
        (local_batch, config.context_length + 1),
        generator=generator,
        device=device,
    )
    return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()


class NaiveDDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.world_size = dist.get_world_size()
        self.module = module
        with torch.no_grad():
            # 冻结参数也要初始化为同一份；parameters() 自动去重共享权重。
            for parameter in self.module.parameters():
                dist.broadcast(parameter, src=0)
            # buffer 是随模型保存或管理、但不作为普通可训练参数更新的张量
            for buffer in self.module.buffers():
                dist.broadcast(buffer, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @torch.no_grad()
    def sync_gradients(self):
        # 本模型每个 rank 使用相同参数。若条件分支使某参数只在部分
        # rank 上有梯度，不能直接跳过，否则 collective 次序会不一致。
        for parameter in self.module.parameters():
            if parameter.grad is None:
                continue
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, async_op=False)
            parameter.grad.div_(self.world_size)


def synchronize(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "cpu":
        torch.cpu.synchronize()  # 是一个空操作（no-op）来自pytorch官网


def align_ranks(device):
    # 放在计时区间之外，让每轮起点尽量对齐。
    if device.type == "cuda":
        dist.barrier(device_ids=[device.index])
    else:
        dist.barrier()
    synchronize(device)


def run_training_step(ddp_model, optimizer, x, y, device):
    synchronize(device)
    start = time.perf_counter()

    optimizer.zero_grad()
    logits = ddp_model(x)
    loss = cross_entropy(logits, y)
    loss.backward()
    clip_gradient(ddp_model.parameters(), 1.0)
    synchronize(device)  # 确保反向计算已完成，避免计入通信时间。
    after_backward = time.perf_counter()

    ddp_model.sync_gradients()
    synchronize(device)  # async_op=False 仍需要等待 GPU，才能正确计时。
    after_communication = time.perf_counter()

    optimizer.step()  # 每个 rank 都用平均梯度更新自己的模型副本。
    synchronize(device)
    end = time.perf_counter()

    return {
        "step_s": end - start,
        "grad_sync_s": after_communication - after_backward,
        "forward_backward_s": after_backward - start,
        "optimizer_s": end - after_communication,
        # .item() 位于计时之外。
        "loss": loss.detach().item(),
    }


SAMPLE_COLUMNS = ["step_s", "grad_sync_s", "forward_backward_s", "optimizer_s"]


def summarize_samples(all_samples):
    """Input [rank, iteration, 4]; columns are listed in SAMPLE_COLUMNS."""
    samples = torch.as_tensor(all_samples, dtype=torch.float64).cpu()
    if samples.ndim != 3 or samples.shape[2] != 4 or min(samples.shape[:2]) < 1:
        raise ValueError("Expected timing samples of shape [rank, iteration, 4].")
    if not torch.isfinite(samples).all() or (samples < 0).any():
        raise ValueError("Timings must be finite and nonnegative.")
    if (samples[:, :, 0] <= 0).any() or (samples[:, :, 1] > samples[:, :, 0]).any():
        raise ValueError("Total time must be positive and at least gradient sync time.")

    # 每轮选择总耗时最长的 rank，并取该 rank 的所有阶段数据。
    # 不把不同 rank 的 max(comm) / max(step) 混在一起。
    slowest = samples[:, :, 0].argmax(dim=0)
    paired = samples[slowest, torch.arange(samples.shape[1])]
    step = paired[:, 0].tolist()
    comm = paired[:, 1].tolist()
    return {
        "mean_step_s": statistics.mean(step),
        "median_step_s": statistics.median(step),
        "mean_grad_sync_s": statistics.mean(comm),
        "median_grad_sync_s": statistics.median(comm),
        "grad_sync_fraction": sum(comm) / sum(step),
        "mean_forward_backward_s": paired[:, 2].mean().item(),
        "mean_optimizer_s": paired[:, 3].mean().item(),
        "slowest_rank_per_iteration": slowest.tolist(),
        "per_rank": [
            {
                "rank": rank,
                "mean_step_s": row[:, 0].mean().item(),
                "mean_grad_sync_s": row[:, 1].mean().item(),
                "grad_sync_fraction": (row[:, 1].sum() / row[:, 0].sum()).item(),
            }
            for rank, row in enumerate(samples)
        ],
    }


def worker(rank, args):
    print(f"Running basic DDP example on rank {rank}.")
    device = torch.device(f"cuda:{rank}" if args.device == "cuda" else "cpu") # device(type='cuda', index=0)
    if device.type == "cuda":
        torch.cuda.set_device(device)  # Set the current device,把当前进程和具体的物理 GPU 焊死的动作
    else:
        torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed + rank)
    # 明确 FP32 matmul 设置，便于复现实验。
    torch.set_float32_matmul_precision("highest")
    dist.init_process_group(  #加载nccl
        backend="nccl" if device.type == "cuda" else "gloo",
        rank=rank,
        world_size=args.world_size,
        timeout=timedelta(minutes=30),
    )
    try:
        config = ModelConfigXL() if args.model == "xl" else ModelConfigXL(
            vocab_size=128, d_model=32, d_ff=64,
            num_layers=2, num_heads=4, context_length=16,
        )
        dtype = getattr(torch, args.dtype)
        local_batch = args.global_batch_size // args.world_size
        model = NaiveDDP(instantiate_model(config, device)).train()
        parameters = [p for p in model.parameters() if p.requires_grad]
        if args.optimizer == "adamw":
            optimizer = AdamW(model.parameters(),lr=args.lr)
        else:
            optimizer = torch.optim.SGD(
                parameters, lr=args.lr, weight_decay=args.weight_decay,
                foreach=False,
            )
        x, y = get_train_data(config, local_batch, device, rank, args.seed)
        if rank == 0:
            n_params = sum(p.numel() for p in model.parameters())
            print(
                f"model={args.model}, parameters={n_params:,}, "
                f"world_size={args.world_size}, global_batch={args.global_batch_size}, "
                f"local_batch={local_batch}, dtype={args.dtype}, "
                f"optimizer={args.optimizer}", flush=True,
            )
            print(f"Warming up for {args.warmup} full training steps...", flush=True)

        for _ in range(args.warmup):
            align_ranks(device)
            result = run_training_step(model, optimizer, x, y, device)
            if not math.isfinite(result["loss"]):
                raise RuntimeError("Non-finite loss during warmup.")

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if rank == 0:
            print(f"Measuring {args.steps} training steps...", flush=True)
        samples = []
        losses = []
        for _ in range(args.steps):
            align_ranks(device)
            result = run_training_step(model, optimizer, x, y, device)
            if not math.isfinite(result["loss"]):
                raise RuntimeError("Non-finite loss during measurement.")
            samples.append([result[key] for key in SAMPLE_COLUMNS])
            losses.append(result["loss"])

        # 全部计时结束后才收集结果，all_gather 自身不计入训练时间。
        local_samples = torch.tensor(samples, dtype=torch.float64, device=device)
        gathered = [torch.empty_like(local_samples) for _ in range(args.world_size)]
        dist.all_gather(gathered, local_samples)
        rank_info = {"rank": rank, "device": str(device), "losses": losses}
        if device.type == "cuda":
            properties = torch.cuda.get_device_properties(device)
            rank_info.update(
                gpu_name=properties.name,
                total_memory_bytes=properties.total_memory,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            )
        hardware = [None] * args.world_size
        dist.all_gather_object(hardware, rank_info)

        if rank == 0:
            all_samples = torch.stack(gathered).cpu()
            summary = summarize_samples(all_samples)
            report = {
                "settings": vars(args),
                "model_config": asdict(config),
                "local_batch_size": local_batch,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "parameter_bytes_per_rank": sum(p.numel() * p.element_size() for p in model.parameters()),
                "gradient_bytes_per_rank": sum(p.numel() * p.element_size() for p in parameters),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "nccl_version": torch.cuda.nccl.version() if device.type == "cuda" else None,
                "hardware": hardware,
                "measurement": {
                    "clock": "time.perf_counter with CUDA synchronization at stage boundaries",
                    "input": "preallocated synthetic next-token batch; distinct per rank; reused each step",
                    "grad_sync_scope": "per-parameter all-reduce, peer waiting, division by world size, synchronization overhead",
                    "excluded": "initialization, input creation, warmup, pre-step barrier, result collection, printing",
                    "aggregation": "slowest total-step rank per iteration; use that same rank for all stages",
                    "fraction": "sum(paired gradient sync times) / sum(paired total step times)",
                    "precision": "full parameter/gradient dtype; no autocast; FP32 matmul precision highest",
                },
                "summary": summary,
                "sample_columns": SAMPLE_COLUMNS,
                "samples_by_rank": all_samples.tolist(),
            }
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
            print(f"Mean step:       {summary['mean_step_s'] * 1e3:.3f} ms")
            print(f"Median step:     {summary['median_step_s'] * 1e3:.3f} ms")
            print(f"Mean grad sync:  {summary['mean_grad_sync_s'] * 1e3:.3f} ms")
            print(f"Grad sync / step: {summary['grad_sync_fraction']:.2%}")
            print(f"Results saved to {output.resolve()}", flush=True)
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--model", choices=["xl", "tiny"], default="xl")
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--output", default="naive_ddp_results.json")
    args = parser.parse_args()
    if args.world_size < 1 or args.global_batch_size < 1 or args.global_batch_size % args.world_size:
        parser.error("global-batch-size must be positive and divisible by world-size.")
    if args.steps < 1 or args.warmup < 1 or args.cpu_threads < 1:
        parser.error("steps, warmup, and cpu-threads must be positive.")
    if args.device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() < args.world_size:
            parser.error("Not enough visible GPUs. For a CPU check use --device cpu --model tiny.")
    elif args.model != "tiny":
        parser.error("CPU mode is restricted to --model tiny for a quick check.")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(args.master_port)
    mp.spawn(worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()




