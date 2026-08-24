import sys
import time
import torch
import argparse
import timeit
import numpy as np
sys.path.insert(0, '../cs336-basics')
from torch.amp import autocast, GradScaler # 引入 AMP 核心组件
from torch.profiler import profile, record_function, ProfilerActivity
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import get_cosine_lr, AdamW
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy, clip_gradient

def print_mps_memory(stage=""):
    # current_allocated_memory: 当前模型 Tensor 真实正在使用的显存
    allocated = torch.mps.current_allocated_memory() / (1024 ** 2)
    # driver_allocated_memory: PyTorch 缓存池已经向 Mac 操作系统（Metal 驱动）申请的总显存
    driver = torch.mps.driver_allocated_memory() / (1024 ** 2)
    print(f"[{stage}] Tensor 真实占用: {allocated:.2f} MB | Metal 驱动已分配池: {driver:.2f} MB")


def get_train_data(vocab_size: int, batch: int, 
        context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:

    dummy_input = np.random.randint(0, vocab_size, size=(2 * context_length + 1,))

    return get_batch(dummy_input, batch, context_length, device)

def main():
    parser = argparse.ArgumentParser(description="Train a language model")
    # 训练超参数
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--warmup_iters", type=int, default=50)

    # 模型架构
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--clip_grad", type=float, default=1.0)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps")

    args = parser.parse_args()

    lsr = args.learning_rate
    vocab_size = args.vocab_size
    device = args.device
    context_length = args.context_length
    batch_size = args.batch_size
    d_model = args.d_model
    d_ff = args.d_ff
    n_layers = args.n_layers
    n_heads = args.n_heads
    clip_grad = args.clip_grad
    device = args.device

    model = BasicsTransformerLM(vocab_size, context_length, d_model, n_layers, n_heads, d_ff, 10000.0).to(device)
    print_mps_memory("加载模型权重后")

    model = torch.compile(model, backend="aot_eager")

    optimizer = AdamW(model.parameters(),lr=lsr)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params:,}")

    model.train()

    for step in range(10):
        
        new_lr = get_cosine_lr(step, args.max_lr, args.min_lr, args.warmup_iters, 5000)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

        x,y = get_train_data(vocab_size, batch_size, context_length, device)
        logits = model(x)
        print_mps_memory("前向传播（生成激活值）后")
        torch.mps.synchronize()

        loss = cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        print_mps_memory("反向传播（生成梯度）后")
        torch.mps.synchronize()

        clip_gradient(model.parameters(), clip_grad)
        optimizer.step()
        print_mps_memory("优化过程后")
        torch.mps.synchronize()

        torch.mps.empty_cache()

    print("预热结束")

    # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    #     for step in range(5):
    #         with record_function("1_lr_update"):
    #             new_lr = get_cosine_lr(step, args.max_lr, args.min_lr, args.warmup_iters, 5000)
    #             for param_group in optimizer.param_groups:
    #                 param_group['lr'] = new_lr

    #         with record_function("2_data_loading"):
    #             x,y = get_train_data(vocab_size, batch_size, context_length, device)
            
    #         with record_function("3_forward_pass"):
    #             optimizer.zero_grad(set_to_none=True)
    #             logits = model(x)
    #             torch.mps.synchronize()

    #         with record_function("4_loss_computation"):
    #             loss = cross_entropy(logits, y)
    #             optimizer.zero_grad(set_to_none=True)

    #         with record_function("5_backward_pass"):
    #             loss.backward()
    #             torch.mps.synchronize()

    #         with record_function("6_optimizer_step"):
    #             clip_gradient(model.parameters(), clip_grad)
    #             optimizer.step()
    #             torch.mps.synchronize()

    # # 循环结束后，将收集到的性能数据导出为 JSON 文件
    # prof.export_chrome_trace("ft16-t-trace.json")
    # print("✅ 火焰图数据已成功导出至 trace.json")

if __name__ == '__main__':
    main()
 