import math
import torch
import time
from einops import einsum
from itertools import product
from cs336_basics.model import RotaryEmbedding, TransformerBlock, scaled_dot_product_attention

batch_size = 8
d_models = [16, 32, 64, 128]
seq_lens = [256, 1024, 4096]

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
sync_fn = torch.cuda.synchronize if torch.cuda.is_available() else torch.mps.synchronize

print("========== 第一阶段：Eager Mode (原生动态图) ==========")
for d_model, seq_len in product(d_models, seq_lens):

    print(f"\n--- Testing d_model={d_model}, seq_len={seq_len} ---")

    try:
        # 题目 (iii): 创建随机张量。注意必须加上 requires_grad=True 才能测反向传播！
        Q = torch.randn(batch_size, seq_len, d_model, requires_grad=True, device=device)
        K = torch.randn(batch_size, seq_len, d_model, requires_grad=True, device=device)
        V = torch.randn(batch_size, seq_len, d_model, requires_grad=True, device=device)
        
        # 题目 (vi): Warm up (预热) 避免冷启动误差
        for _ in range(5):
            out = scaled_dot_product_attention(Q, K, V)
            out.sum().backward()
        sync_fn()

        # 题目 (iv): 测量 100 次前向传播
        start_fwd = time.perf_counter()
        for _ in range(100):
            out = scaled_dot_product_attention(Q, K, V)
        sync_fn()
        fwd_time = time.perf_counter() - start_fwd
        print(f"Forward Time (100 passes): {fwd_time:.4f} s")
        
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            print(f"Memory before backward: {mem_allocated:.2f} MiB")
        elif torch.backends.mps.is_available():
            mem_allocated = torch.mps.current_allocated_memory() / (1024 ** 2)
            print(f"[MPS] Memory before backward: {mem_allocated:.2f} MiB")

        # 题目 (v): 测量 100 次反向传播
        # 细节陷阱：如果在循环里连续调用 100 次 backward，必须带上 retain_graph=True
        start_bwd = time.perf_counter()
        for _ in range(100):
            # 严谨起见，每次清空梯度，防止 100 次累加导致计算开销变大或显存溢出
            if Q.grad is not None:
                Q.grad = None
                K.grad = None
                V.grad = None
            out = scaled_dot_product_attention(Q, K, V)
            out.sum().backward()

        sync_fn()
        total_time = time.perf_counter() - start_bwd
        bwd_time = total_time - fwd_time
        print(f"Backward Time (100 passes): {bwd_time:.4f} s")

        if torch.cuda.is_available():
            mem_after_100 = torch.cuda.memory_allocated() / (1024 ** 2)
            print(f"Memory AFTER 100 backwards (Activations + Gradients): {mem_after_100:.2f} MiB")
        elif torch.backends.mps.is_available():
            mem_after_100 = torch.mps.current_allocated_memory() / (1024 ** 2)
            print(f"🍏 [MPS] Memory AFTER 100 backwards: {mem_after_100:.2f} MiB")

        # 测试完毕清空显存，防止影响下一轮循环
        del Q, K, V, out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("🚨 状态: Out of Memory (OOM)")
            if torch.cuda.is_available():
                torch.cuda.empty_cache() # 发生 OOM 后必须手动清空显存救场
        else:
            raise e

print("============ 第二阶段:带上torch.compile======================")
compiled_attention = torch.compile(scaled_dot_product_attention)

for d_model, seq_len in product(d_models, seq_lens):

    print(f"\n--- Testing d_model={d_model}, seq_len={seq_len} ---")

    try:
        # 题目 (iii): 创建随机张量。注意必须加上 requires_grad=True 才能测反向传播！
        Q = torch.randn(batch_size, seq_len, d_model, requires_grad=True, device=device)
        K = torch.randn(batch_size, seq_len, d_model, requires_grad=True, device=device)
        V = torch.randn(batch_size, seq_len, d_model, requires_grad=True, device=device)
        
        # 题目 (vi): Warm up (预热) 避免冷启动误差
        for _ in range(5):
            out = compiled_attention(Q, K, V)
            out.sum().backward()
        sync_fn()

        # 题目 (iv): 测量 100 次前向传播
        start_fwd = time.perf_counter()
        for _ in range(100):
            out = compiled_attention(Q, K, V)
        sync_fn()
        fwd_time = time.perf_counter() - start_fwd
        print(f"Forward Time (100 passes): {fwd_time:.4f} s")
        
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            print(f"Memory before backward: {mem_allocated:.2f} MiB")
        elif torch.backends.mps.is_available():
            mem_allocated = torch.mps.current_allocated_memory() / (1024 ** 2)
            print(f"[MPS] Memory before backward: {mem_allocated:.2f} MiB")

        # 题目 (v): 测量 100 次反向传播
        # 细节陷阱：如果在循环里连续调用 100 次 backward，必须带上 retain_graph=True
        start_bwd = time.perf_counter()
        for _ in range(100):
            # 严谨起见，每次清空梯度，防止 100 次累加导致计算开销变大或显存溢出
            if Q.grad is not None:
                Q.grad = None
                K.grad = None
                V.grad = None
            out = compiled_attention(Q, K, V)
            out.sum().backward()

        sync_fn()
        total_time = time.perf_counter() - start_bwd
        bwd_time = total_time - fwd_time
        print(f"Backward Time (100 passes): {bwd_time:.4f} s")

        if torch.cuda.is_available():
            mem_after_100 = torch.cuda.memory_allocated() / (1024 ** 2)
            print(f"Memory AFTER 100 backwards (Activations + Gradients): {mem_after_100:.2f} MiB")
        elif torch.backends.mps.is_available():
            mem_after_100 = torch.mps.current_allocated_memory() / (1024 ** 2)
            print(f"🍏 [MPS] Memory AFTER 100 backwards: {mem_after_100:.2f} MiB")

        # 测试完毕清空显存，防止影响下一轮循环
        del Q, K, V, out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("🚨 状态: Out of Memory (OOM)")
            if torch.cuda.is_available():
                torch.cuda.empty_cache() # 发生 OOM 后必须手动清空显存救场
        else:
            raise e
