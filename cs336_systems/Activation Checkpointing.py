import torch
from torch.utils.checkpoint import checkpoint
from cs336_basics.model import RotaryEmbedding, TransformerBlock
from cs336_basics.nn_utils import cross_entropy, clip_gradient

d_model, d_ff, num_heads, context_length = 2560, 10240, 16, 2048
block = TransformerBlock(d_model=d_model, d_ff=d_ff, num_heads=num_heads,
positional_encoder=RotaryEmbedding(dim=d_model // num_heads, context_length=context_length))

# Fuse as much torch.compile will allow
block = torch.compile(block, fullgraph=True).to("mps")
print("model compile finish!!")

x = torch.randn((4, context_length, d_model), requires_grad=True).to("mps")
y = torch.randint(low=0, high=2560, size=(4, context_length), dtype=torch.long).to("mps")

total_size_bytes = 0
def pack_hook(t):
    if isinstance(t, torch.nn.Parameter): # Skip logging parameters to avoid double counting
        return t
    global total_size_bytes
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    total_size_bytes += t.numel() * t.element_size()
    print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
    return t

def unpack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Loading residual: {shape=}, {dtype=}, {grad_fn=}")
    
    if torch.backends.mps.is_available():
        current_mem = torch.mps.driver_allocated_memory() / (1024 ** 2)
        print(f"🔥 [反向传播进行中] 瞬间显存: {current_mem:.2f} MiB")
    
    return t


def two_blocks(x):
    x = block(x)
    x = block(x)
    return x

def four_blocks(x):
    x = block(x)
    x = block(x)
    x = block(x)
    x = block(x)
    return x

class MemoryProfiler:
    def __enter__(self):
        # 在代码块开始前，清空之前的显存峰值记录，重新开始统计
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        elif torch.backends.mps.is_available():
            # 针对本地 Mac Apple Silicon 芯片的清空指令
            torch.mps.empty_cache()
            
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 代码块执行完毕后，抓取最高峰值
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"🔥 [CUDA] Peak Memory: {peak_memory:.2f} MiB")
        elif torch.backends.mps.is_available():
            # 获取 Mac GPU 分配的显存
            peak_memory = torch.mps.driver_allocated_memory() / (1024 ** 2)
            print(f"🍏 [MPS] Peak Memory (Current Allocation): {peak_memory:.2f} MiB")
        else:
            print("⚠️ 未检测到 GPU，运行在 CPU 模式")


def four_blocks_checkpoint(x):
    # checkpoint throws out all the saved tensors until the backward pass
    # when getting to the checkpointed block in the backward pass,
    # it reruns a forward pass to produce the saved tensors,
    # then completes normal backward pass.
    x = checkpoint(two_blocks, x, use_reentrant=False)
    x = checkpoint(two_blocks, x, use_reentrant=False)
    return x

def recursive_block(x):
    def attn_forward(h):
        return block.attn(block.ln1(h))

    x_attn =checkpoint(attn_forward, x, use_reentrant=False)
    x = x + x_attn

    def ffn_forward(h):
        return block.ffn(block.ln2(h))

    x_ffn = checkpoint(ffn_forward, x, use_reentrant=False)
    x = x + x_ffn
    return x

def recursive_blocks(x, num):
    for _ in range(num):
        x = checkpoint(recursive_block, x, use_reentrant=False)
    return x

def checkpoint_blocks_forward(x, num):
    for _ in range(num):
        x = checkpoint(block, x, use_reentrant=False)  # 80M * num
    return x

# with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
#     x = checkpoint_blocks_forward(x,4)
#     loss = cross_entropy(x, y)
#     loss.backward()

with MemoryProfiler(), torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y_hat = checkpoint_blocks_forward(x, 4)
    loss = cross_entropy(y_hat, y)
    loss.backward()

# with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
#     x = block(x)

print(f"Total size of saved tensors in single TransformerBlock: {total_size_bytes /(1024**2):.2f} MiB")

# step1
# Saving residual: shape=torch.Size([4, 2048, 2560]), dtype=torch.float32, grad_fn=None  #输入
# Saving residual: shape=torch.Size([4, 2048, 1]), dtype=torch.float32, grad_fn=None  #“均方根倒数”（1/RMS(X)）
# Saving residual: shape=torch.Size([2048, 80]), dtype=torch.float32, grad_fn=None  #Rope, dk/2 = 80
# Saving residual: shape=torch.Size([2048, 80]), dtype=torch.float32, grad_fn=None
# Saving residual: shape=torch.Size([64, 2048, 2048]), dtype=torch.float32, grad_fn=None  # N*N [4, 16, 2048, 160] × [4, 16, 160, 2048]
# Saving residual: shape=torch.Size([4, 16, 2048, 1]), dtype=torch.float32, grad_fn=None # softmax中的max
# Saving residual: shape=torch.Size([4, 16, 2048, 1]), dtype=torch.int64, grad_fn=None  # 最大值的所在位置（argmax indices），Autograd 引擎为了给 max 算子求导而机械保存的“路标”。最大值的所在位置（argmax indices），Autograd 引擎为了给 max 算子求导而机械保存的“路标”。
# Saving residual: shape=torch.Size([4, 16, 2048, 1]), dtype=torch.float32, grad_fn=None #softmax求和的分母
# Saving residual: shape=torch.Size([4, 16, 2048, 2048]), dtype=torch.float32, grad_fn=None # softmax输出的P矩阵
# Saving residual: shape=torch.Size([1, 8192, 2560]), dtype=torch.float32, grad_fn=None
# Saving residual: shape=torch.Size([4, 2048, 1]), dtype=torch.float32, grad_fn=None
# Saving residual: shape=torch.Size([1, 8192, 10240]), dtype=torch.float32, grad_fn=None
# Saving residual: shape=torch.Size([1, 8192, 10240]), dtype=torch.float32, grad_fn=None
# Saving residual: shape=torch.Size([1, 10240, 8192]), dtype=torch.float32, grad_fn=None
# Saving residual: shape=torch.Size([1, 2560, 8192]), dtype=torch.float32, grad_fn=None
# Saving residual: shape=torch.Size([1, 2560, 8192]), dtype=torch.float32, grad_fn=None
# Saving residual: shape=torch.Size([64, 160, 2048]), dtype=torch.float32, grad_fn=None
# Saving residual: shape=torch.Size([64, 160, 2048]), dtype=torch.float32, grad_fn=None
# Saving residual: shape=torch.Size([64, 2048, 160]), dtype=torch.float32, grad_fn=None
# Saving residual: shape=torch.Size([1, 2560, 8192]), dtype=torch.float32, grad_fn=None
# Total size of saved tensors in single TransformerBlock: 3651.31 MiB