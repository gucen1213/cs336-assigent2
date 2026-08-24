import torch
import math
import triton
import triton.language as tl

# Q: torch.Size([4, 128, 64])
class MyFlashAttnAutogradFunctionClass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # 输入(Batch, Heads, SeqLen, HeadDim)
        B, N, d = Q.shape

        Bq = 16
        Bk = 16

        O = torch.zeros_like(Q)
        L = torch.zeros((B, N, 1),device=Q.device, dtype=Q.dtype)

        scale = 1.0 / math.sqrt(d)
        # 外层围着Bq维度切开
        for i in range(0, N, Bq):
            Qi = Q[:, i:i+Bq, :] #(B, H, B_q, d)

            Mi = torch.full((B, Bq, 1),float('-inf'),device=Q.device)
            Li = torch.zeros((B, Bq, 1),device=Q.device)
            Oi = torch.zeros((B, Bq, d),device=Q.device)

            for j in range(0, N, Bk):
                Kj = K[:, j:j+Bk, :]
                Vj = V[:, j:j+Bk, :]

                Sij = (Qi @ Kj.transpose(-2,-1)) * scale

                Mij = torch.maximum(Mi, torch.max(Sij, dim=-1, keepdim=True).values)
                Pij = torch.exp(Sij - Mij) # [:,Bq,Bk]

                alpha = torch.exp(Mi - Mij) # 校准因子
                Li = Li * alpha + torch.sum(Pij,dim=-1,keepdim=True)
                Oi = Pij @ Vj + Oi * alpha
                Mi = Mij  #更新最大值

            Oi = Oi / Li #Softmax 归一化
            Li = Mi + torch.log(Li)
            # 将处理好的块写回 Global Memory
            O[:, i:i+Bq, :] = Oi
            L[:, i:i+Bq, :] = Li

        ctx.save_for_backward(L.squeeze(-1), Q, K, V, O)
        return O

    @staticmethod
    def backward(ctx, dO):
        # 题目明确说明：暂时只需让它 raise NotImplementedError
        raise NotImplementedError

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,                     # 张量首地址指针
    stride_qb, stride_qq, stride_qd,  # stride_qd=1：对应内存维度。stride_qq = d 需要跳到下一个q。 stride_qb = N * d:跳到下一个batch
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,               # L 通常是 (Batch, SeqLen)
    N_QUERIES, N_KEYS,                  # 序列长度，缩放因子，维度
    scale, 
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # 1. 解析 Grid 并锁定身份 (取代了 PyTorch 的外层循环 i)
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,  #基地址
        shape=(N_QUERIES, D),  # N_QUERIES：序列总长度。剥离掉 Batch 之后，完整的 2D 数据表有多大。Triton 编译器在后台会用这个全局形状来自动做越界检查
        strides=(stride_qq, stride_qd),   # 物理内存步长：在这个 (N_QUERIES, D) 的 2D 表格里垂直方向和水平方向走的步长。
        offsets=(query_tile_index * Q_TILE_SIZE, 0), # 当前块的起始偏移量
        block_shape=(Q_TILE_SIZE, D), # 当前加载的块大小
        order=(1, 0), # 内存连续性布局，维度0是连续的既D
    )
    # K 和 V 从第 0 行开始，准备在循环中推进
    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0), # 因为我们要遍历整个序列的 K 和 V，所以指针从第 0 行开始，准备在循环里用 tl.advance 往下走。
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # 在超高速 SRAM 中初始化累加器
    Mi = tl.zeros([Q_TILE_SIZE], dtype=tl.float32) - float('inf')
    Li = tl.zeros([Q_TILE_SIZE], dtype=tl.float32)
    Oi = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32) # 即 PyTorch 里的 Oi

    Qi = tl.load(Q_block_ptr)

    # 单一内层循环 (取代了 PyTorch 的内层循环 j)
    for k_idx in range(0, N_KEYS, K_TILE_SIZE):
        Kj = tl.load(K_block_ptr)
        Vj = tl.load(V_block_ptr)

        # 核心运算：计算 Sij = Q @ K^T
        # tl.trans 会在加载后转置 K，从而满足点积维度
        Sij = tl.dot(Qi, tl.trans(Kj)) * scale

        Mij = tl.maximum(Mi, tl.max(Sij, axis=1))
        Pij = tl.exp(Sij - Mij[:, None])
        
        alpha = tl.exp(Mi - Mij)

        # 缩放历史结果并累加
        Li = Li * alpha + tl.sum(Pij, axis=1)
        Oi = Oi * alpha[:, None] + tl.dot(Pij, Vj)

        Mi = Mij

        # 推进指针，读取下一个 Block
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    # 6. 后处理归一化
    Oi = Oi / Li[:, None]
    Li = Mi + tl.log(Li)

    # 7. 写回 Global Memory
    O_block_ptr = tl.make_block_ptr(
        base=O_ptr+ batch_index * stride_ob,
        shape=(N_QUERIES, D), 
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D), 
        order=(1, 0)
    )
    # 将 FP32 的 O 强转回输入精度 (如 FP16) 写入
    tl.store(O_block_ptr, Oi.to(O_block_ptr.type.element_ty)) 
    
    l_offset = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE) #tl.arange 生成这批地址
    l_ptrs = L_ptr + batch_index * stride_lb + l_offset * stride_lq

    tl.store(l_ptrs, Li)


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # 1. 解析输入维度
        B, N, d = Q.shape
        
        # 2. 在 PyTorch 端“开辟显存”来存放输出
        # Triton 自身是不能凭空创建张量的，必须由 PyTorch 准备好空盘子
        O = torch.empty_like(Q)
        L = torch.empty((B, N), device=Q.device, dtype=torch.float32)
        
        # 3. 定义分块大小 (对应你内核里的 constexpr)
        BLOCK_Q = 16
        BLOCK_K = 16
        scale = 1.0 / math.sqrt(d)

        grid = (triton.cdiv(N, BLOCK_Q), B)
        
        # 5. 点火发射！(启动 Triton Kernel)
        # 注意这里的传参顺序，必须和 flash_fwd_kernel 的签名一模一样
        flash_fwd_kernel[grid](
            Q, K, V, 
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2), # stride_qb, stride_qq, stride_qd
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),              # L 是 2D 张量，只有 2 个 stride
            N, N,                                  # N_QUERIES, N_KEYS
            scale,
            D=d, 
            Q_TILE_SIZE=BLOCK_Q, 
            K_TILE_SIZE=BLOCK_K
        )

        # 6. 保存反向传播需要的变量 (Autograd 终于出场了)
        ctx.save_for_backward(L, Q, K, V, O)
        return O

    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError