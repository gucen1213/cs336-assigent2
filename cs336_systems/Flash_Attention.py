import torch
import math
# import triton
# import triton.language as tl

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
            Qi = Q[:, i:i+Bq, :] 

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
    def backward(ctx,dO):
        L,Q,K,V,O = ctx.saved_tensors
        B, N, d = Q.shape
        Bq, Bk = 16, 16
        scale = 1.0 / math.sqrt(d)
        # 提前计算全局 D 向量
        # dO 和 O 逐元素相乘，并在特征维度 d 上求和，保持维度以便广播 (B, N, 1)
        D = (dO * O).sum(dim=-1, keepdim=True)

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        for j in range(0, N, Bk):
            Kj = K[:, j:j+Bk, :]
            Vj = V[:, j:j+Bk, :]
            # 当前j块建立局部梯度累加器
            dKj = torch.zeros_like(Kj)
            dVj = torch.zeros_like(Vj)

            for i in range(0, N, Bq):
                Qi = Q[:, i:i+Bq, :]
                Oi = O[:, i:i+Bq, :]
                dOi = dO[:, i:i+Bq, :]

                Di = D[:, i:i+Bq, :]
                Li = L[:, i:i+Bq].unsqueeze(-1) # (B, Bq, 1)
                # 𝑺=𝑸𝑲⊤/√𝑑
                Sij = Qi @ Kj.transpose(-2, -1) * scale
                Pij = torch.exp(Sij - Li)
                # 𝒅𝑽=𝑷⊤𝒅𝑶
                dVj += Pij.transpose(-2, -1) @ dOi
                # 𝒅𝑷=𝒅𝑶𝑽⊤
                dPij = dOi @ Vj.transpose(-2, -1)

                # dSij = Pij(dPij - Di) 𝑑𝑆𝑖𝑗=𝑃𝑖𝑗(𝑑𝑃𝑖𝑗−𝐷𝑖)
                dSij = Pij * (dPij - Di) * scale
                # dQ 需要累加所有 j 块的结果，所以直接写回全局内存
                # dSij=[B, Bq, Bk] @ Kj=[B, Bk, d] = [B, Bq, d]
                dQ[:, i:i+Bq, :] += dSij @ Kj
                dKj += dSij.transpose(-2, -1) @ Qi

            dK[:, j:j+Bk, :] = dKj
            dV[:, j:j+Bk, :] = dVj

        return dQ, dK, dV, None


# @triton.jit
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
    is_causal: tl.constexpr,
):
    # 解析 Grid 并锁定身份 (取代了 PyTorch 的外层循环 i)
    # Program indices
    query_tile_index = tl.program_id(0) # Triton 中用来获取当前内核实例在并行网格（Grid）中坐标位置
    batch_index = tl.program_id(1)
    # tl.make_block_ptr 的操作，本质上就是利用 Block 的概念，将全局内存的数据切块搬运到这块独立的物理 SRAM 中，供这群特定的线程共同复用。
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

    offs_q = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    # 如果是 causal，当前 Query 块（索引 query_tile_index）最多只需要看到它自己所在的那个块为止。
    loop_end = (query_tile_index + 1) * Q_TILE_SIZE if is_causal else N_KEYS

    # 单一内层循环 (取代了 PyTorch 的内层循环 j)
    for k_idx in range(0, N_KEYS, K_TILE_SIZE):
        Kj = tl.load(K_block_ptr)
        Vj = tl.load(V_block_ptr)

        # 核心运算：计算 Sij = Q @ K^T
        # tl.trans 会在加载后转置 K，从而满足点积维度
        Sij = tl.dot(Qi, tl.trans(Kj)) * scale

        if is_causal:
            # 2. 计算当前 Key 块的全局序列列索引
            offs_k = k_idx + tl.arange(0, K_TILE_SIZE)
            causal_mask = offs_q[:, None] >= offs_k[None, :] # 只要 q 的索引大于等于 k 的索引，就是合法访问 (True)
            Sij = tl.where(causal_mask, Sij, float('-inf')) # 用 tl.where 把 False 的地方替换成 -inf

        Mij = tl.maximum(Mi, tl.max(Sij, axis=1)) # 在块内做了最大值，Triton 自动处理了线程间的通信
        Pij = tl.exp(Sij - Mij[:, None])
        
        alpha = tl.exp(Mi - Mij)

        # 缩放历史结果并累加
        Li = Li * alpha + tl.sum(Pij, axis=1) # tl.sum() 在块内做了归约，Triton 自动处理了线程间的通信
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


def flash_backward_kernel(
    L_ptr ,Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, D_vec,# 输入变量全部传齐
    dQ_ptr, dK_ptr, dV_ptr,  
    stride_qb, stride_qq, stride_qd, # stride_qd=1：对应内存维度。stride_qq = d 需要跳到下一个q。 stride_qb = N * d:跳到下一个batch
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,              # L 是 2D 张量，只有 2 个 stride
    stride_dob, stride_doq, stride_dod,
    stride_dvb,stride_dvq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr, 
    Q_TILE_SIZE: tl.constexpr, 
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    key_tile_index = tl.program_id(0)  # Triton 中用来获取当前内核实例在并行网格（Grid）中坐标位置,返回一个从 0 到 (N/BLOCK_K−1) 的整数。
    batch_index = tl.program_id(1) # 它对应 grid 定义里的第二个元素 B（Batch 维度）

    # 外层指针定死 K 和 V (当前 Grid 实例在 SRAM 中只负责这一个 K/V 块)
    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb, 
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0), # 基于base地址的偏移量，Grid中x轴的寻址。
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    Kj = tl.load(K_block_ptr)
    Vj = tl.load(V_block_ptr)
    dKj = tl.zeros([K_TILE_SIZE, D], dtype=tl.float32)
    dVj = tl.zeros([K_TILE_SIZE, D], dtype=tl.float32)

    # 预计算 Causal 掩码所需的 K 块真实行号
    offs_k = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
    offs_d = tl.arange(0, D)

    # 2. 计算因果掩码模式下，Q 遍历的起始坐标
    start_q = (key_tile_index * K_TILE_SIZE // Q_TILE_SIZE) * Q_TILE_SIZE if is_causal else 0

    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb, # 确定Q的batch
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(start_q, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(start_q, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        base=dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(start_q, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    for query_offset in range(start_q, N_QUERIES, Q_TILE_SIZE):
        Qi = tl.load(Q_block_ptr)
        Oi = tl.load(O_block_ptr)
        dOi = tl.load(dO_block_ptr)

        # 计算1D变量L和D_vec的偏置并加载
        offs_q = query_offset + tl.arange(0, Q_TILE_SIZE)
        li = tl.load(L_ptr + batch_index * stride_lb + offs_q * stride_lq)
        Di = tl.load(D_vec + batch_index * stride_dvb + offs_q * stride_dvq)
        # 重计算
        Sij = tl.dot(Qi, tl.trans(Kj)) * scale
        if is_causal:
            mask = offs_q[:, None] >= offs_k[None, :]
            Sij = tl.where(mask, Sij, float('-inf'))

        Pij = tl.exp(Sij - li[:, None])
        # 计算dV
        dVj += tl.dot(tl.trans(Pij).to(Qi.dtype), dOi) # 𝒅𝑽=𝑷⊤𝒅
        # 计算 dP 与 dS
        dPij = tl.dot(dOi, tl.trans(Vj).to(dOi.dtype)) # 𝒅𝑷=𝒅𝑶𝑽⊤
        dSij = Pij * (dPij - Di[:, None]) * scale # dSij = Pij(dPij - Di) 𝑑𝑆𝑖𝑗=𝑃𝑖𝑗(𝑑𝑃𝑖𝑗−𝐷𝑖)
        # 累加 dK (不断融合进 SRAM 中的 dK_block)
        dKj += tl.dot(tl.trans(dSij).to(Qi.dtype), Qi)
        dQi = tl.dot(dSij.to(Qi.dtype), Kj)
        # 原子加法写回 dQ (多线程争抢修改同一个 Q)，打造一个[Q_TILE_SIZE, D]
        dQ_ptrs = dQ_ptr + batch_index * stride_qb + offs_q[:, None] * stride_qq + offs_d[None, :] * stride_qd 
        tl.atomic_add(dQ_ptrs, dQi)

        Q_block_ptr = tl.advance(Q_block_ptr, (Q_TILE_SIZE, 0))
        O_block_ptr = tl.advance(O_block_ptr, (Q_TILE_SIZE, 0))
        dO_block_ptr = tl.advance(dO_block_ptr, (Q_TILE_SIZE, 0))

    dK_block_ptr = tl.make_block_ptr(
        base=dK_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
        base=dV_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    tl.store(dK_block_ptr, dKj.to(Kj.dtype))
    tl.store(dV_block_ptr, dVj.to(Vj.dtype))


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # 解析输入维度
        B, N, d = Q.shape
        
        # 在 PyTorch 端“开辟显存”来存放输出
        # Triton 自身是不能凭空创建张量的，必须由 PyTorch 准备好空盘子
        O = torch.empty_like(Q)
        L = torch.empty((B, N), device=Q.device, dtype=torch.float32)
        
        # 定义分块大小 (对应你内核里的 constexpr)
        BLOCK_Q = 16
        BLOCK_K = 16
        scale = 1.0 / math.sqrt(d)

        grid = (triton.cdiv(N, BLOCK_Q), B) # 设置线程块组成的逻辑网格,x轴:N/BLOCK_K个块。 y轴：B

        # ▲ Y轴：pid_y (Batch/Heads 独立维度，总高度 = B)
        # │─────────────────────────────────────────────────────────
        # │ 同样按 pid_x 切片，并行处理 Batch 1 ...
        # ├─────────────────────────────────────────────────────────
        # │ [0,0]      [1,0]      [2,0]      [3,0]  ...  [Nx,0]  <-- 全部隶属于 Batch 0, Head 0
        # └─────────────────────────────────────────────────────────► X轴：pid_x (序列切分维度，总长度 = N / BLOCK_Q)
        
        # 点火发射！(启动 Triton Kernel)
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
            K_TILE_SIZE=BLOCK_K,
            is_causal=is_causal,
            )

        # 6. 保存反向传播需要的变量 (Autograd 终于出场了)
        ctx.save_for_backward(L, Q, K, V, O) # PyTorch 官方接口，用来保存上下文ctx的显存里
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        L,Q,K,V,O = ctx.saved_tensors
        B, N, d = Q.shape

        BLOCK_Q = 16
        BLOCK_K = 16
        scale = 1.0 / math.sqrt(d)

        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        D_vec = (dO * O).sum(dim=-1) #(B,N)

        grid = (triton.cdiv(N, BLOCK_K), B) # 形成一个2D网格,x轴:N/BLOCK_K个块。 y轴：B

        flash_backward_kernel[grid](
            L,Q, K, V, O, dO, D_vec,# 输入变量全部传齐
            dQ, dK, dV,  # 输出空盘子传齐
            Q.stride(0), Q.stride(1), Q.stride(2), # stride_qb, stride_qq, stride_qd
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),              # L 是 2D 张量，只有 2 个 stride
            dO.stride(0), dO.stride(1), dO.stride(2),
            D_vec.stride(0), D_vec.stride(1),
            N, N,                                  # N_QUERIES, N_KEYS
            scale,
            D=d, 
            Q_TILE_SIZE=BLOCK_Q, 
            K_TILE_SIZE=BLOCK_K,
            is_causal=ctx.is_causal,
        )

        return dQ, dK, dV, None