import torch
import torch.distributed as dist
from typing import Any, Type
from torch.optim import Optimizer
import torch.nn as nn
from cs336_basics.model import Linear, Embedding, BasicsTransformerLM

class FSDP(nn.Module):
    def __init__(self, module: torch.nn.Module, Compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.rank = dist.get_rank()
        self.compte_dtype = Compute_dtype
        self.world_size = dist.get_world_size()
        self.local_params = {}     # _init_weight_shard 分片之后得到的本地params信息
        self.bucket_size = 3 * 1024 *1024  # 每个分桶大小
        self.handles = []
        self.scatter_bucket = []   # reduce-scatter 桶
        self.reduce_bucket = []    # all-reduce 桶
        self._broadcast_parameters()
        self._create_bucket()
        self._create_forward_hook()

    def _create_forward_hook(self):
        """
        找出模型中的 Linear 和 Embedding 层,注册forward_hook
        """
        for name, layer in self.module.named_modules():
            if isinstance(layer,(Linear, Embedding)):
                self.local_params[name] = self._init_weight_shard(layer.weight, name) #进行分片
                def before_forward_hook(mod, inputs, layer_name=name):
                    self._on_before_forward(layer_name, mod)
                # 每次某个层即将执行 forward() 时，PyTorch 会先调用你注册的函数
                handle = layer.register_forward_pre_hook(before_forward_hook)  #在此layer上注册hook函数

    @torch.no_grad()
    def _broadcast_parameters(self):
        if self.world_size > 1:
            for param in self.module.parameters():
                dist.broadcast(param, src=0)

    def _create_bucket(self):
        """
        为每层的梯度分桶,每个桶里有一个layer或多个layer的参数
        """
        shard_ids = {id(layer.weight) for layer in self.module.modules() 
                    if isinstance(layer, (Linear, Embedding))}
        buckets =  {
            "scatter": self.scatter_bucket,
            "reduce": self.reduce_bucket,
        }
        current = {"scatter": [], "reduce": []}
        sizes = {"scatter": 0, "reduce": 0}

        for p in reversed(list(self.module.parameters())):
            if not p.requires_grad:
                continue
            kind = "scatter" if id(p) in shard_ids else "reduce"
            numel = p.numel()
            if kind == "scatter":
                shard_numel = (numel + self.world_size - 1) // self.world_size
                numel = shard_numel * self.world_size
            p_size = numel * 4  # 这里选择 FP32 梯度通信
            if current[kind] and (
                sizes[kind] + p_size > self.bucket_size
            ):
                buckets[kind].append(current[kind])
                current[kind] = []
                sizes[kind] = 0
            current[kind].append(p)
            sizes[kind] += p_size
        if current[kind]:
            buckets[kind].append(current[kind])

        for bucket_idx, bucket_params in enumerate(buckets["scatter"]):
            """
            为每个桶分配输入输出等信息
            """
            original_numels = [p.numel() for p in bucket_params]
            shard_numels = [(n + self.world_size - 1) // self.world_size for n in original_numels]
            output_numel = sum(shard_numels)
            device = bucket_params[0].device

            in_buffer = torch.zeros(output_numel * self.world_size,
                                    device=device,
                                    dtype=torch.float32,)
            out_buffer = torch.zeros(output_numel,               # rank用来接收的数据
                                    device=device,
                                    dtype=torch.float32,)

            buckets["scatter"][bucket_idx] = {
                                        "kind" : "scatter",
                                        "out_buffer" : out_buffer,
                                        "in_buffer" : in_buffer,
                                        "params" : bucket_params,
                                        "original_numels": original_numels,
                                        "shard_numels": shard_numels,
                                        "trigger" : False,
                                        "ready" : set()}

            bucket = buckets["scatter"][bucket_idx]
            hook = self._send_bucket(bucket)   # 钩子函数
            for param in bucket["params"]:
                param.register_post_accumulate_grad_hook(hook)

        for bucket_idx, bucket_params in enumerate(buckets["reduce"]):
            device = bucket_params[0].device
            original_numels = [p.numel() for p in bucket_params]
            out_buffer = torch.zeros(sum(original_numels),
                                        device=device,
                                        dtype=torch.float32,)
            buckets["reduce"][bucket_idx] = {
                                        "kind" : "reduce",
                                        "out_buffer" : out_buffer,
                                        "params" : bucket_params,
                                        "trigger" : False,
                                        "params" : bucket_params,
                                        "ready" : set()}
            bucket = buckets["reduce"][bucket_idx]
            hook = self._send_bucket(bucket)
            for param in bucket["params"]:
                param.register_post_accumulate_grad_hook(hook)

    def _send_bucket(self, buckets):
        """
        输入: bucket = buckets["scatter"]/buckets["reduce"],把桶里的每层参数都平分给不同的rank
        等待一个桶里的所有参数梯度就绪,然后把它们打包,发起一次异步梯度通信
        每个桶里有一个layer或多个layer的参数
        """
        def hook(param: torch.Tensor) -> None:  # backward 时调用
            buckets["ready"].add(id(param))
            if (buckets["trigger"] == False) and (len(buckets["ready"]) == len(buckets["params"])):
                buckets["trigger"] = True
                print("发送梯度桶...........")
                if buckets["kind"] == "scatter":
                    input_rows = buckets["in_buffer"].view(self.world_size,buckets["out_buffer"].numel())
                    offset = 0
                    for p, n, s in zip(buckets["params"], buckets["original_numels"], buckets["shard_numels"]):
                        grad = p.grad.detach().reshape(-1)
                        print("grad.numel = ", grad.numel())  
                        padded_grad = buckets["in_buffer"].new_zeros(
                            self.world_size * s
                        )
                        print("original_numels = ", n)
                        padded_grad[:n].copy_(grad)  #复制原始梯度
                        input_rows[:, offset:offset + s].copy_(
                            padded_grad.view(self.world_size, s)
                        )
                        offset += s
                    handle = dist.reduce_scatter_tensor(buckets["out_buffer"],   # 本rank用来接收结果的Tensor
                                                        buckets["in_buffer"],    # 本rank提供的Tensor列表，本rank提供的各份梯度
                                                        op=dist.ReduceOp.SUM,
                                                        async_op=True)

                elif buckets["kind"] == "reduce":
                    offset = 0
                    for p in buckets["params"]:
                        grad = p.grad.detach().reshape(-1)
                        n = grad.numel()
                        buckets["out_buffer"][offset:offset + n].copy_(grad)
                        offset += n
                    handle = dist.all_reduce(buckets["out_buffer"],
                                            op=dist.ReduceOp.SUM,
                                            async_op=True)
                else:
                    raise ValueError(f"Unknown bucket kind: {buckets['kind']}")

                self.handles.append((handle, buckets["out_buffer"]))
                print("发送梯度桶完成...........")
        return hook

    @torch.no_grad()
    def _init_weight_shard(self, layer_param, layer_name): 
        """
        切分每层layer的参数到各自的rank上,并保留原始精度。
        layer_param: 每一层的参数
        layer_name: 每一层的名字
        """
        layer_param_shape = tuple(layer_param.shape)
        layer_param_numel = layer_param.numel()
        # 向上取整获取每个rank的参数量，多余补0
        shared_param_numel = (layer_param_numel + self.world_size - 1) // self.world_size
        rank_start = self.rank * shared_param_numel
        # 分配本地的FP32的值的buffer，用于优化器更新
        shared_master_shard = torch.zeros(shared_param_numel, 
                                    dtype=torch.float32, 
                                    device=layer_param.device)
        # 获取当前rank上的有效参数量
        shared_valid_numel = max(0, min(shared_param_numel, layer_param_numel - rank_start))
        flat_layer_param = layer_param.detach().reshape(-1)
        # 把当前rank上对应的参数复制到当前rank的主权重参数上，多余的用0补充
        if shared_valid_numel > 0:
            shared_master_shard[:shared_valid_numel].copy_(
                flat_layer_param[rank_start : rank_start + shared_valid_numel]
            )
        layer_param.data = shared_master_shard  
        del flat_layer_param
        # 返回每个rank上分片状态
        return {
            "param_name" :          (f"{layer_name}.weight" if layer_name else "weight"),
            "param_shape" :         layer_param_shape,   #原始层的形状
            "param_numel" :         layer_param_numel,   #原始层参数总和
            "rank_start" :          rank_start,          #对应rank分片的起始位置
            "rank_numel" :          shared_param_numel,  #对应rank的分片的数据量
            "shared_valid_numel":   shared_valid_numel,  #对应rank上的实际数据量（不包含0）
            "shared_master_shard":  shared_master_shard, #对应rank上实际主权重数据（包含0）
        }

    def _on_before_forward(self, name, layer):
        """
        forward之前聚合所有rank上对应层的参数,使得每个rank有完整的层。
        """
        print(f"即将执行 forward:{name}，类型:{type(layer).__name__}")

        local_params_info = self.local_params[name]
        shared_master = local_params_info["shared_master_shard"]
        # 发起当前层的参数聚合
        if shared_master.dtype != torch.float32:
            raise ValueError("master_shard 必须是 FP32")
        # 准备当前rank上发送的数据
        if self.compte_dtype is None:
            send = shared_master
        else:
            send = shared_master.to(dtype=self.compte_dtype)
        # 创建all-gather接收的buffer
        buffer = [torch.empty_like(send)
                    for _ in range(self.world_size)] 
        print(f"本地rank{self.rank}, {name}  执行 all-gather")
        work = dist.all_gather(buffer, send, async_op=True)
        task = {
            "work": work,
            "buffer": buffer,
            "send": send,
        }
        # 等待通信完成，去掉 padding，恢复原始形状
        task["work"].wait()
        fully_layer_params = torch.cat(task["buffer"], dim=0)
        layer.weight.data = fully_layer_params[:local_params_info["param_numel"]] \
                                                .view(local_params_info["param_shape"])

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle, input_buffer in self.handles:
            handle.wait()
            print("梯度桶收到...........")
        self.handles.clear()

        
