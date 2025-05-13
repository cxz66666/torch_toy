import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import Dataset, DataLoader
import os
import time
import functools

class LinearBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4096, 14436).to(torch.bfloat16)
        self.linear2 = nn.Linear(14436, 4096).to(torch.bfloat16)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

class MyModel(nn.Module):
    def __init__(self, num_blocks=32):
        super().__init__()
        layers = [LinearBlock() for _ in range(num_blocks)]
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc_layers(x)

# 3. 构造模拟训练数据
class DummyDataset(Dataset):
    def __init__(self, num_samples, input_features, output_features):
        self.num_samples = num_samples
        self.input_features = input_features
        self.output_features = output_features

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        data = torch.randn(self.input_features)
        target = torch.randn(self.output_features) # 假设目标与模型输出维度一致
        return data, target

def get_model_parameter_count(model: nn.Module, count_trainable_only: bool = False) -> int:
    total_params = 0
    for param in model.parameters():
        if count_trainable_only and not param.requires_grad:
            continue
        total_params += param.numel()
    return total_params


# 4. FSDP 设置和训练函数
def setup(rank, world_size):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355' # 选择一个未被占用的端口
    print(f"Setting up process group for rank {rank} out of {world_size}")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print("Process group initialized.")

def cleanup():
    dist.destroy_process_group()

def fsdp_main(rank, world_size, args):
    setup(rank, world_size)

    model = MyModel(num_blocks=args.num_blocks)
    # model = model.to(rank) # 将模型移动到当前 rank 的 GPU

    if rank == 0:
        total_params_base = get_model_parameter_count(model)
        print(f"Total parameters (base model): {total_params_base / 1e6:.2f}M")
        total_params_base_trainable = get_model_parameter_count(model,count_trainable_only=True)
        print(f"Total trainable parameters (base model): {total_params_base_trainable / 1e6:.2f}M")
    # FSDP 自动包装策略
    # 我们希望 FSDP 包装我们的 LinearBlock 实例
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LinearBlock}
    )
    
    # 使用 FSDP 包装模型
    # 对于纯粹的性能测试，可以关闭 CPU offload
    # 如果模型过大导致显存不足，可以开启 cpu_offload=CPUOffload(offload_params=True)
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        device_id=torch.cuda.current_device(), # 确保 FSDP 知道当前设备
        mixed_precision=torch.distributed.fsdp.MixedPrecision( # 可选：混合精度
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        ),
    )
    if rank == 0:
        total_params_fsdp = get_model_parameter_count(fsdp_model)
        print(f"Total parameters (FSDP model): {total_params_fsdp / 1e6:.2f}M")

        total_params_fsdp_trainable = get_model_parameter_count(fsdp_model, count_trainable_only=True)
        print(f"Total trainable parameters (FSDP model): {total_params_fsdp_trainable / 1e6:.2f}M")

        torch.cuda.synchronize(torch.cuda.current_device()) # 确保所有操作完成
        allocated_memory_fsdp_rank = torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024**2)
        reserved_memory_fsdp_rank = torch.cuda.memory_reserved(torch.cuda.current_device()) / (1024**2)

        print(f"\n--- GPU Memory Usage on Rank {rank} (FSDP Model) ---")
        print(f"Memory Allocated: {allocated_memory_fsdp_rank:.2f} MB")
        print(f"Memory Reserved: {reserved_memory_fsdp_rank:.2f} MB")


    # 优化器 (必须在模型 FSDP 包装之后创建)
    optimizer = optim.Adam(fsdp_model.parameters(), lr=args.lr)

    # 损失函数
    criterion = nn.MSELoss()

    # 模拟数据加载器
    # 注意：对于大型数据集，应该使用 DistributedSampler
    # 这里为了简单，每个 rank 处理自己的数据子集（通过 num_samples_per_rank）
    # 或者每个 rank 都加载完整数据，但在训练时只处理一部分
    dataset = DummyDataset(args.num_samples_per_epoch, 4096, 4096)
    # 使用 DistributedSampler 来确保数据在不同 GPU 之间正确划分
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True # 通常训练时需要打乱
    )
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers, # 根据系统调整
        pin_memory=True
    )

    # 训练循环
    fsdp_model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_start_time = time.time()
        num_batches = 0
        if train_sampler: # 如果使用了sampler，需要在每个epoch开始时设置
            train_sampler.set_epoch(epoch)

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(torch.bfloat16).to(rank), target.to(torch.bfloat16).to(rank) # 将数据移动到当前 rank 的 GPU

            optimizer.zero_grad()
            output = fsdp_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if rank == 0 and (batch_idx % args.log_interval == 0):
                print(f"Epoch: {epoch+1}/{args.epochs} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        epoch_duration = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        samples_per_second = (args.num_samples_per_epoch / world_size * world_size) / epoch_duration if epoch_duration > 0 else 0


        avg_loss_tensor = torch.tensor([avg_epoch_loss], device=rank)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG) # 计算所有 rank 的平均 loss

        if rank == 0:
            print(f"-" * 50)
            print(f"Epoch {epoch+1} Summary:")
            print(f"  Average Loss: {avg_loss_tensor.item():.4f}")
            print(f"  Epoch Duration: {epoch_duration:.2f} seconds")
            print(f"  Samples/sec: {samples_per_second:.2f}")
            print(f"-" * 50)

    cleanup()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch FSDP Linear Model Benchmark')
    parser.add_argument('--batch_size', type=int, default=16, # 根据显存调整
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=2, # 简单测试，少量 epoch
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--num_blocks', type=int, default=32,
                        help='number of LinearBlock layers in the model (default: 32)')
    parser.add_argument('--num_samples_per_epoch', type=int, default=8000, # 总样本数
                        help='number of samples per epoch (default: 8000)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--num_workers', type=int, default=2, # DataLoader workers
                        help='number of dataloader workers (default: 2)')
    args = parser.parse_args()

    # torchrun 会自动设置这些环境变量
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0)) # 全局 rank
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) # 节点内 rank

    # FSDP通常使用全局 rank
    # 如果 world_size > 1 (即 torchrun 启动了多个进程)
    if world_size > 1:
        fsdp_main(rank=rank, world_size=world_size, args=args)
    else: # 单进程运行 (例如调试，不使用 FSDP)
        print("Running in single process mode (no FSDP).")
        if torch.cuda.is_available():
            fsdp_main(rank=0, world_size=1, args=args)
        else:
            print("CUDA not available. FSDP example requires CUDA.")
