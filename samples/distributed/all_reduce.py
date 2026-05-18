import torch
import torch.distributed as dist
import os
import time
import argparse

def main():
    """
    使用 torchrun 启动的 AllReduce 性能测试脚本。
    """
    parser = argparse.ArgumentParser(description="PyTorch All-Reduce NCCL Performance Test")
    parser.add_argument('--tensor_size', type=int, default=256*1024*1024,
                        help='张量中的 float32 元素数量 (默认: 256M, 即 1GB)')
    parser.add_argument('--iterations', type=int, default=20,
                        help='用于性能测量的迭代次数')
    parser.add_argument('--warmup', type=int, default=5,
                        help='预热迭代的次数')
    args = parser.parse_args()

    # 1. 初始化分布式环境
    # torchrun 会自动设置 'RANK', 'WORLD_SIZE', 'LOCAL_RANK' 环境变量
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="nccl")

    # 将当前进程绑定到正确的GPU设备
    torch.cuda.set_device(local_rank)
    
    if global_rank == 0:
        print("="*40)
        print(f"PyTorch NCCL All-Reduce Test on {world_size} GPUs")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Version: {torch.version.cuda}")
        try:
            print(f"NCCL Version: {torch.cuda.nccl.version()}")
        except AttributeError:
            print("NCCL Version: Not Available")
        print("="*40)
        tensor_size_bytes = args.tensor_size * 4
        tensor_size_gb = tensor_size_bytes / (1024**3)
        print(f"Tensor Size (per GPU): {tensor_size_gb:.3f} GB")
        print(f"Warmup Iterations: {args.warmup}")
        print(f"Test Iterations: {args.iterations}")
        print("-" * 40)


    # 2. 准备测试数据
    tensor = torch.ones(args.tensor_size, device=f'cuda:{local_rank}', dtype=torch.float32)

    # 3. 验证正确性
    verify_tensor = tensor.clone()
    dist.all_reduce(verify_tensor, op=dist.ReduceOp.SUM)
    expected_value = float(world_size)
    
    # 仅在 rank 0 上打印验证结果
    if global_rank == 0:
        if torch.allclose(verify_tensor, torch.full_like(verify_tensor, expected_value)):
            print("Verification PASSED. All-Reduce results are correct.")
        else:
            print(f"Verification FAILED! Expected {expected_value}, got {verify_tensor[0]}. Aborting.")
            dist.destroy_process_group()
            return
    
    # 等待所有进程完成验证
    dist.barrier()

    # 4. 预热
    if global_rank == 0:
        print("Starting warmup...")
    for _ in range(args.warmup):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize() # 等待所有GPU操作完成

    # 5. 正式测试
    if global_rank == 0:
        print("Warmup finished. Starting performance measurement...")
    
    # 在开始计时前进行同步
    dist.barrier()
    start_time = time.time()

    for _ in range(args.iterations):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    # 在结束计时前，确保所有GPU上的AllReduce操作都已完成
    torch.cuda.synchronize()
    end_time = time.time()
    
    # 6. 计算并打印结果 (仅在 rank 0 上)
    if global_rank == 0:
        total_time = end_time - start_time
        avg_time_per_iter = total_time / args.iterations
        tensor_size_bytes = args.tensor_size * 4
        
        # 理论总线带宽公式 (Ring All-reduce)
        # 数据量在一次all-reduce中几乎是 2 * (N-1)/N * tensor_size
        bus_bw_gbps = (2 * (world_size - 1) / world_size * tensor_size_bytes) / avg_time_per_iter / 1e9
        
        print("-" * 40)
        print("Performance Results (from Rank 0):")
        print(f"Average time per All-Reduce: {avg_time_per_iter * 1000:.4f} ms")
        print(f"Estimated Bus Bandwidth: {bus_bw_gbps:.4f} GB/s")
        print("="*40)

    # 7. 清理
    dist.destroy_process_group()

if __name__ == "__main__":
    main()