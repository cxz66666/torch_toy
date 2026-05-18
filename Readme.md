# Torch Toy Samples

This repository is a collection of small PyTorch, Triton, CUDA, and distributed
training experiments. The code is organized by topic rather than as a reusable
package, so each sample can stay self-contained.

## Layout

- `samples/fp8/`: FP8 quantization, GEMM, batch-matmul, and Float8/FSDP samples.
- `samples/triton/`: Triton block-order, BMM, and SM occupancy experiments.
- `samples/cuda/`: CUDA C++ kernels, CUDA graph demos, TMA, memory-ordering, P2P,
  NCCL, and optimizer kernels.
- `samples/distributed/`: torchrun/NCCL/FSDP checkpoint and all-reduce samples.
- `samples/python_runtime/`: Python runtime behavior demos.
- `tools/`: Log parsing and one-off helper tools.
- `scripts/`: Small operational scripts shared by the samples.

## Useful Commands

Run a small all-reduce smoke:

```bash
torchrun --nproc_per_node=2 samples/distributed/all_reduce.py --tensor_size 1024 --iterations 2 --warmup 1
```

Run the FSDP2 FP8 all-gather benchmark with profiling:

```bash
nsys profile -o test_%p --capture-range=cudaProfilerApi --capture-range-end=stop -t mpi,cuda,nvtx,ucx \
  torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
  samples/distributed/fsdp_fp8/fsdp2_float8_allgather_benchmark.py \
  --fp8=true --fp8_all_gather=true --force_recompute=true --precompute-scale=true
```

Run NCU on a locally built CUDA sample binary:

```bash
scripts/ncu_aout.sh ncu_report%i ./a.out
```
