# Agent Notes

这个仓库是个人 sample code 集合，不是可安装的 Python/CUDA 包。维护时优先让每个样例自洽、清楚、可独立运行；不要为了“复用”强行抽出公共框架。

## 仓库地图

- `samples/fp8/`: FP8 量化、`torch._scaled_mm`、FP8 BMM、Triton TMA、torchao/FSDP Float8 训练样例。
- `samples/triton/`: Triton load/store block order、LRM BMM forward/backward、SM occupier 实验。
- `samples/cuda/`: CUDA C++ 样例，包括 graph capture、pooling kernel 优化迭代、memory consistency、IPC/P2P、NCCL-like copy-reduce、optimizer kernel、TMA skeleton。
- `samples/distributed/`: `torchrun`、NCCL all-reduce、async checkpoint、FSDP1/FSDP2 FP8 benchmark。
- `samples/python_runtime/`: Python runtime/GIL 行为样例。
- `tools/`: 仓库级辅助工具。最重要的是 `tools/run_sample_smokes.py`。
- `scripts/`: 轻量操作脚本，例如 NCU 包装和清理 GPU Python 进程。

## 探索方式

- 用 `rg --files` 和 `rg` 先定位样例，不要按旧根目录文件名假设位置。
- 文件名通常表达样例意图：`*_benchmark.py` 偏性能观察，`*_demo.py` 偏概念演示，`*_ncu.py` 偏 profiling 入口。
- CUDA pooling 目录保留了多版优化迭代，文件之间不一定共享最佳实现；改动时要把它们当作不同实验版本看待。
- 分布式和 FP8 文件对 shape、batch、alignment 很敏感，尤其 FP8 matmul 常要求最后一维或 batch 维满足 16 对齐。

## 修改原则

- 保持每个样例可直接运行，优先添加 argparse 参数和小规模 smoke 输入。
- 不要把所有样例改成统一库式抽象；这个仓库的价值是每个文件能独立表达一个实验。
- 删除重复样例时，要保留覆盖面更完整、参数更清楚、验证路径更稳的版本。
- 对实验性能力做显式 capability check。比如当前 Triton 3.3.1 没有 `tl.make_tensor_descriptor` 时，TMA Python 样例应清楚 skip，而不是抛 AttributeError。
- 中文注释使用简体中文。

## 本地验证

语法检查：

```bash
PYTHONPYCACHEPREFIX=/tmp/torch_toy_pycache python -m compileall -q samples tools
```

Python/GPU smoke：

```bash
python tools/run_sample_smokes.py --suite python --keep-going
```

CUDA 编译 smoke：

```bash
python tools/run_sample_smokes.py --suite cuda-compile --cuda-arch sm_90 --keep-going
```

CUDA 运行 smoke：

```bash
python tools/run_sample_smokes.py --suite cuda-run --cuda-arch sm_90 --keep-going
```

`cuda-run` 会自动生成 pooling 样例所需的 `binary_data.bin` 小输入。`samples/cuda/graphs/test_debug.cu` 是故意触发 illegal memory access 的调试样例，harness 会把它作为预期失败处理；`sync_error.cu` 成功打印 stale L1 现象时算复现成功。

## 远端验证

推荐远端：

- Host: `h20-gl-1`
- Conda: `torch2.7`
- 参考环境：PyTorch `2.7.1+cu128`、Triton `3.3.1`、8 张 NVIDIA H20。
- torchao 与 PyTorch 2.7 兼容版本使用 `torchao==0.10.0`。较新的 torchao 版本可能要求更高 PyTorch 版本。
- PyTorch CUDA extension 样例需要 `ninja`。

常用流程是在远端临时目录验证，不要直接污染远端 clean checkout：

```bash
tar --exclude=.git --exclude=.vscode -czf /tmp/torch_toy_validation.tar.gz .
scp /tmp/torch_toy_validation.tar.gz h20-gl-1:/tmp/torch_toy_validation.tar.gz
ssh h20-gl-1 'bash -lc "
  rm -rf ~/torch_toy_validation &&
  mkdir -p ~/torch_toy_validation &&
  tar -xzf /tmp/torch_toy_validation.tar.gz -C ~/torch_toy_validation &&
  source ~/miniconda3/etc/profile.d/conda.sh &&
  conda activate torch2.7 &&
  cd ~/torch_toy_validation &&
  python tools/run_sample_smokes.py --suite python --keep-going
"'
```

需要 CUDA C++ 全量验证时，在同一远端目录继续跑：

```bash
python tools/run_sample_smokes.py --suite cuda-compile --cuda-arch sm_90 --keep-going
python tools/run_sample_smokes.py --suite cuda-run --cuda-arch sm_90 --keep-going
```

## 已知兼容性点

- `tma_persistent_bmm.py` 的 Python Triton TMA 路径依赖 `tl.make_tensor_descriptor`。当前远端 Triton 3.3.1 不提供该 API，因此 smoke 中会 skip tutorial TMA matmul。
- `samples/cuda/tma/2stage_producer_consumer.cu` 是可编译运行的 TMA/async bulk skeleton，但 producer 侧使用普通 shared-memory fill 来兼容当前 CUDA headers 暴露的 overload。
- `samples/fp8/batch_matmul/lagrange_batch_dense.py` 使用 `torch._scaled_mm` 表达 batch-specific FP8 dense 的核心意图，避免依赖不稳定的内部 Lagrange FP8 API 名称。
- `tools/run_sample_smokes.py` 是修改样例后的第一验证入口；新增样例时优先给它补一个小规模 smoke 命令。
