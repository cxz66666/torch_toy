#!/usr/bin/env python3
"""Run small smoke tests for the sample-code repository.

The goal is not to replace benchmarks. Each command is intentionally small so a
remote GPU machine can quickly check that a sample still expresses its intended
idea after directory or API cleanup.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = Path(os.environ.get("TORCH_TOY_SMOKE_BUILD", "/tmp/torch_toy_smokes"))


PYTHON_SMOKES: list[tuple[str, list[str], int]] = [
    (
        "python_runtime/gil_thread_demo",
        ["python", "samples/python_runtime/gil_thread_demo.py", "--seconds", "0.2", "--work-items", "1000"],
        20,
    ),
    (
        "triton/load_store_block_order",
        ["python", "samples/triton/load_store_block_order.py"],
        60,
    ),
    (
        "triton/bmm_forward",
        ["python", "samples/triton/bmm/lrm_bmm_forward.py", "--smoke", "--batch", "1", "--m", "64", "--n", "64", "--k", "64"],
        120,
    ),
    (
        "triton/bmm_grad_activation",
        ["python", "samples/triton/bmm/lrm_bmm_grad_activation.py", "--smoke", "--batch", "1", "--m", "64", "--n", "64", "--k", "64"],
        120,
    ),
    (
        "triton/bmm_grad_weight",
        ["python", "samples/triton/bmm/lrm_bmm_grad_weight.py", "--smoke", "--batch", "1", "--m", "64", "--n", "64", "--k", "64"],
        120,
    ),
    (
        "triton/bmm_with_sm_occupier",
        [
            "python",
            "samples/triton/bmm/bmm_with_sm_occupier.py",
            "--batch",
            "1",
            "--m",
            "256",
            "--n",
            "256",
            "--k",
            "256",
            "--repeats",
            "1",
            "--warmup",
            "1",
            "--occupy-blocks",
            "1",
            "--occupy-threads",
            "64",
            "--occupy-shmem-kb",
            "0",
            "--occupy-startup-sleep",
            "0.01",
        ],
        180,
    ),
    (
        "fp8/scaled_mm_demo",
        ["python", "samples/fp8/quantization/scaled_mm_demo.py", "--m", "512", "--n", "512", "--k", "512"],
        120,
    ),
    (
        "fp8/per_tensor_cast",
        [
            "python",
            "samples/fp8/quantization/per_tensor_cast_benchmark.py",
            "--batch",
            "1",
            "--m",
            "64",
            "--n",
            "64",
            "--warmup",
            "1",
            "--reps",
            "1",
        ],
        180,
    ),
    (
        "fp8/per_channel_quant",
        ["python", "samples/fp8/quantization/per_channel_quant_benchmark.py", "--smoke", "--batch", "1", "--m", "64", "--n", "64"],
        180,
    ),
    (
        "fp8/per_tensor_quant_reduce",
        ["python", "samples/fp8/quantization/per_tensor_quant_ncu.py", "--batch", "1", "--m", "64", "--n", "64"],
        120,
    ),
    (
        "fp8/bmm_column_major",
        ["python", "samples/fp8/gemm/bmm_column_major_benchmark.py", "--smoke", "--batch", "1", "--m", "64", "--n", "64", "--k", "64"],
        120,
    ),
    (
        "fp8/per_channel_bmm",
        ["python", "samples/fp8/gemm/per_channel_bmm_ncu.py", "--batch", "1", "--m", "64", "--n", "64", "--k", "64", "--accumulate-every", "4"],
        120,
    ),
    (
        "fp8/tma_persistent_bmm",
        [
            "python",
            "samples/fp8/gemm/tma_persistent_bmm.py",
            "--mode",
            "tutorial",
            "--prec",
            "fp16",
            "-K",
            "64",
            "--warmup-reps",
            "1",
            "--reps",
            "1",
        ],
        180,
    ),
    (
        "fp8/lagrange_batch_dense",
        [
            "python",
            "samples/fp8/batch_matmul/lagrange_batch_dense.py",
            "--batch",
            "1",
            "--seq-len",
            "16",
            "--in-features",
            "64",
            "--out-features",
            "64",
            "--no-fp8-all-gather",
        ],
        120,
    ),
    (
        "cuda_graph/capture_torch",
        ["python", "samples/cuda/graphs/capture_torch.py"],
        120,
    ),
    (
        "cuda_graph/capture_multibs_torch",
        ["python", "samples/cuda/graphs/capture_multibs_torch.py", "--max-batch-size", "4", "--feature-dim", "64"],
        120,
    ),
    (
        "cuda_graph/capture_overlap_torch",
        ["python", "samples/cuda/graphs/capture_overlap_torch.py", "--nanosec", "1000000"],
        240,
    ),
    (
        "distributed/all_reduce",
        [
            "torchrun",
            "--nproc_per_node=2",
            "--rdzv_backend",
            "c10d",
            "--rdzv_endpoint",
            "localhost:0",
            "samples/distributed/all_reduce.py",
            "--tensor_size",
            "1024",
            "--iterations",
            "2",
            "--warmup",
            "1",
        ],
        120,
    ),
    (
        "distributed/async_checkpoint_fsdp",
        ["python", "samples/distributed/async_checkpoint_fsdp.py", "--world-size", "1", "--steps", "1"],
        180,
    ),
    (
        "distributed/fsdp1",
        [
            "torchrun",
            "--nproc_per_node=2",
            "--rdzv_backend",
            "c10d",
            "--rdzv_endpoint",
            "localhost:0",
            "samples/distributed/fsdp_fp8/fsdp1_allgather_benchmark.py",
            "--batch_size",
            "1",
            "--epochs",
            "1",
            "--num_blocks",
            "1",
            "--hidden_size",
            "64",
            "--intermediate_size",
            "128",
            "--num_samples_per_epoch",
            "2",
            "--num_workers",
            "0",
            "--log_interval",
            "1",
        ],
        180,
    ),
    (
        "distributed/fsdp2_float8",
        [
            "torchrun",
            "--nproc_per_node=2",
            "--rdzv_backend",
            "c10d",
            "--rdzv_endpoint",
            "localhost:0",
            "samples/distributed/fsdp_fp8/fsdp2_float8_allgather_benchmark.py",
            "--batch_size",
            "16",
            "--epochs",
            "1",
            "--num_blocks",
            "1",
            "--hidden_size",
            "64",
            "--intermediate_size",
            "128",
            "--num_samples_per_epoch",
            "32",
            "--num_workers",
            "0",
            "--log_interval",
            "1",
            "--fp8",
            "true",
            "--fp8_all_gather",
            "false",
            "--force_recompute",
            "false",
        ],
        240,
    ),
]


CUDA_SOURCES = [
    "samples/cuda/comm/nccl/copy_reduce.cu",
    "samples/cuda/graphs/capture.cu",
    "samples/cuda/graphs/malloc_async_capture.cu",
    "samples/cuda/graphs/malloc_pool_capture.cu",
    "samples/cuda/graphs/test_debug.cu",
    "samples/cuda/kernels/pooling/baseline.cu",
    "samples/cuda/kernels/pooling/opt1.cu",
    "samples/cuda/kernels/pooling/opt2_needalign.cu",
    "samples/cuda/kernels/pooling/opt3_tryreducereg.cu",
    "samples/cuda/kernels/pooling/opt4_100occup_reduce_reg.cu",
    "samples/cuda/kernels/pooling/opt5_100occup_32_reg.cu",
    "samples/cuda/kernels/pooling/opt6_persistent_kerenl_32_reg.cu",
    "samples/cuda/kernels/pooling/opt7.5_ldgst_with_atomic.cu",
    "samples/cuda/kernels/pooling/opt7_emb_to_smem_atomic_add.cu",
    "samples/cuda/kernels/pooling/opt8_ldgst_with_assign.cu",
    "samples/cuda/kernels/pooling/opt9_ldgst_store_async_with_assign.cu",
    "samples/cuda/kernels/pooling/opt10_store_with_tma_only_64_dim.cu",
    "samples/cuda/kernels/transpose/baseline.cu",
    "samples/cuda/memory_consistency/sync_correct.cu",
    "samples/cuda/memory_consistency/sync_error.cu",
    "samples/cuda/optimizers/adamom/baseline.cu",
    "samples/cuda/optimizers/rmspropv2/baseline.cu",
    "samples/cuda/tma/2stage_producer_consumer.cu",
]

EXPECTED_FAILURE_RUNS = {
    "samples/cuda/graphs/test_debug.cu": "intentional illegal memory access sample",
}


def find_nvcc() -> str | None:
    candidates = [shutil.which("nvcc")]
    for env_name in ("CUDA_HOME", "CUDA_PATH"):
        cuda_root = os.environ.get(env_name)
        if cuda_root:
            candidates.append(str(Path(cuda_root) / "bin" / "nvcc"))
    candidates.extend(["/usr/local/cuda/bin/nvcc", "/usr/local/cuda-12/bin/nvcc"])

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def run_command(name: str, cmd: list[str], timeout: int, expect_failure: bool = False) -> bool:
    print(f"\n=== {name} ===", flush=True)
    print(" ".join(cmd), flush=True)
    try:
        completed = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        output = completed.stdout.strip()
        if output:
            print(output[-6000:], flush=True)
        ok = completed.returncode == 0
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "").strip() if isinstance(exc.stdout, str) else ""
        if output:
            print(output[-6000:], flush=True)
        print(f"TIMEOUT after {timeout}s", flush=True)
        ok = False

    if expect_failure:
        print("EXPECTED FAILURE" if not ok else "UNEXPECTED PASS", flush=True)
        return not ok
    print("PASS" if ok else "FAIL", flush=True)
    return ok


def compile_cuda(source: str, cuda_arch: str, nvcc: str) -> tuple[str, bool]:
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    binary = BUILD_ROOT / source.replace("/", "_").replace(".", "_")
    cmd = [
        nvcc,
        "-std=c++20",
        "-O2",
        "-lineinfo",
        f"-arch={cuda_arch}",
        "--extended-lambda",
        source,
        "-o",
        str(binary),
    ]
    ok = run_command(f"compile {source}", cmd, timeout=180)
    return str(binary), ok


def ensure_pooling_binary_data() -> None:
    edge_length = 4096
    emb_dim = 64
    emb_rows = 128
    pooling_rows = 64
    emb_table_length = emb_rows * emb_dim
    pooling_table_length = pooling_rows * emb_dim

    edge_in = [i % emb_rows for i in range(edge_length)]
    edge_out = [(i * 7) % pooling_rows for i in range(edge_length)]
    emb_table = [((i % 97) - 48) / 97.0 for i in range(emb_table_length)]

    path = REPO_ROOT / "binary_data.bin"
    with path.open("wb") as f:
        f.write(struct.pack("<iii", edge_length, emb_table_length, pooling_table_length))
        f.write(struct.pack("<q", emb_dim))
        f.write(struct.pack(f"<{edge_length}i", *edge_in))
        f.write(struct.pack(f"<{edge_length}i", *edge_out))
        f.write(struct.pack(f"<{emb_table_length}f", *emb_table))
    print(
        "Generated pooling input: "
        f"edges={edge_length}, emb_rows={emb_rows}, pooling_rows={pooling_rows}, emb_dim={emb_dim}",
        flush=True,
    )


def run_p2p(cuda_arch: str) -> bool:
    name = "cuda/comm/p2p"
    print(f"\n=== {name} ===", flush=True)
    cmd = ["make", "-C", "samples/cuda/comm/p2p", f"NVCCFLAGS=-std=c++20 -O2 -lineinfo -arch={cuda_arch}"]
    if not run_command("compile cuda/comm/p2p", cmd, timeout=180):
        return False

    server = subprocess.Popen(
        ["samples/cuda/comm/p2p/server"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    time.sleep(1)
    try:
        client = subprocess.run(
            ["samples/cuda/comm/p2p/client"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=60,
        )
        try:
            server_out, _ = server.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()
            server_out, _ = server.communicate(timeout=5)
        print((server_out or "")[-3000:], flush=True)
        print((client.stdout or "")[-3000:], flush=True)
        ok = server.returncode == 0 and client.returncode == 0
    finally:
        if server.poll() is None:
            server.kill()
    print("PASS" if ok else "FAIL", flush=True)
    return ok


def run_python_smokes(keep_going: bool) -> bool:
    all_ok = True
    for name, cmd, timeout in PYTHON_SMOKES:
        ok = run_command(name, cmd, timeout)
        all_ok = all_ok and ok
        if not ok and not keep_going:
            return False
    return all_ok


def run_cuda_smokes(cuda_arch: str, mode: str, keep_going: bool) -> bool:
    nvcc = find_nvcc()
    if nvcc is None:
        print("nvcc is not available in PATH, CUDA_HOME, CUDA_PATH, or /usr/local/cuda/bin.", flush=True)
        return False
    print(f"Using nvcc: {nvcc}", flush=True)

    all_ok = True
    if mode != "compile":
        ensure_pooling_binary_data()
    for source in CUDA_SOURCES:
        binary, ok = compile_cuda(source, cuda_arch, nvcc)
        all_ok = all_ok and ok
        if not ok:
            if not keep_going:
                return False
            continue

        if mode == "compile":
            continue

        expect_failure = source in EXPECTED_FAILURE_RUNS
        ok = run_command(
            f"run {source}",
            [binary],
            timeout=120,
            expect_failure=expect_failure,
        )
        all_ok = all_ok and ok
        if not ok and not keep_going:
            return False

    if mode != "compile":
        ok = run_p2p(cuda_arch)
        all_ok = all_ok and ok
        if not ok and not keep_going:
            return False
    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Run torch_toy sample smoke tests.")
    parser.add_argument("--suite", choices=["python", "cuda-compile", "cuda-run", "all"], default="python")
    parser.add_argument("--cuda-arch", default="sm_90")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    ok = True
    if args.suite in {"python", "all"}:
        ok = run_python_smokes(args.keep_going) and ok
    if args.suite in {"cuda-compile", "all"}:
        ok = run_cuda_smokes(args.cuda_arch, "compile", args.keep_going) and ok
    if args.suite == "cuda-run":
        ok = run_cuda_smokes(args.cuda_arch, "run", args.keep_going) and ok

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
