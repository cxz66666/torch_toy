#!/usr/bin/env python3
# mypy: allow-untyped-defs
"""
Profile PyTorch Distributed Checkpoint save paths on one 8-GPU node.

Example:
  torchrun --standalone --nproc_per_node=8 samples/distributed/dcp_hsdp_tp_profile.py \
      --out-dir /tmp/dcp_hsdp_tp_profile \
      --profile --profile-variants sync,async_thread_plan_meta_pinned

The default mesh is 2 x 2 x 2:
  dp_replicate=2, dp_shard=2, tp=2
so FSDP2 uses HSDP over (dp_replicate, dp_shard), while tensor parallel
uses the tp dimension.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pickle
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from torch.distributed.checkpoint.planner import SavePlanner
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict
from torch.distributed.checkpoint.state_dict_saver import AsyncCheckpointerType
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp._fully_shard import MixedPrecisionPolicy
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from torch.profiler import ProfilerActivity, profile, record_function


class SwiGLUBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False, device=device, dtype=dtype)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False, device=device, dtype=dtype)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


class BigMLP(nn.Module):
    def __init__(
        self,
        layers: int,
        dim: int,
        hidden_dim: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                SwiGLUBlock(dim, hidden_dim, device=device, dtype=dtype)
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class CollectiveLogger:
    def __init__(self, path: Path, rank: int, enabled: bool) -> None:
        self.path = path
        self.rank = rank
        self.enabled = enabled
        self.variant = ""
        self.step = -1
        self._file = None
        self._originals: dict[str, Callable[..., Any]] = {}

    def __enter__(self) -> "CollectiveLogger":
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("a", buffering=1)
        for name in (
            "gather_object",
            "scatter_object_list",
            "broadcast_object_list",
            "all_gather_object",
        ):
            self._patch(name)
        return self

    def __exit__(self, *exc_info) -> None:
        for name, fn in self._originals.items():
            setattr(dist, name, fn)
        if self._file is not None:
            self._file.close()

    @contextlib.contextmanager
    def annotate(self, variant: str, step: int):
        old_variant, old_step = self.variant, self.step
        self.variant, self.step = variant, step
        try:
            yield
        finally:
            self.variant, self.step = old_variant, old_step

    def _patch(self, name: str) -> None:
        original = getattr(dist, name)
        self._originals[name] = original

        def wrapped(*args, **kwargs):
            obj_nbytes = self._estimate_object_bytes(name, args, kwargs)
            start = time.perf_counter()
            ok = False
            try:
                result = original(*args, **kwargs)
                ok = True
                return result
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                self._write_event(name, elapsed_ms, obj_nbytes, ok)

        setattr(dist, name, wrapped)

    def _estimate_object_bytes(self, name: str, args, kwargs) -> int | None:
        if not self.enabled:
            return None
        try:
            if name in ("gather_object", "all_gather_object"):
                obj = kwargs.get("obj", args[0] if args else None)
            elif name == "broadcast_object_list":
                object_list = kwargs.get("object_list", args[0] if args else None)
                obj = object_list[0] if object_list else None
            elif name == "scatter_object_list":
                input_list = kwargs.get(
                    "scatter_object_input_list", args[1] if len(args) > 1 else None
                )
                obj = input_list if input_list is not None else None
            else:
                obj = None
            return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            return None

    def _write_event(
        self, name: str, elapsed_ms: float, obj_nbytes: int | None, ok: bool
    ) -> None:
        if not self.enabled or self._file is None:
            return
        event = {
            "rank": self.rank,
            "variant": self.variant,
            "step": self.step,
            "collective": name,
            "elapsed_ms": round(elapsed_ms, 3),
            "object_nbytes": obj_nbytes,
            "ok": ok,
            "time": time.time(),
        }
        self._file.write(json.dumps(event) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/tmp/dcp_hsdp_tp_profile")
    parser.add_argument("--backend", default="cpu:gloo,cuda:nccl")
    parser.add_argument("--dp-replicate", type=int, default=2)
    parser.add_argument("--dp-shard", type=int, default=2)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--dim", type=int, default=6144)
    parser.add_argument("--hidden-dim", type=int, default=16384)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-saves", type=int, default=3)
    parser.add_argument("--warmup-saves", type=int, default=1)
    parser.add_argument("--writer-threads", type=int, default=8)
    parser.add_argument("--sync-files", action="store_true")
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument(
        "--discard-checkpoints-after-save",
        action="store_true",
        help="Delete each checkpoint directory after timing/profiling it.",
    )
    parser.add_argument(
        "--variants",
        default=(
            "sync,"
            "async_thread,"
            "async_thread_plan,"
            "async_thread_plan_meta,"
            "async_thread_plan_meta_pinned,"
            "async_process_plan_meta_pinned"
        ),
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-variants", default="sync,async_thread_plan_meta_pinned")
    parser.add_argument("--log-collectives", action="store_true", default=True)
    parser.add_argument("--no-log-collectives", dest="log_collectives", action="store_false")
    return parser.parse_args()


def dtype_from_arg(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def reset_dcp_plan_caches() -> None:
    for attr in (
        "_cached_save_plan",
        "_cached_final_save_plan",
        "_cached_all_plans",
        "_cached_global_plan",
        "_cached_global_metadata",
        "_cached_metadata",
    ):
        cache = getattr(SavePlanner, attr, None)
        if cache is not None:
            cache.clear()


def set_cached_metadata_env(enabled: bool) -> str | None:
    old = os.environ.get("TORCH_DCP_ENABLE_CACHED_META")
    if enabled:
        os.environ["TORCH_DCP_ENABLE_CACHED_META"] = "1"
    else:
        os.environ.pop("TORCH_DCP_ENABLE_CACHED_META", None)
    return old


def restore_cached_metadata_env(old: str | None) -> None:
    if old is None:
        os.environ.pop("TORCH_DCP_ENABLE_CACHED_META", None)
    else:
        os.environ["TORCH_DCP_ENABLE_CACHED_META"] = old


def build_model_and_optimizer(args: argparse.Namespace, rank: int):
    device = torch.device(f"cuda:{rank}")
    dtype = dtype_from_arg(args.dtype)
    use_data_parallel = args.dp_replicate * args.dp_shard > 1
    if use_data_parallel:
        mesh = init_device_mesh(
            "cuda",
            (args.dp_replicate, args.dp_shard, args.tp),
            mesh_dim_names=("dp_replicate", "dp_shard", "tp"),
        )
        hsdp_mesh = mesh["dp_replicate", "dp_shard"]
        tp_mesh = mesh["tp"]
    else:
        mesh = init_device_mesh("cuda", (args.tp,), mesh_dim_names=("tp",))
        hsdp_mesh = None
        tp_mesh = mesh

    model = BigMLP(
        args.layers,
        args.dim,
        args.hidden_dim,
        device=device,
        dtype=dtype,
    )
    tp_plan = {}
    for i in range(args.layers):
        tp_plan[f"layers.{i}.w1"] = ColwiseParallel()
        tp_plan[f"layers.{i}.w3"] = ColwiseParallel()
        tp_plan[f"layers.{i}.w2"] = RowwiseParallel()
    model = parallelize_module(model, tp_mesh, tp_plan)

    if hsdp_mesh is not None:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=dtype if dtype != torch.float32 else None,
            reduce_dtype=torch.float32 if dtype != torch.float32 else None,
        )
        for layer in model.layers:
            fully_shard(layer, mesh=hsdp_mesh, mp_policy=mp_policy)
        fully_shard(model, mesh=hsdp_mesh, mp_policy=mp_policy)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    return model, optimizer


def initialize_optimizer_state(model, optimizer, args: argparse.Namespace, rank: int) -> None:
    dtype = dtype_from_arg(args.dtype)
    x = torch.randn(
        args.batch,
        args.seq_len,
        args.dim,
        device=f"cuda:{rank}",
        dtype=dtype,
    )
    out = model(x)
    loss = out.float().square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()


def make_state_dict(model, optimizer):
    model_sd, optim_sd = get_state_dict(
        model,
        optimizer,
        options=StateDictOptions(full_state_dict=False),
    )
    return {"model": model_sd, "optimizer": optim_sd}


def make_writer(path: Path, args: argparse.Namespace, pinned_staging: bool):
    return dcp.FileSystemWriter(
        path,
        thread_count=args.writer_threads,
        sync_files=args.sync_files,
        cache_staged_state_dict=pinned_staging,
        overwrite=True,
    )


def make_planner(variant: str):
    return DefaultSavePlanner(enable_plan_caching="_plan" in variant)


def maybe_profile(args: argparse.Namespace, variant: str, step: int, rank: int, out_dir: Path):
    should_profile = args.profile and variant in set(args.profile_variants.split(","))
    if not should_profile:
        return contextlib.nullcontext(None)
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    trace_dir = out_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def profiled():
        with profile(
            activities=activities,
            profile_memory=True,
            record_shapes=False,
            with_stack=False,
        ) as prof:
            yield prof
        trace_path = trace_dir / f"{variant}_step{step}_rank{rank}.json"
        prof.export_chrome_trace(str(trace_path))

    return profiled()


def save_once(
    state_dict: dict[str, Any],
    ckpt_dir: Path,
    variant: str,
    args: argparse.Namespace,
    writer,
    planner,
) -> tuple[float, float]:
    use_async = variant.startswith("async_")
    use_process = variant.startswith("async_process")

    start = time.perf_counter()
    if use_async:
        checkpointer_type = (
            AsyncCheckpointerType.PROCESS if use_process else AsyncCheckpointerType.THREAD
        )
        fut = dcp.async_save(
            state_dict,
            checkpoint_id=ckpt_dir,
            storage_writer=writer,
            planner=planner,
            async_checkpointer_type=checkpointer_type,
        )
        submit_done = time.perf_counter()
        fut.result()
        end = time.perf_counter()
        return submit_done - start, end - start

    dcp.save(state_dict, checkpoint_id=ckpt_dir, storage_writer=writer, planner=planner)
    end = time.perf_counter()
    return end - start, end - start


def summarize_times(rows: list[dict[str, Any]], rank: int) -> None:
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, rows)
    if rank != 0:
        return
    flat = [row for rank_rows in gathered for row in rank_rows]
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in flat:
        if row["measured"]:
            by_variant.setdefault(row["variant"], []).append(row)
    print("\nMeasured checkpoint timings, max over ranks per step:")
    for variant, entries in by_variant.items():
        by_step: dict[int, list[dict[str, Any]]] = {}
        for entry in entries:
            by_step.setdefault(entry["step"], []).append(entry)
        blocking = []
        total = []
        for step_entries in by_step.values():
            blocking.append(max(e["blocking_s"] for e in step_entries))
            total.append(max(e["total_s"] for e in step_entries))
        print(
            f"  {variant:32s} "
            f"blocking_avg={sum(blocking) / len(blocking):8.3f}s "
            f"total_avg={sum(total) / len(total):8.3f}s "
            f"steps={sorted(by_step)}"
        )


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    expected = args.dp_replicate * args.dp_shard * args.tp
    if world_size != expected:
        raise RuntimeError(f"world_size={world_size} does not match mesh product={expected}")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(args.backend)
    out_dir = Path(args.out_dir)

    if rank == 0 and not args.keep_checkpoints:
        shutil.rmtree(out_dir, ignore_errors=True)
    dist.barrier()
    out_dir.mkdir(parents=True, exist_ok=True)

    model, optimizer = build_model_and_optimizer(args, local_rank)
    initialize_optimizer_state(model, optimizer, args, local_rank)
    state_dict = make_state_dict(model, optimizer)

    local_param_numel = sum(p.numel() for p in model.parameters())
    total_param_numel = torch.tensor([local_param_numel], device=f"cuda:{local_rank}")
    dist.all_reduce(total_param_numel, op=dist.ReduceOp.MAX)
    if rank == 0:
        approx_params = args.layers * 3 * args.dim * args.hidden_dim
        print(
            "Model ready: "
            f"approx_global_params={approx_params / 1e9:.2f}B, "
            f"max_rank_param_numel={int(total_param_numel.item()) / 1e9:.2f}B"
        )

    rows: list[dict[str, Any]] = []
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    log_path = out_dir / "collectives" / f"rank{rank}.jsonl"
    with CollectiveLogger(log_path, rank, args.log_collectives) as comm_logger:
        for variant in variants:
            reset_dcp_plan_caches()
            meta_cache = "_meta" in variant
            old_env = set_cached_metadata_env(meta_cache)
            try:
                pinned_staging = "_pinned" in variant
                writer = make_writer(
                    out_dir / "checkpoints" / variant / "_initial",
                    args,
                    pinned_staging,
                )
                planner = make_planner(variant)
                for step in range(args.num_saves):
                    ckpt_dir = out_dir / "checkpoints" / variant / f"step_{step:03d}"
                    if rank == 0 and ckpt_dir.exists() and not args.keep_checkpoints:
                        shutil.rmtree(ckpt_dir, ignore_errors=True)
                    dist.barrier()
                    with comm_logger.annotate(variant, step):
                        with maybe_profile(args, variant, step, rank, out_dir):
                            with record_function(f"dcp_save/{variant}/step_{step}"):
                                blocking_s, total_s = save_once(
                                    state_dict,
                                    ckpt_dir,
                                    variant,
                                    args,
                                    writer,
                                    planner,
                                )
                    torch.cuda.synchronize()
                    dist.barrier()
                    if args.discard_checkpoints_after_save:
                        if rank == 0:
                            shutil.rmtree(ckpt_dir, ignore_errors=True)
                        dist.barrier()
                    measured = step >= args.warmup_saves
                    rows.append(
                        {
                            "rank": rank,
                            "variant": variant,
                            "step": step,
                            "blocking_s": blocking_s,
                            "total_s": total_s,
                            "measured": measured,
                        }
                    )
                    if rank == 0:
                        print(
                            f"{variant} step={step} "
                            f"blocking={blocking_s:.3f}s total={total_s:.3f}s "
                            f"{'(warmup)' if not measured else ''}"
                        )
            finally:
                restore_cached_metadata_env(old_env)

    metrics_path = out_dir / f"metrics_rank{rank}.json"
    metrics_path.write_text(json.dumps(rows, indent=2))
    summarize_times(rows, rank)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
