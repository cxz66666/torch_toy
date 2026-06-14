# NCCL Symmetric LSA AllReduce Agent Playbook

This note is for future agents repeating or extending the NCCL symmetric LSA allreduce investigation on `h20-gl-1` and `b200-maliva-1`.

## Targets

- H20 host: `h20-gl-1`, conda env `torch2.8`, NCCL source `~/nccl`, tests `~/nccl-tests`.
- B200 host: `b200-maliva-1`, conda env `torch2.10`, NCCL source `~/nccl`, tests `~/nccl-tests`.
- Benchmark surface: 8 GPUs, `all_reduce_perf`, 4 GiB message, `-R 2` symmetric buffer registration, float sum.
- Kernel of interest: `AllReduce_RSxLDMC_AGxSTMC`.
- Source of interest: `~/nccl/src/device/symmetric/all_reduce.cuh`, function `allreduceMultimem`.

## Core Findings

1. Hopper/H20 LSA multimem allreduce really uses `allreduceMultimem`.
   - With `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL`, the 4 GiB run prints:
     `AllReduce [Symmetric]: 4294967296 Bytes -> Kernel AllReduce_RSxLDMC_AGxSTMC ...`
   - Source chain:
     - `src/device/symmetric/all_reduce.cuh`: `ncclSymkRun_AllReduce_RSxLDMC_AGxSTMC`
     - that function calls `allreduceMultimem(gtn, gt, red, input.multimemPtr(...), output.multimemPtr(...), nElts)`
   - A minimal patch to `allreduceMultimem` caused H20 1-CTA performance to collapse from about `458 GB/s` to about `91 GB/s`, which is a strong runtime confirmation.

2. Do not replace the temporary array with immediate load-store.
   - Tested patch:
     ```cpp
     for (int u = 0; u < UnrollPacks; u++) {
       BytePack<BytePerPack> tmp =
           applyLoadMultimem<Red, BytePerPack>(red, inputUptr + cursor + u * WARP_SIZE * BytePerPack);
       multimem_st_global(outputUptr + cursor + u * WARP_SIZE * BytePerPack, tmp);
     }
     ```
   - Baseline code does all unrolled loads into `tmp[UnrollPacks]`, then all stores.
   - Result: immediate-store is much slower at low CTA counts and only catches up at high CTA counts.

3. B200 results had a known contamination during the patched 8-GPU run.
   - `b200-maliva-1` had a root process on GPU5:
     `python3 triton/fused_batch_swiglu/autotune.py`
   - Do not kill root or unknown user jobs without explicit approval.
   - The B200 low-CTA regression is too large to ignore, but exact B200 patched numbers should be treated as contaminated unless rerun on a clean node.

## Critical Pitfalls

### H20 `RPATH` Can Defeat `LD_LIBRARY_PATH`

On H20, `~/nccl-tests/build/all_reduce_perf` had `DT_RPATH=$CONDA_PREFIX/lib`, and `ldd` resolved conda NCCL even when `LD_LIBRARY_PATH=$HOME/nccl/build/lib:...` was set.

Use `LD_PRELOAD` for all runs where self-built NCCL must be guaranteed:

```bash
export LD_PRELOAD=$HOME/nccl/build/lib/libnccl.so.2
export LD_LIBRARY_PATH=$HOME/nccl/build/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
```

Always verify:

```bash
LD_PRELOAD=$HOME/nccl/build/lib/libnccl.so.2 \
LD_LIBRARY_PATH=$HOME/nccl/build/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-} \
ldd ~/nccl-tests/build/all_reduce_perf | grep nccl
```

### Debug Logging Pollutes Performance

Use debug only for path confirmation:

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL NCCL_SYM_CTAS=1 \
./build/all_reduce_perf -b 4294967296 -e 4294967296 -g 8 -R 2 -w 1 -n 2 -c 0
```

Disable debug for benchmark numbers:

```bash
unset NCCL_DEBUG NCCL_DEBUG_SUBSYS
```

### Revert Is Part of the Experiment

If you patch `~/nccl`, restore it before finishing:

```bash
cd ~/nccl
git apply -R /tmp/your.patch
git diff --check
git status --short
make -j24 src.build CUDA_HOME=$CUDA_HOME CUDA_LIB=$CUDA_LIB CUDA_INC=$CUDA_INC NVCC_GENCODE="$NVCC_GENCODE"
git status --short
```

Do not use `git reset --hard` unless the user explicitly asked for it.

## Build Commands

H20:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch2.8
export CUDA_HOME=$CONDA_PREFIX
export CUDA_LIB=$CONDA_PREFIX/lib
export CUDA_INC=$CONDA_PREFIX/include
export NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"

cd ~/nccl
make -j24 src.build CUDA_HOME=$CUDA_HOME CUDA_LIB=$CUDA_LIB CUDA_INC=$CUDA_INC NVCC_GENCODE="$NVCC_GENCODE"

cd ~/nccl-tests
make -j24 CUDA_HOME=$CUDA_HOME CUDA_LIB=$CUDA_LIB CUDA_INC=$CUDA_INC NCCL_HOME=$HOME/nccl/build NVCC_GENCODE="$NVCC_GENCODE"
```

B200:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch2.10
export CUDA_HOME=$CONDA_PREFIX
export CUDA_LIB=$CONDA_PREFIX/lib
export CUDA_INC=$CONDA_PREFIX/include
export NVCC_GENCODE="-gencode=arch=compute_100,code=sm_100"

cd ~/nccl
make -j24 src.build CUDA_HOME=$CUDA_HOME CUDA_LIB=$CUDA_LIB CUDA_INC=$CUDA_INC NVCC_GENCODE="$NVCC_GENCODE"

cd ~/nccl-tests
make -j24 CUDA_HOME=$CUDA_HOME CUDA_LIB=$CUDA_LIB CUDA_INC=$CUDA_INC NCCL_HOME=$HOME/nccl/build NVCC_GENCODE="$NVCC_GENCODE"
```

## Benchmark Command

Use this pattern for each label in `default 1 2 3 4 8 16 32 64`.

```bash
export LD_PRELOAD=$HOME/nccl/build/lib/libnccl.so.2
export LD_LIBRARY_PATH=$HOME/nccl/build/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
unset NCCL_DEBUG NCCL_DEBUG_SUBSYS

if [ "$cta" = default ]; then
  unset NCCL_SYM_CTAS
  label=default
else
  export NCCL_SYM_CTAS=$cta
  label=cta$cta
fi

cd ~/nccl-tests
./build/all_reduce_perf \
  -b 4294967296 -e 4294967296 -f 2 \
  -g 8 -R 2 -d float -o sum \
  -w 5 -n 20 -c 0 \
  -J "$out/$label.json" \
  > "$out/$label.log" 2> "$out/$label.err"
```

## Results From This Run

Average bus bandwidth, 4 GiB, 8 GPUs:

| Machine | CTA | Baseline tmp array | Immediate-store patch |
|---|---:|---:|---:|
| H20 | default | 481.7 GB/s | 177.7 GB/s |
| H20 | 1 | 458.0 GB/s | 90.8 GB/s |
| H20 | 2 | 481.3 GB/s | 177.4 GB/s |
| H20 | 4 | 482.1 GB/s | 331.9 GB/s |
| H20 | 8 | 484.3 GB/s | 470.4 GB/s |
| H20 | 16 | 484.5 GB/s | 474.2 GB/s |
| H20 | 32 | 484.1 GB/s | 476.3 GB/s |
| H20 | 64 | 483.1 GB/s | 476.8 GB/s |
| B200 | default | 765.0 GB/s | 122.5 GB/s |
| B200 | 1 | 298.6 GB/s | 39.5 GB/s |
| B200 | 2 | 530.8 GB/s | 76.7 GB/s |
| B200 | 3 | 714.1 GB/s | 115.9 GB/s |
| B200 | 4 | 722.6 GB/s | 159.4 GB/s |
| B200 | 8 | 738.3 GB/s | 274.1 GB/s |
| B200 | 16 | 746.4 GB/s | 497.6 GB/s |
| B200 | 32 | 751.5 GB/s | 741.3 GB/s |
| B200 | 64 | 759.7 GB/s | 799.1 GB/s |

B200 patched numbers were collected while GPU5 had an external root process. Treat exact B200 values as contaminated; the large low-CTA regression is still directionally clear.

## Artifacts

Local artifacts from this investigation:

- `outputs/nccl_lsa_allreduce_immediate_store_compare_4g.csv`
- `outputs/nccl_lsa_allreduce_immediate_store_compare_4g_h20.png`
- `outputs/nccl_lsa_allreduce_immediate_store_compare_4g_b200.png`
- `work/nccl_lsa_immediate_h20_20260611_213959/`
- `work/nccl_lsa_immediate_b200_20260611_134007/`

## Recommended Next Experiments

1. Rerun B200 on a clean 8-GPU node or after explicit approval to stop the root GPU5 process.
2. If optimizing register pressure, do not collapse load and store phases naively. The array likely helps expose independent multimem loads before stores, which matters especially at low CTA counts.
3. Use compiler resource reports or SASS comparison next:
   - compare register count
   - inspect multimem load/store scheduling
   - check whether load grouping enables better memory-level parallelism
4. Try smaller changes that reduce live range without destroying load grouping, for example split `UnrollPacks` into two groups instead of one load-store pair.
