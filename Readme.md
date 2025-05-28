

**test_fsdp2_fp8_allgather:** nsys profile -o test_%p --capture-range=cudaProfilerApi --capture-range-end=stop  -t mpi,cuda,nvtx,ucx  torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint="localhost:0" test_fsdp2_fp8_allgather.py --fp8=true --fp8_all_gather=true --force_recompute=true --precompute_sclae=true

