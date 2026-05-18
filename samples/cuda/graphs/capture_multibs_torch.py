import torch
from contextlib import contextmanager

@contextmanager
def graph_capture(pool=None, stream=None, capture_error_mode: str = "global", dump_path=None):
    g = torch.cuda.CUDAGraph()
    if dump_path is not None:
        g.enable_debug_mode()
    with torch.cuda.graph(cuda_graph=g, pool=pool, stream=stream, capture_error_mode=capture_error_mode):
        yield g
    if dump_path is not None:
        g.debug_dump(dump_path)

import ctypes

# Load the CUDA runtime library
cudart = ctypes.CDLL('libcudart.so')

# Define cudaMemcpyKind enumeration as in the CUDA API
cudaMemcpyHostToHost = 0
cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2
cudaMemcpyDeviceToDevice = 3
cudaMemcpyDefault = 4

# Setup the prototype of the cudaMemcpyAsync function
cudaMemcpyAsync = cudart.cudaMemcpyAsync
cudaMemcpyAsync.argtypes = [
    ctypes.c_void_p,          # void* dst
    ctypes.c_void_p,          # const void* src
    ctypes.c_size_t,          # size_t count
    ctypes.c_int,             # enum cudaMemcpyKind
    ctypes.c_void_p           # cudaStream_t stream
]
cudaMemcpyAsync.restype = ctypes.c_int


static_a = None
static_b = None
static_output = None

def compute(batchsize):
    a = static_a[:batchsize].to("cuda", non_blocking=True)
    b = static_b[:batchsize].to("cuda", non_blocking=True)
    output = (a ** 2 + b * 2)
    result = cudaMemcpyAsync(static_output.data_ptr(), output.data_ptr(), output.numel() * output.element_size(), cudaMemcpyDeviceToHost, torch.cuda.current_stream().cuda_stream)
    assert result == 0
    return static_output[:batchsize]

def report_memory(prefix):
    free, total = torch.cuda.mem_get_info()
    used = total - free
    print(f"{prefix}: Used: {used / 1024 / 1024} MB, Free: {free / 1024 / 1024} MB, Total: {total / 1024 / 1024} MB")


def main():
    import argparse

    global static_a, static_b, static_output

    parser = argparse.ArgumentParser(description="Capture a family of CUDA graphs for multiple batch sizes.")
    parser.add_argument("--max-batch-size", type=int, default=128)
    parser.add_argument("--feature-dim", type=int, default=1024)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This sample requires CUDA.")

    static_a = torch.zeros((args.max_batch_size, args.feature_dim), device="cpu").pin_memory()
    static_b = torch.zeros((args.max_batch_size, args.feature_dim), device="cpu").pin_memory()
    static_output = torch.zeros((args.max_batch_size, args.feature_dim), device="cpu").pin_memory()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(1, args.max_batch_size + 1):
            compute(i)
    torch.cuda.current_stream().wait_stream(s)

    report_memory("Before capture")
    graphs = [0]
    memory_pool = None
    for i in range(1, args.max_batch_size + 1):
        with graph_capture(pool=memory_pool) as g:
            compute(i)
        graphs.append(g)
        memory_pool = g.pool()
    report_memory("After capture")

    replay_batch = min(2, args.max_batch_size)
    static_a[:replay_batch] += 1
    static_b[:replay_batch] += 2
    graphs[replay_batch].replay()
    torch.cuda.current_stream().synchronize()
    print(static_output[:replay_batch])


if __name__ == "__main__":
    main()
