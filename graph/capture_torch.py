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


# Placeholder input used for capture
static_a = torch.zeros((5,), device="cpu")
static_a = static_a.pin_memory()
static_b = torch.zeros((5,), device="cpu")
static_b = static_b.pin_memory()
static_output = torch.zeros((5,), device="cpu")
static_output = static_output.pin_memory()

def compute():
    a = static_a.to("cuda", non_blocking=True)
    b = static_b.to("cuda", non_blocking=True)
    output = (a + b)
    result = cudaMemcpyAsync(static_output.data_ptr(), output.data_ptr(), output.numel() * output.element_size(), cudaMemcpyDeviceToHost, torch.cuda.current_stream().cuda_stream)
    assert result == 0
    return static_output

# Warmup before capture
# torch的default stream和创建出来的stream都是non-blocking stream，不会和0号stream同步
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        compute()
torch.cuda.current_stream().wait_stream(s)

# Captures the graph
# To allow capture, automatically sets a side stream as the current stream in the context
with torch.cuda.nvtx.range("capture"):
    with graph_capture(dump_path="graph.dot") as g:
        compute()

# Run the graph
g.replay()
torch.cuda.current_stream().synchronize()
print(static_output)
static_a += 1
static_b += 2
g.replay()
torch.cuda.current_stream().synchronize()
print(static_output)
