#include <cuda_runtime.h>
#include <iostream>

// Error checking macro and function
#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void addKernel(int *a, int *b, int *c, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

void logMemoryStatus(const char* message) {
    // Memory information variables
    size_t free_mem, total_mem;
    cudaCheck(cudaMemGetInfo(&free_mem, &total_mem));
    float free_gb = free_mem / (float)(1 << 30);  // Convert bytes to gigabytes
    float total_gb = total_mem / (float)(1 << 30);

    // Variables for graph memory attributes
    size_t usedMemCurrent, usedMemHigh, reservedMemCurrent, reservedMemHigh;

    // Retrieve graph memory usage information
    cudaCheck(cudaDeviceGetGraphMemAttribute(0, cudaGraphMemAttrUsedMemCurrent, &usedMemCurrent));
    cudaCheck(cudaDeviceGetGraphMemAttribute(0, cudaGraphMemAttrUsedMemHigh, &usedMemHigh));
    cudaCheck(cudaDeviceGetGraphMemAttribute(0, cudaGraphMemAttrReservedMemCurrent, &reservedMemCurrent));
    // cudaCheck(cudaDeviceGetGraphMemAttribute(0, cudaGraphMemAttrReservedMemHigh, &reservedMemHigh));

    // Print basic memory info
    std::cout << message << " - Free Memory: " << free_gb << " GB, Total Memory: " << total_gb << " GB, Graph Memory Usage: " << usedMemCurrent / (double)(1 << 30) << " GB, Graph Reserved Memory: " << reservedMemCurrent / (double)(1 << 30) << " GB\n";
}

int main() {
    cudaMemPool_t mempool;
    cudaDeviceGetDefaultMemPool(&mempool, 0);
    uint64_t threshold = 0; // UINT64_MAX;
    cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);

    const int N = 1024 * 1024 * 256;
    const int bytes = N * sizeof(int);
    int *a, *b, *c, *h_c;

    // Allocate device memory for a and b
    cudaCheck(cudaMalloc(&a, bytes));
    cudaCheck(cudaMalloc(&b, bytes));

    // Initialize a and b on the host
    int *h_a = new int[N];
    int *h_b = new int[N];
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = i;
    }

    // Copy data from host to device
    cudaCheck(cudaMemcpy(a, h_a, bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(b, h_b, bytes, cudaMemcpyHostToDevice));

    // Allocate host memory for the result
    h_c = new int[N];

    // Create a stream
    cudaStream_t stream;
    cudaCheck(cudaStreamCreate(&stream));

    logMemoryStatus("before capture");

    // Begin graph capture
    cudaGraph_t graph;
    cudaCheck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    // Allocate memory for c during graph capture using cudaMallocAsync
    cudaCheck(cudaMallocAsync(&c, bytes, stream));

    logMemoryStatus("inside capture, after cudaMallocAsync");

    // Launch the add kernel
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    addKernel<<<grid, block, 0, stream>>>(a, b, c, N);

    // Copy the output to CPU using cudaMemcpyAsync
    cudaCheck(cudaMemcpyAsync(h_c, c, bytes, cudaMemcpyDeviceToHost, stream));

    // End graph capture
    cudaCheck(cudaStreamEndCapture(stream, &graph));

    // Launch the graph
    cudaGraphExec_t graphExec;
    cudaCheck(cudaGraphInstantiateWithFlags(&graphExec, graph, cudaGraphInstantiateFlagAutoFreeOnLaunch));

    logMemoryStatus("before execution");

    cudaCheck(cudaGraphLaunch(graphExec, stream));
    cudaCheck(cudaStreamSynchronize(stream));
    logMemoryStatus("after the first execution");

    cudaCheck(cudaGraphLaunch(graphExec, stream));
    cudaCheck(cudaStreamSynchronize(stream));
    logMemoryStatus("after the second execution");

    // Free c using cudaFreeAsync within graph capture
    cudaCheck(cudaFreeAsync(c, stream));
    cudaCheck(cudaStreamSynchronize(stream));
    logMemoryStatus("after cudaFreeAsync");

    cudaCheck(cudaDeviceGraphMemTrim(0));
    logMemoryStatus("after cudaDeviceGraphMemTrim");

    // Output the graph to a .dot file
    cudaCheck(cudaGraphDebugDotPrint(graph, "graph.dot", cudaGraphDebugDotFlagsVerbose));

    // Check result
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            correct = false;
            break;
        }
    }
    if (correct) {
        std::cout << "Results are correct!" << std::endl;
    } else {
        std::cout << "Results are incorrect!" << std::endl;
    }

    // Cleanup

    cudaCheck(cudaGraphDestroy(graph));
    cudaCheck(cudaGraphExecDestroy(graphExec));

    cudaCheck(cudaFree(a));
    cudaCheck(cudaFree(b));
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    cudaCheck(cudaStreamDestroy(stream));

    return 0;
}
