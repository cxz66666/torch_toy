#include <iostream>
#include <cuda_runtime.h>

#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Kernel functions to perform computation
__global__ void kernel1(int64_t *data, int64_t repeat) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (size_t i = 0; i < repeat; i++) {
        data[idx] += 1;
    }
}

__global__ void kernel2(int64_t *data, int64_t repeat) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (size_t i = 0; i < repeat; i++) {
        data[idx] += 2;
    }
}

__global__ void kernel3(int64_t *data, int64_t repeat) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (size_t i = 0; i < repeat; i++) {
        data[idx] -= 1;
    }
}

int main() {
    const int dataSize = 1024;
    const int printSize = 10;
    int64_t *h_data = new int64_t[dataSize]; // Host data
    int64_t *d_data1, *d_data2; // Device data

    // Initialize host data
    for (int i = 0; i < dataSize; i++) {
        h_data[i] = 0;
    }

    // Allocate memory on the device
    cudaCheck(cudaMalloc((void**)&d_data1, dataSize * sizeof(int64_t)));
    cudaCheck(cudaMalloc((void**)&d_data2, dataSize * sizeof(int64_t)));

    // Transfer data from host to device
    cudaCheck(cudaMemcpy(d_data1, h_data, dataSize * sizeof(int64_t), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_data2, h_data, dataSize * sizeof(int64_t), cudaMemcpyHostToDevice));

    // Define grid and block dimensions
    dim3 blockDim(256);
    dim3 gridDim((dataSize + blockDim.x - 1) / blockDim.x);

    // Create streams and event
    cudaStream_t stream1, stream2;
    cudaEvent_t event1, event2;
    cudaCheck(cudaEventCreate(&event1));
    cudaCheck(cudaEventCreate(&event2));
    int priorityHigh, priorityLow;
    cudaCheck(cudaDeviceGetStreamPriorityRange(&priorityLow, &priorityHigh));
    cudaCheck(cudaStreamCreate(&stream1));
    cudaCheck(cudaStreamCreateWithPriority(&stream2, cudaStreamDefault, priorityHigh));

    // Start stream capture on stream1
    cudaCheck(cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal));

    const int64_t repeat = 1000;

    // Execute kernels with stream capture
    kernel1<<<gridDim, blockDim, 0, stream1>>>(d_data1, repeat);
    cudaCheck(cudaEventRecord(event1, stream1)); // Record event1 after kernel1 execution

    // Use events to synchronize
    cudaCheck(cudaStreamWaitEvent(stream2, event1, 0));
    kernel2<<<gridDim, blockDim, 0, stream2>>>(d_data1, repeat);
    cudaCheck(cudaEventRecord(event2, stream2)); // Record event after kernel2
    kernel3<<<gridDim, blockDim, 0, stream1>>>(d_data2, repeat);
    cudaCheck(cudaStreamWaitEvent(stream1, event2, 0)); // Wait on event2 in stream1
    // End stream capture
    cudaGraph_t graph;
    cudaCheck(cudaStreamEndCapture(stream1, &graph));

    // Instantiate and launch the graph
    cudaGraphExec_t instance;
    cudaCheck(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));
    cudaCheck(cudaGraphLaunch(instance, stream1));

    // Synchronize stream to complete graph execution
    cudaCheck(cudaStreamSynchronize(stream1));

    // Print graph in DOT format
    cudaCheck(cudaGraphDebugDotPrint(graph, "graph.dot", cudaGraphDebugDotFlagsVerbose));

    // Transfer data back from device to host
    cudaCheck(cudaMemcpy(h_data, d_data1, dataSize * sizeof(int64_t), cudaMemcpyDeviceToHost));

    // Display results
    std::cout << "Data after kernel1 and kernel2:" << std::endl;
    for (int i = 0; i < printSize; i++) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    cudaCheck(cudaMemcpy(h_data, d_data2, dataSize * sizeof(int64_t), cudaMemcpyDeviceToHost));
    std::cout << "Data after kernel3:" << std::endl;
    for (int i = 0; i < printSize; i++) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup
    cudaCheck(cudaFree(d_data1));
    cudaCheck(cudaFree(d_data2));
    delete[] h_data;
    cudaCheck(cudaGraphDestroy(graph));
    cudaCheck(cudaGraphExecDestroy(instance));
    cudaCheck(cudaStreamDestroy(stream1));
    cudaCheck(cudaStreamDestroy(stream2));
    cudaCheck(cudaEventDestroy(event1));
    cudaCheck(cudaEventDestroy(event2));
    return 0;
}
