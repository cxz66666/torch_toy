// CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 CUDA_COREDUMP_SHOW_PROGRESS=1
// CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory'
// CUDA_COREDUMP_FILE="/persistent_dir/cuda_coredump_%h.%p.%t"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA error checking macro
#define cuda_check(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA Error at %s:%d - %s: %s\n", __FILE__, __LINE__, #call,      \
             cudaGetErrorString(err));                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Kernel with illegal memory access - accesses memory beyond allocated bounds
__global__ void illegalMemoryAccessKernel(int *data, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // This will cause illegal memory access - accessing beyond allocated memory
  // We allocate 'size' elements but access up to size * 2
  if (idx < size * 2) { // Access twice the allocated size
    for (int i = 0; i < 10000; i++) {
      data[idx - 1000000000 + i] =
          idx; // This will cause illegal access for idx == 0
    }
  }
}

// Simple kernel with no errors
__global__ void normalKernel(int *data, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    data[idx] = idx;
  }
}

int main() {
  printf("CUDA Illegal Memory Access Test\n");
  printf("===============================\n\n");

  int size = 100;
  int *h_data = (int *)malloc(size * sizeof(int));
  int *d_data;

  // Initialize host memory
  for (int i = 0; i < size; i++) {
    h_data[i] = 0;
  }

  // Allocate device memory
  cuda_check(cudaMalloc(&d_data, (unsigned long long)(size) * sizeof(int)));
  cuda_check(
      cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice));

  // Launch kernel with illegal memory access
  int blockSize = 256;
  int numBlocks = (size + blockSize - 1) / blockSize;

  printf("Launching kernel with out-of-bounds access...\n");
  illegalMemoryAccessKernel<<<numBlocks, blockSize>>>(d_data, size);

  normalKernel<<<numBlocks, blockSize>>>(d_data, size);

  cuda_check(
      cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost));
  for (int i = 0; i < 5; i++) {
    printf("%d ", h_data[i]);
  }
  printf("\n");

  // Synchronize to catch any runtime errors
  cuda_check(cudaDeviceSynchronize());

  printf("Test completed.\n");

  // Cleanup
  cuda_check(cudaFree(d_data));
  free(h_data);

  return 0;
}
