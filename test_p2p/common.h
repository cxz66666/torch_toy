#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Socket 路徑，使用 UNIX Domain Socket 進行本地行程間通訊
#define SOCKET_PATH "/tmp/cuda_ipc_socket"

// 分配的顯存大小 (例如，1024 個浮點數)
#define BUFFER_SIZE (1024 * 1024 * 1024 * sizeof(float))

// CUDA 錯誤檢查宏
#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#endif  // COMMON_H
