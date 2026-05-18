#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Socket 路径，使用 UNIX Domain Socket 进行本地进程间通信
#define SOCKET_PATH "/tmp/cuda_ipc_socket"

// 分配的显存大小 (例如，1024 个浮点数)
#define BUFFER_SIZE (1024 * 1024 * 1024 * sizeof(float))

// CUDA 错误检查宏
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
