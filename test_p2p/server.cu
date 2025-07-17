#include "common.h"
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <iostream>
#include <stdio.h>

// Kernel: 初始化設備內存
__global__ void initialize_data(float* data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    data[idx] = static_cast<float>(idx);
  }
}

// Kernel: 打印內存中的值 (用於驗證)
__global__ void print_data(float* data, int n) {
  for (int i = 0; i < 10 && i < n; ++i) {
    printf("Server after client modification, data[%d] = %f\n", i, data[i]);
  }
}

int main() {
  int device_id = 0;
  CHECK_CUDA(cudaSetDevice(device_id));

  // 1. 分配並初始化顯存
  float* d_ptr = nullptr;
  const int num_elements = BUFFER_SIZE / sizeof(float);
  CHECK_CUDA(cudaMalloc(&d_ptr, BUFFER_SIZE));
  std::cout << "[Server] Allocated device memory at: "
            << static_cast<void*>(d_ptr) << std::endl;

  initialize_data<<<(num_elements + 255) / 256, 256>>>(d_ptr, num_elements);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  std::cout << "[Server] Initialized data on GPU." << std::endl;

  // 2. 獲取 IPC 內存句柄 (Export)
  cudaIpcMemHandle_t ipc_handle;
  CHECK_CUDA(cudaIpcGetMemHandle(&ipc_handle, d_ptr));
  std::cout << "[Server] Exported memory handle." << std::endl;

  // 3. 設置 Socket 伺服器
  int server_fd, client_fd;
  struct sockaddr_un addr;

  server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (server_fd == -1) {
    perror("socket error");
    exit(EXIT_FAILURE);
  }

  memset(&addr, 0, sizeof(struct sockaddr_un));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);

  // 刪除舊的 socket 文件
  unlink(SOCKET_PATH);

  if (bind(server_fd, (struct sockaddr*)&addr, sizeof(struct sockaddr_un)) ==
      -1) {
    perror("bind error");
    close(server_fd);
    exit(EXIT_FAILURE);
  }

  if (listen(server_fd, 5) == -1) {
    perror("listen error");
    close(server_fd);
    exit(EXIT_FAILURE);
  }

  std::cout << "[Server] Waiting for a client connection on " << SOCKET_PATH
            << "..." << std::endl;
  client_fd = accept(server_fd, NULL, NULL);
  if (client_fd == -1) {
    perror("accept error");
    close(server_fd);
    exit(EXIT_FAILURE);
  }
  std::cout << "[Server] Client connected." << std::endl;

  // 4. 通過 Socket 發送 IPC 句柄
  if (send(client_fd, &ipc_handle, sizeof(cudaIpcMemHandle_t), 0) == -1) {
    perror("send error");
  } else {
    std::cout << "[Server] Sent IPC handle to client." << std::endl;
  }

  // 5. 等待客戶端完成的信號
  char confirmation_buf;
  if (recv(client_fd, &confirmation_buf, sizeof(confirmation_buf), 0) > 0) {
    std::cout << "[Server] Received confirmation from client." << std::endl;
  }

  // 6. (可選) 檢查客戶端是否修改了數據
  std::cout << "[Server] Checking data after client ran..." << std::endl;
  print_data<<<1, 1>>>(d_ptr, num_elements);
  CHECK_CUDA(cudaDeviceSynchronize());

  // 7. 清理
  close(client_fd);
  close(server_fd);
  unlink(SOCKET_PATH);
  CHECK_CUDA(cudaFree(d_ptr));
  std::cout << "[Server] Cleaned up and exited." << std::endl;

  return 0;
}
