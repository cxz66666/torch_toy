#include "common.h"
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <iostream>

// Kernel: 驗證並修改共享內存中的數據
__global__ void verify_and_modify_data(float* data, int n, bool* success_flag) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx == 0) {
    *success_flag = true;  // 先假設成功
  }
  __syncthreads();

  if (idx < n) {
    float expected_value = static_cast<float>(idx);
    // 驗證初始值
    if (data[idx] != expected_value) {
      printf("Client verification FAILED at index %d! Expected %f, Got %f\n",
             idx, expected_value, data[idx]);
      *success_flag = false;
    }
    // 修改數據
    data[idx] *= 2.0f;
  }
}

int main() {
  int device_id = 0;
  CHECK_CUDA(cudaSetDevice(device_id));

  // 1. 設置 Socket 客戶端
  int sock_fd;
  struct sockaddr_un addr;

  sock_fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (sock_fd == -1) {
    perror("socket error");
    exit(EXIT_FAILURE);
  }

  memset(&addr, 0, sizeof(struct sockaddr_un));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);

  std::cout << "[Client] Connecting to server..." << std::endl;
  if (connect(sock_fd, (struct sockaddr*)&addr, sizeof(struct sockaddr_un)) ==
      -1) {
    perror("connect error");
    close(sock_fd);
    exit(EXIT_FAILURE);
  }
  std::cout << "[Client] Connected to server." << std::endl;

  // 2. 接收 IPC 句柄
  cudaIpcMemHandle_t ipc_handle;
  if (recv(sock_fd, &ipc_handle, sizeof(cudaIpcMemHandle_t), 0) == -1) {
    perror("recv error");
    close(sock_fd);
    exit(EXIT_FAILURE);
  }
  std::cout << "[Client] Received IPC handle from server." << std::endl;

  // 3. 打開 IPC 句柄 (Import)
  float* d_ptr_imported = nullptr;
  CHECK_CUDA(cudaIpcOpenMemHandle((void**)&d_ptr_imported, ipc_handle,
                                  cudaIpcMemLazyEnablePeerAccess));
  std::cout << "[Client] Imported memory handle, mapped to local address: "
            << static_cast<void*>(d_ptr_imported) << std::endl;

  // 4. 在共享內存上啟動 Kernel
  const int num_elements = BUFFER_SIZE / sizeof(float);
  bool* d_success_flag;
  bool h_success_flag = true;
  CHECK_CUDA(cudaMalloc(&d_success_flag, sizeof(bool)));

  std::cout << "[Client] Launching kernel to verify and modify data..."
            << std::endl;
  verify_and_modify_data<<<(num_elements + 255) / 256, 256>>>(
      d_ptr_imported, num_elements, d_success_flag);

  float* local_ptr;
  CHECK_CUDA(cudaMalloc((void**)&local_ptr, BUFFER_SIZE));
  CHECK_CUDA(cudaMemcpyAsync(local_ptr, d_ptr_imported, BUFFER_SIZE,
                             cudaMemcpyDeviceToDevice));
  std::cout << "cudaMemcpyAsync completed." << std::endl;
  sleep(2);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(&h_success_flag, d_success_flag, sizeof(bool),
                        cudaMemcpyDeviceToHost));

  if (h_success_flag) {
    std::cout << "[Client] Verification SUCCESS. Data modified." << std::endl;
  } else {
    std::cout << "[Client] Verification FAILED." << std::endl;
  }

  // 5. 通知伺服器已完成
  char confirmation_buf = 'K';  // 'K' for "OK"
  if (send(sock_fd, &confirmation_buf, sizeof(confirmation_buf), 0) == -1) {
    perror("send confirmation error");
  } else {
    std::cout << "[Client] Sent confirmation to server." << std::endl;
  }

  // 6. 清理
  CHECK_CUDA(cudaFree(d_success_flag));
  CHECK_CUDA(cudaIpcCloseMemHandle(d_ptr_imported));
  close(sock_fd);
  std::cout << "[Client] Cleaned up and exited." << std::endl;

  return 0;
}
