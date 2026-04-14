#include <cuda/ptx>
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>

template <int ELEMS_PER_TOKEN, int NUM_TOKENS>
__global__ void tma_2stage_ring_kernel(const half *g_in, half *g_out) {
#if __CUDA_ARCH__ >= 900
  constexpr int kStages = 2;
  constexpr int kInFlightS2G = 1; // 2-stage ring里通常保留1个in-flight
  constexpr uint32_t kBytes = ELEMS_PER_TOKEN * sizeof(half);

  // 2个slot的shared ring buffer
  extern __shared__ __align__(16) unsigned char smem_raw[];
  half *smem_slots =
      reinterpret_cast<half *>(smem_raw); // [kStages][ELEMS_PER_TOKEN]

  // [stage][0] producer->consumer, [stage][1] consumer->producer
  __shared__ alignas(8) uint64_t ring_mbarrier[kStages][2];

  if (threadIdx.x == 0) {
    for (int s = 0; s < kStages; ++s) {
      cuda::ptx::mbarrier_init(&ring_mbarrier[s][0], 1);
      cuda::ptx::mbarrier_init(&ring_mbarrier[s][1], 1);
    }
    // 让mbarrier初始化结果对async proxy可见（TMA会走async proxy）
    cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
  }
  __syncthreads();

  // Producer: thread 0，负责 G2S TMA
  if (threadIdx.x == 0) {
    int p_stage = 0;
    uint32_t p_wait_cons_parity = 0;

    for (int t = 0; t < NUM_TOKENS; ++t) {
      // 等consumer释放当前stage
      while (!cuda::ptx::mbarrier_try_wait_parity(&ring_mbarrier[p_stage][1],
                                                  p_wait_cons_parity)) {
      }

      half *smem_dst = smem_slots + p_stage * ELEMS_PER_TOKEN;
      const half *gmem_src = g_in + t * ELEMS_PER_TOKEN;

      // launch G2S TMA
      cuda::ptx::cp_async_bulk(cuda::ptx::space_shared, cuda::ptx::space_global,
                               reinterpret_cast<void *>(smem_dst),
                               reinterpret_cast<const void *>(gmem_src), kBytes,
                               &ring_mbarrier[p_stage][0]);

      // 告诉mbarrier本次expect的tx bytes，consumer随后wait parity
      cuda::ptx::mbarrier_arrive_expect_tx(
          cuda::ptx::sem_release, cuda::ptx::scope_cta, cuda::ptx::space_shared,
          &ring_mbarrier[p_stage][0], kBytes);

      // stage推进 + parity翻转
      p_stage += 1;
      if (p_stage == kStages) {
        p_stage = 0;
        p_wait_cons_parity ^= 1;
      }
    }
  }

  // Consumer: thread 32，负责消费shared并做 S2G TMA
  if (threadIdx.x == 32) {
    int c_stage = 0;
    uint32_t c_wait_prod_parity = 0;
    int in_flight_s2g = 0;
    int last_issued_stage = -1;

    for (int t = 0; t < NUM_TOKENS; ++t) {
      // 等producer把当前stage的数据搬到shared
      while (!cuda::ptx::mbarrier_try_wait_parity(&ring_mbarrier[c_stage][0],
                                                  c_wait_prod_parity)) {
      }

      half *smem_src = smem_slots + c_stage * ELEMS_PER_TOKEN;
      half *gmem_dst = g_out + t * ELEMS_PER_TOKEN;

      // 这里模拟普通线程对shared有写（可选）
      smem_src[0] = __hadd(smem_src[0], __float2half(1.0f));

      // 关键：普通shared写 -> async proxy可见
      cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);

      // launch S2G TMA
      cuda::ptx::cp_async_bulk(cuda::ptx::space_global, cuda::ptx::space_shared,
                               reinterpret_cast<void *>(gmem_dst),
                               reinterpret_cast<const void *>(smem_src),
                               kBytes);

      // S2G用group语义管理in-flight
      cuda::ptx::cp_async_bulk_commit_group();
      in_flight_s2g += 1;
      last_issued_stage = c_stage;

      if (in_flight_s2g > kInFlightS2G) {
        // 等最老的一批读完shared，再释放对应slot
        cuda::ptx::cp_async_bulk_wait_group_read(
            cuda::ptx::n32_t<kInFlightS2G>{});
        int notify_stage = (c_stage - kInFlightS2G + kStages) % kStages;
        cuda::ptx::mbarrier_arrive(&ring_mbarrier[notify_stage][1]);
        in_flight_s2g -= 1;
      }

      // stage推进 + parity翻转
      c_stage += 1;
      if (c_stage == kStages) {
        c_stage = 0;
        c_wait_prod_parity ^= 1;
      }
    }

    // flush尾部in-flight
    cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
    if (in_flight_s2g > 0 && last_issued_stage >= 0) {
      cuda::ptx::mbarrier_arrive(&ring_mbarrier[last_issued_stage][1]);
    }
  }
#endif
}

int main() {
  constexpr int kElemsPerToken = 1024;
  constexpr int kNumTokens = 8;
  dim3 grid(1), block(64); // 至少要有thread 0和thread 32
  size_t smem_bytes = 2 * kElemsPerToken * sizeof(half);
  tma_2stage_ring_kernel<kElemsPerToken, kNumTokens>
      <<<grid, block, smem_bytes, stream>>>(g_in, g_out);
}