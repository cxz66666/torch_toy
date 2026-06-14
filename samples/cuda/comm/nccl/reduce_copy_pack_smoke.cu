/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Minimal standalone smoke benchmark for the pack-copy core used by NCCL
 * reduceCopy-style paths. One CUDA block represents one NCCL channel.
 *
 * This intentionally excludes FIFO/head/tail/proxy work. It only compares the
 * steady copy loop, now driven through the same SIMPLE chunk/slice sizing used
 * by AllGather ring+SIMPLE:
 *   NCCL_STEPS=8, StepPerSlice=2, SlicePerChunk=2, default SIMPLE buffer=4MiB.
 *
 * Modes:
 *   - aligned16: 16-byte packs, 16B-aligned src/dst
 *   - fallback1: 1-byte packs, src/dst shifted by 8B from a 16B boundary
 *   - fallback4: 4-byte packs, src/dst shifted by 8B from a 16B boundary
 *
 * Compile:
 *
 *   nvcc -O3 -std=c++17 -arch=sm_90 \
 *     reduce_copy_pack_smoke.cu -o reduce_copy_pack_smoke
 *
 * Run:
 *
 *   ./reduce_copy_pack_smoke \
 *     --total-bytes 10242424 --iters 2000 --warmup 100 \
 *     --channels 1,2,4,8,16,32,64
 *************************************************************************/

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#define CUDA_CHECK(cmd)                                                                              \
  do {                                                                                               \
    cudaError_t err__ = (cmd);                                                                       \
    if (err__ != cudaSuccess) {                                                                      \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
      std::exit(1);                                                                                  \
    }                                                                                                \
  } while (0)

namespace {

constexpr int kWarpSize = 32;
constexpr int kNccLUnrollSm80Plus = 8;
constexpr int kNcclSteps = 8;
constexpr size_t kNcclSimpleGrainBytes = 512;

__host__ __device__ size_t divUpSize(size_t a, size_t b) {
  return (a + b - 1) / b;
}

__host__ __device__ size_t minSize(size_t a, size_t b) {
  return a < b ? a : b;
}

__host__ __device__ size_t maxSize(size_t a, size_t b) {
  return a > b ? a : b;
}

struct ChannelPart {
  size_t offset;
  size_t bytes;
};

__host__ __device__ ChannelPart calcChannelPart(size_t totalBytes, int channels, int channel) {
  const size_t cells = divUpSize(totalBytes, kNcclSimpleGrainBytes);
  const size_t cellsPerChannel = divUpSize(cells, static_cast<size_t>(channels));
  const size_t cellOffset = minSize(static_cast<size_t>(channel) * cellsPerChannel, cells);
  const size_t cellEnd = minSize(cellOffset + cellsPerChannel, cells);
  const size_t byteOffset = cellOffset * kNcclSimpleGrainBytes;
  const size_t byteEnd = minSize(cellEnd * kNcclSimpleGrainBytes, totalBytes);
  ChannelPart part = {byteOffset, byteEnd > byteOffset ? byteEnd - byteOffset : 0};
  return part;
}

template <int Bytes>
struct Pack;

template <>
struct Pack<1> {
  uint8_t x;
};

template <>
struct Pack<4> {
  uint32_t x;
};

template <>
struct alignas(16) Pack<16> {
  uint32_t x;
  uint32_t y;
  uint32_t z;
  uint32_t w;
};

template <int Bytes>
__device__ __forceinline__ Pack<Bytes> loadVolatileGlobal(const uint8_t* ptr);

template <>
__device__ __forceinline__ Pack<1> loadVolatileGlobal<1>(const uint8_t* ptr) {
  Pack<1> out;
  uint32_t tmp;
  asm volatile("ld.volatile.global.u8 %0, [%1];" : "=r"(tmp) : "l"(ptr) : "memory");
  out.x = static_cast<uint8_t>(tmp);
  return out;
}

template <>
__device__ __forceinline__ Pack<4> loadVolatileGlobal<4>(const uint8_t* ptr) {
  Pack<4> out;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(out.x) : "l"(ptr) : "memory");
  return out;
}

template <>
__device__ __forceinline__ Pack<16> loadVolatileGlobal<16>(const uint8_t* ptr) {
  Pack<16> out;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w)
               : "l"(ptr)
               : "memory");
  return out;
}

template <int Bytes>
__device__ __forceinline__ void storeGlobal(uint8_t* ptr, Pack<Bytes> value);

template <>
__device__ __forceinline__ void storeGlobal<1>(uint8_t* ptr, Pack<1> value) {
  asm volatile("st.global.u8 [%0], %1;" ::"l"(ptr), "r"(static_cast<uint32_t>(value.x)) : "memory");
}

template <>
__device__ __forceinline__ void storeGlobal<4>(uint8_t* ptr, Pack<4> value) {
  asm volatile("st.global.u32 [%0], %1;" ::"l"(ptr), "r"(value.x) : "memory");
}

template <>
__device__ __forceinline__ void storeGlobal<16>(uint8_t* ptr, Pack<16> value) {
  asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(value.x), "r"(value.y), "r"(value.z),
               "r"(value.w)
               : "memory");
}

template <int BytePerPack, int Unroll>
__device__ __forceinline__ void copyPacks(uint8_t* dst, const uint8_t* src, size_t nBytes) {
  const int thread = threadIdx.x;
  const int lane = thread & (kWarpSize - 1);
  const int warp = thread / kWarpSize;
  const int nWarps = blockDim.x / kWarpSize;
  const size_t nPacks = nBytes / BytePerPack;
  const size_t packsPerBlockIter = static_cast<size_t>(nWarps) * kWarpSize * Unroll;
  const size_t fullIters = nPacks / packsPerBlockIter;

  for (size_t iter = 0; iter < fullIters; ++iter) {
    const size_t base = iter * packsPerBlockIter + static_cast<size_t>(warp) * kWarpSize * Unroll + lane;
#pragma unroll
    for (int u = 0; u < Unroll; ++u) {
      const size_t pack = base + static_cast<size_t>(u) * kWarpSize;
      const size_t offset = pack * BytePerPack;
      Pack<BytePerPack> value = loadVolatileGlobal<BytePerPack>(src + offset);
      storeGlobal<BytePerPack>(dst + offset, value);
    }
  }

  const size_t tailStart = fullIters * packsPerBlockIter;
  for (size_t pack = tailStart + thread; pack < nPacks; pack += blockDim.x) {
    const size_t offset = pack * BytePerPack;
    Pack<BytePerPack> value = loadVolatileGlobal<BytePerPack>(src + offset);
    storeGlobal<BytePerPack>(dst + offset, value);
  }
}

template <int BytePerPack>
__device__ __forceinline__ void copyTailBytes(uint8_t* dst, const uint8_t* src, size_t nBytes) {
  const size_t tailStart = (nBytes / BytePerPack) * BytePerPack;
  for (size_t byte = tailStart + threadIdx.x; byte < nBytes; byte += blockDim.x) {
    Pack<1> value = loadVolatileGlobal<1>(src + byte);
    storeGlobal<1>(dst + byte, value);
  }
}

__host__ __device__ size_t calcNcclSimpleChunkBytes(size_t scalarBytes, int slicePerChunk, int stepPerSlice,
                                                    size_t simpleBuffBytes) {
  const size_t stepSizeElts = simpleBuffBytes / kNcclSteps / scalarBytes;
  return stepSizeElts * static_cast<size_t>(stepPerSlice) * static_cast<size_t>(slicePerChunk) * scalarBytes;
}

__host__ __device__ size_t calcNcclSimpleSliceBytes(size_t chunkBytes, size_t scalarBytes, int slicePerChunk,
                                                    int stepPerSlice, size_t simpleBuffBytes) {
  const size_t chunkElts = chunkBytes / scalarBytes;
  const size_t stepSizeElts = simpleBuffBytes / kNcclSteps / scalarBytes;
  const size_t baseSliceElts = stepSizeElts * static_cast<size_t>(stepPerSlice);

  // Mirrors Primitives::genericOp:
  //   sliceSize = max(divUp(nelem, 16*SlicePerChunk)*16, sliceSize/32)
  // where nelem and sliceSize are in T elements.
  const size_t balancedSliceElts = divUpSize(chunkElts, 16 * static_cast<size_t>(slicePerChunk)) * 16;
  const size_t floorSliceElts = baseSliceElts / 32;
  return maxSize(balancedSliceElts, floorSliceElts) * scalarBytes;
}

__host__ __device__ size_t countNcclSimpleSlices(size_t bytes, size_t scalarBytes, int slicePerChunk, int stepPerSlice,
                                                 size_t simpleBuffBytes) {
  const size_t chunkLimitBytes = calcNcclSimpleChunkBytes(scalarBytes, slicePerChunk, stepPerSlice, simpleBuffBytes);
  size_t count = 0;
  size_t chunkOffset = 0;
  while (chunkOffset < bytes) {
    const size_t chunkBytes = minSize(chunkLimitBytes, bytes - chunkOffset);
    const size_t sliceBytes = calcNcclSimpleSliceBytes(chunkBytes, scalarBytes, slicePerChunk, stepPerSlice,
                                                       simpleBuffBytes);
    size_t sliceOffset = 0;
    for (int slice = 0; slice < slicePerChunk && sliceOffset < chunkBytes; ++slice) {
      const size_t workBytes = minSize(sliceBytes, chunkBytes - sliceOffset);
      sliceOffset += workBytes;
      count += 1;
    }
    chunkOffset += chunkBytes;
  }
  return count;
}

template <int BytePerPack, int Unroll>
__global__ __launch_bounds__(512, 1) void reduceCopyPackSmokeKernel(
  uint8_t* dstBase, const uint8_t* srcBase, size_t bytesPerChannel, size_t totalBytes, size_t channelStride,
  size_t pointerShift, size_t scalarBytes, int slicePerChunk, int stepPerSlice, size_t simpleBuffBytes,
  bool fixedTotal) {
  const int channel = blockIdx.x;
  ChannelPart part;
  if (fixedTotal) {
    part = calcChannelPart(totalBytes, gridDim.x, channel);
  } else {
    part = {static_cast<size_t>(channel) * channelStride, bytesPerChannel};
  }

  uint8_t* dst = dstBase + pointerShift + part.offset;
  const uint8_t* src = srcBase + pointerShift + part.offset;
  const size_t channelBytes = part.bytes;

  const size_t chunkLimitBytes = calcNcclSimpleChunkBytes(scalarBytes, slicePerChunk, stepPerSlice, simpleBuffBytes);
  size_t chunkOffset = 0;

  while (chunkOffset < channelBytes) {
    const size_t chunkBytes = minSize(chunkLimitBytes, channelBytes - chunkOffset);
    const size_t sliceBytes =
      calcNcclSimpleSliceBytes(chunkBytes, scalarBytes, slicePerChunk, stepPerSlice, simpleBuffBytes);
    size_t sliceOffset = 0;

    for (int slice = 0; slice < slicePerChunk && sliceOffset < chunkBytes; ++slice) {
      const size_t workBytes = minSize(sliceBytes, chunkBytes - sliceOffset);
      uint8_t* sliceDst = dst + chunkOffset + sliceOffset;
      const uint8_t* sliceSrc = src + chunkOffset + sliceOffset;
      copyPacks<BytePerPack, Unroll>(sliceDst, sliceSrc, workBytes);
      copyTailBytes<BytePerPack>(sliceDst, sliceSrc, workBytes);
      sliceOffset += workBytes;
    }

    chunkOffset += chunkBytes;
  }
}

enum class Mode {
  Aligned16,
  Fallback1,
  Fallback4,
};

struct ModeSpec {
  Mode mode;
  const char* name;
  size_t scalarBytes;
  int bytePerPack;
  int unroll;
  size_t pointerShift;
};

struct Options {
  size_t bytesPerChannel = 640512;
  size_t totalBytes = 0;
  int iters = 2000;
  int warmup = 100;
  int threads = 512;
  int device = 0;
  int slicePerChunk = 2;
  int stepPerSlice = 2;
  size_t simpleBuffBytes = 1ull << 22;
  std::vector<int> channels = {1, 2, 4, 8, 16, 32, 64};
  std::string modeFilter = "all";
};

size_t alignUp(size_t value, size_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

std::vector<int> parseChannels(const char* text) {
  std::vector<int> values;
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) values.push_back(std::atoi(item.c_str()));
  }
  values.erase(std::remove_if(values.begin(), values.end(), [](int x) { return x <= 0; }), values.end());
  if (values.empty()) {
    std::fprintf(stderr, "--channels produced an empty list\n");
    std::exit(1);
  }
  return values;
}

void usage(const char* argv0) {
  std::fprintf(stderr,
               "Usage: %s [--bytes-per-channel N] [--iters N] [--warmup N]\n"
               "          [--total-bytes N]\n"
               "          [--mode all|aligned16|fallback1|fallback4]\n"
               "          [--threads N] [--channels 1,2,4,8,16] [--device N]\n"
               "          [--slice-per-chunk N] [--step-per-slice N] [--simple-buff-bytes N]\n",
               argv0);
}

Options parseArgs(int argc, char** argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    auto needValue = [&](const char* flag) -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s needs a value\n", flag);
        usage(argv[0]);
        std::exit(1);
      }
      return argv[++i];
    };

    if (std::strcmp(argv[i], "--bytes-per-channel") == 0) {
      opts.bytesPerChannel = std::strtoull(needValue(argv[i]), nullptr, 10);
    } else if (std::strcmp(argv[i], "--total-bytes") == 0) {
      opts.totalBytes = std::strtoull(needValue(argv[i]), nullptr, 10);
    } else if (std::strcmp(argv[i], "--iters") == 0) {
      opts.iters = std::atoi(needValue(argv[i]));
    } else if (std::strcmp(argv[i], "--warmup") == 0) {
      opts.warmup = std::atoi(needValue(argv[i]));
    } else if (std::strcmp(argv[i], "--threads") == 0) {
      opts.threads = std::atoi(needValue(argv[i]));
    } else if (std::strcmp(argv[i], "--channels") == 0) {
      opts.channels = parseChannels(needValue(argv[i]));
    } else if (std::strcmp(argv[i], "--mode") == 0) {
      opts.modeFilter = needValue(argv[i]);
    } else if (std::strcmp(argv[i], "--device") == 0) {
      opts.device = std::atoi(needValue(argv[i]));
    } else if (std::strcmp(argv[i], "--slice-per-chunk") == 0) {
      opts.slicePerChunk = std::atoi(needValue(argv[i]));
    } else if (std::strcmp(argv[i], "--step-per-slice") == 0) {
      opts.stepPerSlice = std::atoi(needValue(argv[i]));
    } else if (std::strcmp(argv[i], "--simple-buff-bytes") == 0) {
      opts.simpleBuffBytes = std::strtoull(needValue(argv[i]), nullptr, 10);
    } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      usage(argv[0]);
      std::exit(0);
    } else {
      std::fprintf(stderr, "Unknown argument: %s\n", argv[i]);
      usage(argv[0]);
      std::exit(1);
    }
  }

  if (opts.totalBytes == 0 && (opts.bytesPerChannel == 0 || opts.bytesPerChannel % 16 != 0)) {
    std::fprintf(stderr, "--bytes-per-channel must be positive and divisible by 16 when --total-bytes is not set\n");
    std::exit(1);
  }
  if (opts.totalBytes != 0 && opts.totalBytes % 4 != 0) {
    std::fprintf(stderr, "--total-bytes must be divisible by 4 so fallback4 can model fp32 elements\n");
    std::exit(1);
  }
  if (opts.iters <= 0 || opts.warmup < 0) {
    std::fprintf(stderr, "--iters must be positive and --warmup must be non-negative\n");
    std::exit(1);
  }
  if (opts.threads <= 0 || opts.threads % kWarpSize != 0 || opts.threads > 1024) {
    std::fprintf(stderr, "--threads must be a positive multiple of 32 and <= 1024\n");
    std::exit(1);
  }
  if (opts.slicePerChunk <= 0 || opts.stepPerSlice <= 0) {
    std::fprintf(stderr, "--slice-per-chunk and --step-per-slice must be positive\n");
    std::exit(1);
  }
  if (opts.simpleBuffBytes == 0 || opts.simpleBuffBytes % kNcclSteps != 0) {
    std::fprintf(stderr, "--simple-buff-bytes must be positive and divisible by %d\n", kNcclSteps);
    std::exit(1);
  }
  if (opts.modeFilter != "all" && opts.modeFilter != "aligned16" && opts.modeFilter != "fallback1" &&
      opts.modeFilter != "fallback4") {
    std::fprintf(stderr, "--mode must be all, aligned16, fallback1, or fallback4\n");
    std::exit(1);
  }
  return opts;
}

void fillPattern(std::vector<uint8_t>& data) {
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<uint8_t>((i * 131u + 17u) & 0xffu);
  }
}

template <Mode mode>
void launchMode(int channels, int threads, uint8_t* dst, const uint8_t* src, size_t bytesPerChannel, size_t totalBytes,
                size_t channelStride, size_t scalarBytes, int slicePerChunk, int stepPerSlice,
                size_t simpleBuffBytes, bool fixedTotal) {
  if constexpr (mode == Mode::Aligned16) {
    reduceCopyPackSmokeKernel<16, kNccLUnrollSm80Plus>
      <<<channels, threads>>>(dst, src, bytesPerChannel, totalBytes, channelStride, 0, scalarBytes, slicePerChunk,
                              stepPerSlice, simpleBuffBytes, fixedTotal);
  } else if constexpr (mode == Mode::Fallback1) {
    reduceCopyPackSmokeKernel<1, kNccLUnrollSm80Plus * (16 / 1) / 2>
      <<<channels, threads>>>(dst, src, bytesPerChannel, totalBytes, channelStride, 8, scalarBytes, slicePerChunk,
                              stepPerSlice, simpleBuffBytes, fixedTotal);
  } else {
    reduceCopyPackSmokeKernel<4, kNccLUnrollSm80Plus * (16 / 4) / 2>
      <<<channels, threads>>>(dst, src, bytesPerChannel, totalBytes, channelStride, 8, scalarBytes, slicePerChunk,
                              stepPerSlice, simpleBuffBytes, fixedTotal);
  }
}

void launchMode(const ModeSpec& spec, int channels, int threads, uint8_t* dst, const uint8_t* src,
                size_t bytesPerChannel, size_t totalBytes, size_t channelStride, const Options& opts) {
  const bool fixedTotal = opts.totalBytes != 0;
  switch (spec.mode) {
    case Mode::Aligned16:
      launchMode<Mode::Aligned16>(channels, threads, dst, src, bytesPerChannel, totalBytes, channelStride,
                                  spec.scalarBytes, opts.slicePerChunk, opts.stepPerSlice, opts.simpleBuffBytes,
                                  fixedTotal);
      break;
    case Mode::Fallback1:
      launchMode<Mode::Fallback1>(channels, threads, dst, src, bytesPerChannel, totalBytes, channelStride,
                                  spec.scalarBytes, opts.slicePerChunk, opts.stepPerSlice, opts.simpleBuffBytes,
                                  fixedTotal);
      break;
    case Mode::Fallback4:
      launchMode<Mode::Fallback4>(channels, threads, dst, src, bytesPerChannel, totalBytes, channelStride,
                                  spec.scalarBytes, opts.slicePerChunk, opts.stepPerSlice, opts.simpleBuffBytes,
                                  fixedTotal);
      break;
  }
}

bool checkMode(const ModeSpec& spec, int channels, const std::vector<uint8_t>& src, const std::vector<uint8_t>& dst,
               size_t bytesPerChannel, size_t totalBytes, size_t channelStride) {
  const size_t shift = spec.pointerShift;
  if (totalBytes != 0) {
    if (std::memcmp(src.data() + shift, dst.data() + shift, totalBytes) != 0) {
      std::fprintf(stderr, "check failed: mode=%s fixed_total_bytes=%zu\n", spec.name, totalBytes);
      return false;
    }
    return true;
  }
  for (int ch = 0; ch < channels; ++ch) {
    const size_t base = static_cast<size_t>(ch) * channelStride + shift;
    if (std::memcmp(src.data() + base, dst.data() + base, bytesPerChannel) != 0) {
      std::fprintf(stderr, "check failed: mode=%s channel=%d base=%zu\n", spec.name, ch, base);
      return false;
    }
  }
  return true;
}

float benchmarkMode(const ModeSpec& spec, int channels, const Options& opts, uint8_t* dDst, const uint8_t* dSrc,
                    size_t payloadBytes, size_t channelStride) {
  CUDA_CHECK(cudaMemset(dDst, 0, payloadBytes + 256));
  for (int i = 0; i < opts.warmup; ++i) {
    launchMode(spec, channels, opts.threads, dDst, dSrc, opts.bytesPerChannel, opts.totalBytes, channelStride, opts);
  }
  CUDA_CHECK(cudaGetLastError());

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < opts.iters; ++i) {
    launchMode(spec, channels, opts.threads, dDst, dSrc, opts.bytesPerChannel, opts.totalBytes, channelStride, opts);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaGetLastError());

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / static_cast<float>(opts.iters);
}

} // namespace

int main(int argc, char** argv) {
  const Options opts = parseArgs(argc, argv);
  CUDA_CHECK(cudaSetDevice(opts.device));

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, opts.device));
  int smCount = 0;
  int l2CacheBytes = 0;
  int memClockKhz = 0;
  int memBusBits = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, opts.device));
  CUDA_CHECK(cudaDeviceGetAttribute(&l2CacheBytes, cudaDevAttrL2CacheSize, opts.device));
  CUDA_CHECK(cudaDeviceGetAttribute(&memClockKhz, cudaDevAttrMemoryClockRate, opts.device));
  CUDA_CHECK(cudaDeviceGetAttribute(&memBusBits, cudaDevAttrGlobalMemoryBusWidth, opts.device));

  const int maxChannels = *std::max_element(opts.channels.begin(), opts.channels.end());
  const bool fixedTotal = opts.totalBytes != 0;
  const size_t payloadBytes = fixedTotal ? opts.totalBytes : static_cast<size_t>(maxChannels) * alignUp(opts.bytesPerChannel + 256, 256);
  const size_t channelStride = fixedTotal ? 0 : alignUp(opts.bytesPerChannel + 256, 256);
  const size_t allocBytes = payloadBytes + 512;

  std::vector<uint8_t> hSrc(allocBytes);
  std::vector<uint8_t> hDst(allocBytes);
  fillPattern(hSrc);

  uint8_t* dSrc = nullptr;
  uint8_t* dDst = nullptr;
  CUDA_CHECK(cudaMalloc(&dSrc, allocBytes));
  CUDA_CHECK(cudaMalloc(&dDst, allocBytes));
  CUDA_CHECK(cudaMemcpy(dSrc, hSrc.data(), allocBytes, cudaMemcpyHostToDevice));

  const ModeSpec modes[] = {
    {Mode::Aligned16, "aligned16", 1, 16, kNccLUnrollSm80Plus, 0},
    {Mode::Fallback1, "fallback1", 1, 1, kNccLUnrollSm80Plus * (16 / 1) / 2, 8},
    {Mode::Fallback4, "fallback4", 4, 4, kNccLUnrollSm80Plus * (16 / 4) / 2, 8},
  };

  for (const auto& spec : modes) {
    if (opts.modeFilter != "all" && opts.modeFilter != spec.name) continue;
    const size_t workBytes = fixedTotal ? opts.totalBytes : opts.bytesPerChannel;
    if (workBytes % spec.scalarBytes != 0) {
      std::fprintf(stderr, "bytes_per_channel=%zu is not divisible by scalar_bytes=%zu for mode=%s\n",
                   workBytes, spec.scalarBytes, spec.name);
      CUDA_CHECK(cudaFree(dDst));
      CUDA_CHECK(cudaFree(dSrc));
      return 1;
    }
    if (opts.simpleBuffBytes % (static_cast<size_t>(kNcclSteps) * spec.scalarBytes) != 0) {
      std::fprintf(stderr, "simple_buff_bytes=%zu is not divisible by NCCL_STEPS*scalar_bytes=%zu for mode=%s\n",
                   opts.simpleBuffBytes, static_cast<size_t>(kNcclSteps) * spec.scalarBytes, spec.name);
      CUDA_CHECK(cudaFree(dDst));
      CUDA_CHECK(cudaFree(dSrc));
      return 1;
    }
  }

  std::printf("# device=%s sm=%d%d sm_count=%d l2_cache_bytes=%d mem_clock_khz=%d mem_bus_bits=%d\n", prop.name,
              prop.major, prop.minor, smCount, l2CacheBytes, memClockKhz, memBusBits);
  std::printf("# bytes_per_channel=%zu total_bytes=%zu fixed_total=%d threads=%d iters=%d warmup=%d\n",
              opts.bytesPerChannel, opts.totalBytes, fixedTotal ? 1 : 0, opts.threads, opts.iters, opts.warmup);
  std::printf("# mode=%s\n", opts.modeFilter.c_str());
  std::printf("# nccl_simple_slicing nccl_steps=%d slice_per_chunk=%d step_per_slice=%d simple_buff_bytes=%zu\n",
              kNcclSteps, opts.slicePerChunk, opts.stepPerSlice, opts.simpleBuffBytes);
  std::printf("# unrolls match ncclCollUnroll(sm80+)=8 and fallback Unroll*(16/sizeof(T))/2\n");
  std::printf("mode,channels,scalar_bytes,byte_per_pack,unroll,ptr_shift,total_bytes,first_part_bytes,last_part_bytes,"
              "chunk_bytes,first_slice_bytes,first_slices,avg_us,logical_GBps,check\n");

  for (const auto& spec : modes) {
    if (opts.modeFilter != "all" && opts.modeFilter != spec.name) continue;
    const size_t chunkBytes =
      calcNcclSimpleChunkBytes(spec.scalarBytes, opts.slicePerChunk, opts.stepPerSlice, opts.simpleBuffBytes);
    for (int channels : opts.channels) {
      const ChannelPart firstPart = fixedTotal ? calcChannelPart(opts.totalBytes, channels, 0)
                                               : ChannelPart{0, opts.bytesPerChannel};
      const ChannelPart lastPart = fixedTotal ? calcChannelPart(opts.totalBytes, channels, channels - 1)
                                              : ChannelPart{0, opts.bytesPerChannel};
      const size_t firstPartChunkBytes = minSize(chunkBytes, firstPart.bytes);
      const size_t rowFirstSliceBytes = calcNcclSimpleSliceBytes(firstPartChunkBytes, spec.scalarBytes,
                                                                 opts.slicePerChunk, opts.stepPerSlice,
                                                                 opts.simpleBuffBytes);
      const size_t firstSlices = countNcclSimpleSlices(firstPart.bytes, spec.scalarBytes, opts.slicePerChunk,
                                                       opts.stepPerSlice, opts.simpleBuffBytes);
      launchMode(spec, channels, opts.threads, dDst, dSrc, opts.bytesPerChannel, opts.totalBytes, channelStride, opts);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(hDst.data(), dDst, payloadBytes + 256, cudaMemcpyDeviceToHost));
      const bool ok = checkMode(spec, channels, hSrc, hDst, opts.bytesPerChannel, opts.totalBytes, channelStride);

      const float avgMs = benchmarkMode(spec, channels, opts, dDst, dSrc, payloadBytes, channelStride);
      const double avgUs = static_cast<double>(avgMs) * 1000.0;
      const double bytes = fixedTotal ? static_cast<double>(opts.totalBytes) :
                                        static_cast<double>(opts.bytesPerChannel) * static_cast<double>(channels);
      const double gbps = bytes / (static_cast<double>(avgMs) * 1.0e-3) / 1.0e9;
      std::printf("%s,%d,%zu,%d,%d,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%.3f,%.2f,%s\n", spec.name, channels,
                  spec.scalarBytes, spec.bytePerPack, spec.unroll, spec.pointerShift,
                  fixedTotal ? opts.totalBytes : opts.bytesPerChannel * static_cast<size_t>(channels),
                  firstPart.bytes, lastPart.bytes, chunkBytes, rowFirstSliceBytes, firstSlices, avgUs, gbps,
                  ok ? "ok" : "FAIL");
      if (!ok) {
        CUDA_CHECK(cudaFree(dDst));
        CUDA_CHECK(cudaFree(dSrc));
        return 2;
      }
    }
  }

  CUDA_CHECK(cudaFree(dDst));
  CUDA_CHECK(cudaFree(dSrc));
  return 0;
}
