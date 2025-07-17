# Makefile for CUDA IPC example

# CUDA 安裝路徑，如果不在默認位置請修改
CUDA_PATH ?= /usr/local/cuda

# 編譯器
CXX := /usr/local/cuda/bin/nvcc

# 執行檔名稱
TARGETS := server client

# 編譯和鏈接旗標
CPPFLAGS := -I$(CUDA_PATH)/include
LDFLAGS := -L$(CUDA_PATH)/lib64
LDLIBS := -lcudart

# -------------------

all: $(TARGETS)

server: server.cu common.h
	$(CXX) $< -o $@ -gencode arch=compute_90a,code=sm_90a $(CPPFLAGS) $(LDFLAGS) $(LDLIBS)

client: client.cu common.h
	$(CXX) $< -o $@ -gencode arch=compute_90a,code=sm_90a $(CPPFLAGS) $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(TARGETS)

.PHONY: all clean
