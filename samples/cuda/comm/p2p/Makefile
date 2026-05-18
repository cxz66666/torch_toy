# Makefile for CUDA IPC example

# CUDA 安装路径，如果不在默认位置请修改
CUDA_PATH ?= /usr/local/cuda

# 编译器
CXX := /usr/local/cuda/bin/nvcc

# 可执行文件名称
TARGETS := server client

# 编译和链接标志
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
