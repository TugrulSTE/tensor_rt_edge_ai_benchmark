# Varsayılan mod: 0 = CPU/g++, 1 = GPU/nvcc (make USE_CUDA=1 şeklinde çağrılabilir)
USE_CUDA ?= 1

TARGET = trt_test

ifeq ($(USE_CUDA), 1)
    # GPU Modu (CUDA Kernel devrede)
    CXX = nvcc
    CXXFLAGS = -O2 -std=c++14 -DUSE_CUDA -I/usr/local/cuda/include
    LDFLAGS = -L/usr/local/cuda/lib64 -lnvinfer -lcudart
    SRCS = main.cu
else
    # CPU Modu (Klasik for döngüsü devrede)
    CXX = g++
    CXXFLAGS = -O2 -std=c++14 -I/usr/local/cuda/include
    LDFLAGS = -L/usr/local/cuda/lib64 -lnvinfer -lcudart -Wl,-rpath,/usr/local/cuda/lib64
    SRCS = main.cpp
endif

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)