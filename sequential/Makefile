# ============================================
#   Ultrasound Reconstruction (CUDA Version)
#   Source file: main.cu
#   Testcases located in ../testcase/
#   Output images in ../result/
# ============================================

# 設定 CUDA 編譯器
NVCC      = nvcc

# 編譯參數：
# -O3: 最高級別優化
# -std=c++17: 使用 C++17 標準
# -Xcompiler: 將後面的參數傳給底層的 C++ 編譯器 (如 gcc/g++)
NVCCFLAGS = -O3 -std=c++17 -Xcompiler "-Wall -Wno-unused-variable"

# 檔案設定
SRCS    = main.cu
OBJS    = $(SRCS:.cu=.o)
TARGET  = ultrasound_gpu

# --- 使用者變數 (預設跑 CASE 01) ---
CASE ?= 01
OUT  ?= output_gpu.png

# --- 路徑設定 (維持與 CPU 版一致) ---
TESTDATA  = ../testcase/$(CASE).DAT
TESTPARAM = ../testcase/$(CASE).txt
TESTBEAM  = beam_$(CASE).dat
RESULTDIR = ../result
TESTPNG   = $(RESULTDIR)/$(OUT)

# ============================================
# Build Rules
# ============================================
all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(OBJS)

# 編譯 .cu 檔案為 .o
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# ============================================
# Run
# Usage:
#   make run
#   make run CASE=02 OUT=image02.png
# ============================================
run: $(TARGET)
	mkdir -p $(RESULTDIR)
	./$(TARGET) $(TESTDATA) $(TESTPARAM) $(TESTBEAM) $(TESTPNG)

# ============================================
# Clean
# ============================================
clean:
	rm -f $(OBJS) $(TARGET) beam_*.dat

.PHONY: all run clean