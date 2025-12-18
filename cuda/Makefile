# ============================================
#   Ultrasound Reconstruction (CUDA version)
#   CUDA parallelization of beamforming
# ============================================

NVCC     = /usr/local/cuda-13.0/bin/nvcc
CXX      = g++
NCU      ?= ncu
NSYS     ?= nsys

# Add CUDA library paths
CUDA_PATH = /usr/local/cuda-13.0
LDFLAGS   = -L$(CUDA_PATH)/lib64 -lcudart

# CUDA flags
NVCCFLAGS = -O3 -std=c++17 --use_fast_math -Xcompiler -Wall,-Wno-unused-variable

# Auto-detect GPU architecture (fallback to common architectures)
# Note: in some environments `nvidia-smi` may print errors to stdout; strip non-numeric output.
GPU_ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.' | grep -Eo '^[0-9]+' || true)
ifeq ($(GPU_ARCH),)
    # Fallback for common architectures (Turing, Ampere, Ada, Hopper)
    NVCCFLAGS += -gencode arch=compute_75,code=sm_75 \
                 -gencode arch=compute_80,code=sm_80 \
                 -gencode arch=compute_86,code=sm_86 \
                 -gencode arch=compute_89,code=sm_89 \
                 -gencode arch=compute_90,code=sm_90
else
    NVCCFLAGS += -arch=sm_$(GPU_ARCH)
endif

# C++ flags for non-CUDA files
CXXFLAGS = -O3 -std=c++17 -Wall -Wno-unused-variable

# Source files
SEQ_DIR   = ../sequential
CUDA_SRCS = beamform_cuda.cu main_cuda.cu
CPP_SRCS  = $(SEQ_DIR)/params.cpp $(SEQ_DIR)/bandpass.cpp $(SEQ_DIR)/scan_convert.cpp

# Object files
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)
CPP_OBJS  = params.o bandpass.o scan_convert.o

TARGET = ultrasound_cuda

# Validation tool (reuse from sequential)
VALIDATE_EXE = ../sequential/validate

# User variables
CASE ?= 01
OUT  ?= output.png
APERTURE_FNUM ?= 0.0

# Beamforming options
#   APERTURE_FNUM=0.0 disables dynamic aperture gating
#   Example: make run CASE=05 OUT=cuda_05.png APERTURE_FNUM=2.0
NVCCFLAGS += -DAPERTURE_FNUM=$(APERTURE_FNUM)

# Paths
TESTDATA  = ../testcase/$(CASE).DAT
TESTPARAM = ../testcase/$(CASE).txt
TESTBEAM  = beam_$(CASE).dat
RESULTDIR = ../result
TESTPNG   = $(RESULTDIR)/$(OUT)

# ============================================
# Build
# ============================================
all: $(TARGET)

$(TARGET): $(CUDA_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(CUDA_OBJS) $(CPP_OBJS)

# CUDA compilation
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# C++ compilation (copy from sequential and compile)
params.o: $(SEQ_DIR)/params.cpp $(SEQ_DIR)/params.h
	$(CXX) $(CXXFLAGS) -c $(SEQ_DIR)/params.cpp -o $@

bandpass.o: $(SEQ_DIR)/bandpass.cpp $(SEQ_DIR)/bandpass.h
	$(CXX) $(CXXFLAGS) -c $(SEQ_DIR)/bandpass.cpp -o $@

scan_convert.o: $(SEQ_DIR)/scan_convert.cpp $(SEQ_DIR)/scan_convert.h
	$(CXX) $(CXXFLAGS) -I$(SEQ_DIR) -c $(SEQ_DIR)/scan_convert.cpp -o $@

# ============================================
# Run (output PNG into ../result/)
# Usage:
#   make run CASE=01 OUT=img01.png
# ============================================
run: $(TARGET)
	mkdir -p $(RESULTDIR)
	./$(TARGET) $(TESTDATA) $(TESTPARAM) $(TESTBEAM) $(TESTPNG)

# ============================================
# Validate result (compare with truth)
# Usage:
#   make check CASE=01 OUT=img01.png
# ============================================
check:
	$(VALIDATE_EXE) $(RESULTDIR)/$(OUT) ../truth/$(CASE).png

# ============================================
# Profile with nvprof
# Usage:
#   make profile CASE=01
# ============================================
profile: $(TARGET)
	mkdir -p $(RESULTDIR)
	nvprof ./$(TARGET) $(TESTDATA) $(TESTPARAM) $(TESTBEAM) $(RESULTDIR)/profile_$(CASE).png

# ============================================
# Profile with Nsight Compute (ncu)
# Usage:
#   make ncu CASE=05 OUT=cuda_05.png
# Output:
#   ncu_case05.ncu-rep (in this directory)
# ============================================
ncu: $(TARGET)
	@command -v $(NCU) >/dev/null 2>&1 || { \
		echo "Error: '$(NCU)' not found. Install Nsight Compute (ncu) or set NCU=/path/to/ncu."; \
		exit 1; \
	}
	mkdir -p $(RESULTDIR)
	$(NCU) --target-processes all --kernel-name-base function \
		-k "regex:.*(preinterpolate_kernel|beamform_kernel_v7).*" \
		--launch-count 2 --set basic -o ncu_case$(CASE) \
		./$(TARGET) $(TESTDATA) $(TESTPARAM) $(TESTBEAM) $(TESTPNG)

# ============================================
# Profile with Nsight Systems (nsys)
# Usage:
#   make nsys CASE=05 OUT=cuda_05.png
# Output:
#   nsys_case05.nsys-rep / nsys_case05.sqlite (in this directory)
# ============================================
nsys: $(TARGET)
	@command -v $(NSYS) >/dev/null 2>&1 || { \
		echo "Error: '$(NSYS)' not found. Install Nsight Systems (nsys) or set NSYS=/path/to/nsys."; \
		exit 1; \
	}
	mkdir -p $(RESULTDIR)
	$(NSYS) profile -o nsys_case$(CASE) --force-overwrite=true \
		./$(TARGET) $(TESTDATA) $(TESTPARAM) $(TESTBEAM) $(TESTPNG)

# ============================================
# Clean
# ============================================
clean:
	rm -f $(CUDA_OBJS) $(CPP_OBJS) $(TARGET) beam_*.dat

.PHONY: all run clean check profile ncu nsys
