# ============================================
#   Ultrasound Reconstruction (CPU + GPU version)
#   Supports .cpp and .cu files
# ============================================

CXX      = g++
CXXFLAGS = -O3 -std=c++17 -Wall -Wno-unused-variable

NVCC      = /usr/local/cuda-13.0/bin/nvcc
NVCCFLAGS = -O3 -std=c++17

# -------------------------
# Source files
# -------------------------
SEQ_DIR   = ../sequential
CPP_SRCS  = $(SEQ_DIR)/main.cpp $(SEQ_DIR)/params.cpp $(SEQ_DIR)/bandpass.cpp $(SEQ_DIR)/scan_convert.cpp
CUDA_SRCS = beamform.cu
SRCS      = $(CPP_SRCS) $(CUDA_SRCS)

TARGET = ultrasound

# --- Baseline GPU kernel ---
BASELINE_CUDA = beamform_baseline.cu
BASELINE_TARGET = ultrasound_baseline

# --- Validation tool ---
VALIDATE_SRC = validate.cpp
VALIDATE_OBJ = $(VALIDATE_SRC:.cpp=.o)
VALIDATE_EXE = validate

# --- User variables ---
CASE ?= 01
OUT  ?= output.png

# --- Paths ---
TESTDATA  = ../testcase/$(CASE).DAT
TESTPARAM = ../testcase/$(CASE).txt
TESTBEAM  = beam_$(CASE).dat
RESULTDIR = ../result
TESTPNG   = $(RESULTDIR)/$(OUT)

# ============================================
# Build
# ============================================

all: $(VALIDATE_EXE) $(TARGET)
	@echo "Build complete: validate + ultrasound"

$(TARGET): $(CUDA_SRCS) $(CPP_SRCS)
	$(NVCC) $(NVCCFLAGS) -I$(SEQ_DIR) -o $@ $(CUDA_SRCS) $(CPP_SRCS)
	@echo "Built ultrasound (GPU)"

$(VALIDATE_EXE): $(SEQ_DIR)/$(VALIDATE_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $(SEQ_DIR)/$(VALIDATE_SRC)
	@echo "Built validate (CPU)"

# ============================================
# Baseline GPU kernel
# ============================================

baseline: $(BASELINE_TARGET)
	@echo "Built baseline GPU kernel: $(BASELINE_TARGET)"

$(BASELINE_TARGET): $(BASELINE_CUDA) $(CPP_SRCS)
	$(NVCC) $(NVCCFLAGS) -I$(SEQ_DIR) -o $@ $(BASELINE_CUDA) $(CPP_SRCS)
	@echo "Built $(BASELINE_TARGET)"

# ============================================
# Run
# ============================================

run: $(TARGET)
	mkdir -p $(RESULTDIR)
	./$(TARGET) $(TESTDATA) $(TESTPARAM) $(TESTBEAM) $(TESTPNG)

run-baseline: $(BASELINE_TARGET)
	mkdir -p $(RESULTDIR)
	./$(BASELINE_TARGET) $(TESTDATA) $(TESTPARAM) beam_baseline_$(CASE).dat $(RESULTDIR)/baseline_$(CASE).png

# ============================================
# Validate result
# ============================================

check: $(VALIDATE_EXE)
	./$(VALIDATE_EXE) $(RESULTDIR)/$(OUT) ../truth/$(CASE).png

check-baseline: $(VALIDATE_EXE)
	./$(VALIDATE_EXE) $(RESULTDIR)/baseline_$(CASE).png ../truth/$(CASE).png

# ============================================
# Clean
# ============================================

clean:
	rm -f $(TARGET) $(VALIDATE_EXE) $(BASELINE_TARGET) beam_*.dat

.PHONY: all run clean check baseline run-baseline check-baseline
