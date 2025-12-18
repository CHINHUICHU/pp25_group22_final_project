# ============================================
#   Ultrasound Reconstruction (CPU + GPU version)
#   Supports .cpp and .cu files
# ============================================

CXX      = g++
CXXFLAGS = -O3 -std=c++17 -Wall -Wno-unused-variable

NVCC      = nvcc
NVCCFLAGS = -O3 -std=c++17 -lineinfo -arch=sm_70

# -------------------------
# Source files
# -------------------------
CPP_SRCS  = main.cpp params.cpp bandpass.cpp scan_convert.cpp
CUDA_SRCS = beamform.cu
SRCS      = $(CPP_SRCS) $(CUDA_SRCS)

CPP_OBJS  = $(CPP_SRCS:.cpp=.o)
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)
OBJS      = $(CPP_OBJS) $(CUDA_OBJS)

TARGET = ultrasound

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

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJS)
	@echo "Built ultrasound (GPU)"

$(VALIDATE_EXE): $(VALIDATE_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(VALIDATE_OBJ)
	@echo "Built validate (CPU)"

# -------------------------
# Compile rules
# -------------------------

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# ============================================
# Run
# ============================================

run: $(TARGET)
	mkdir -p $(RESULTDIR)
	./$(TARGET) $(TESTDATA) $(TESTPARAM) $(TESTBEAM) $(TESTPNG)

# ============================================
# Validate result
# ============================================

check: $(VALIDATE_EXE)
	./$(VALIDATE_EXE) $(RESULTDIR)/$(OUT) ../truth/$(CASE).png

# ============================================
# Clean
# ============================================

clean:
	rm -f $(OBJS) $(TARGET) $(VALIDATE_OBJ) $(VALIDATE_EXE) beam_*.dat

.PHONY: all run clean check
