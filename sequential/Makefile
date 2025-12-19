# ============================================
#   Ultrasound Reconstruction (sequential version)
#   All source files in current folder
#   Testcases located in ../testcase/
#   Output images in ../result/
# ============================================

CXX      = g++
CXXFLAGS = -O3 -std=c++17 -Wall -Wno-unused-variable

# --- Sequential version (default) ---
SRCS  = main.cpp params.cpp bandpass.cpp beamform.cpp scan_convert.cpp
OBJS  = $(SRCS:.cpp=.o)
TARGET = ultrasound

# --- OpenMP version ---
OPENMP_SRCS = main.cpp params.cpp bandpass.cpp beamform_openmp.cpp scan_convert.cpp
OPENMP_OBJS = $(OPENMP_SRCS:.cpp=_openmp.o)
OPENMP_TARGET = ultrasound_openmp
OPENMP_FLAGS = -fopenmp -DUSE_OPENMP

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
all: $(TARGET) $(VALIDATE_EXE)

openmp: $(OPENMP_TARGET) $(VALIDATE_EXE)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

$(OPENMP_TARGET): $(OPENMP_OBJS)
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) -o $@ $(OPENMP_OBJS)

$(VALIDATE_EXE): $(VALIDATE_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(VALIDATE_OBJ)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%_openmp.o: %.cpp
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) -c $< -o $@

# ============================================
# Run (output PNG into ../result/)
# Usage:
#   make run CASE=01 OUT=img01.png
#   make run-openmp CASE=01 OUT=img01.png
# ============================================
run: $(TARGET)
	mkdir -p $(RESULTDIR)
	./$(TARGET) $(TESTDATA) $(TESTPARAM) $(TESTBEAM) $(TESTPNG)

run-openmp: $(OPENMP_TARGET)
	mkdir -p $(RESULTDIR)
	./$(OPENMP_TARGET) $(TESTDATA) $(TESTPARAM) $(TESTBEAM) $(TESTPNG)

# ============================================
# Validate result
# Usage:
#   make check CASE=01 OUT=img01.png
# ============================================
check: $(VALIDATE_EXE)
	./$(VALIDATE_EXE) $(RESULTDIR)/$(OUT) ../truth/$(CASE).png

# ============================================
# Clean
# ============================================
clean:
	rm -f $(OBJS) $(TARGET) $(OPENMP_OBJS) $(OPENMP_TARGET) $(VALIDATE_OBJ) $(VALIDATE_EXE) beam_*.dat

.PHONY: all openmp run run-openmp clean check
