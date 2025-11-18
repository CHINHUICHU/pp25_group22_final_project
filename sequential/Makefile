# ============================================
#   Ultrasound Reconstruction (sequential version)
#   All source files in current folder
#   Testcases located in ../testcase/
#   Output images in ../result/
# ============================================

CXX      = g++
CXXFLAGS = -O3 -std=c++17 -Wall -Wno-unused-variable

SRCS  = main.cpp params.cpp bandpass.cpp beamform.cpp scan_convert.cpp
OBJS  = $(SRCS:.cpp=.o)
TARGET = ultrasound

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
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ============================================
# Run (output PNG into ../result/)
# Usage:
#   make run CASE=01 OUT=img01.png
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


