# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an ultrasound image reconstruction project (pp25_group22_final_project) that processes RF ultrasound data through a pipeline of digital signal processing stages to generate ultrasound images. The project contains a sequential C++ implementation with opportunities for parallelization.

## Architecture

The ultrasound reconstruction pipeline consists of four main stages:

1. **RF Data Loading** (`main.cpp`): Reads binary ultrasound RF data from `.DAT` files
2. **Bandpass Filtering** (`bandpass.cpp/h`): Applies 41-tap FIR filter (1.5-6 MHz)
3. **Beamforming** (`beamform.cpp/h`): Core computational stage for spatial focusing
4. **Scan Conversion** (`scan_convert.cpp/h`): Converts to B-mode PNG images

Key components:
- `params.cpp/h`: Parameter loading from test case configuration files
- `main.cpp`: Main pipeline orchestration
- `validate.cpp`: Image validation utility against ground truth

## Build and Run Commands

All commands should be run from the `sequential/` directory:

### Build
```bash
cd sequential
make
```

### Run Test Cases
```bash
# Run specific test case (01, 02, or 03)
make run CASE=01 OUT=image01.png
make run CASE=02 OUT=image02.png
make run CASE=03 OUT=image03.png
```

### Validate Output
```bash
# Validate generated image against ground truth
./validate ../result/image01.png ../truth/01.png
```

### Clean Build
```bash
make clean
```

## Input/Output Structure

- **Input**: `../testcase/XX.DAT` (RF data, not in repo due to size) + `../testcase/XX.txt` (parameters)
- **Intermediate**: `beam_XX.dat` (beamformed data)
- **Output**: `../result/imageXX.png` (final ultrasound image)
- **Validation**: `../truth/XX.png` (ground truth images)

**Note**: The large RF data files (*.DAT) are excluded from the repository due to GitHub's file size limits. Use the setup script to download and extract them:

```bash
./download_testdata.sh
```

This will download compressed files from Google Drive and extract them to the testcase/ directory.

## Key Parameters

Test case parameters (in `testcase/XX.txt`):
- `Nchan`: Number of channels (128)
- `fs`: Sampling frequency (MHz)
- `fc`: Center frequency (MHz)
- `timeoffset`: Time offset (μs)
- `Nsample`: Number of samples per channel
- `bytes_per_sample`: Data type (2=int16, 4=int32)
- `pitch`: Element spacing (mm)
- `soundv`: Speed of sound (mm/μs)

## Performance Considerations

According to project notes, the main parallelization target is `beamform.cpp`, which contains the most computationally intensive operations. The beamforming stage involves nested loops over channels, beams, and samples that can benefit from parallel processing.