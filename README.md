# pp25_group22_final_project

## Ultrasound Image Reconstruction

This project implements an ultrasound image reconstruction pipeline in C++ that processes RF data through bandpass filtering, beamforming, and scan conversion to generate B-mode ultrasound images.

## Quick Start

1. **Setup test data:**
   ```bash
   ./download_testdata.sh
   ```

2. **Build and run:**
   ```bash
   cd pixel
   make run CASE=01 OUT=image01.png
   ```

3. **Validate results:**
   ```bash
   ./validate ../result/image01.png ../truth/01.png
   ```

For detailed documentation, see [CLAUDE.md](CLAUDE.md).
