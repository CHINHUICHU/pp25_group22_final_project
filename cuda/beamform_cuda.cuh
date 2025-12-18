// beamform_cuda.cuh
#pragma once

#include <vector>

struct BFParams;

// CUDA beamforming function - replaces the CPU version
void run_beamform_cuda(
    const std::vector<std::vector<std::vector<float>>>& rf,
    const BFParams& p,
    const char* beamfile);
