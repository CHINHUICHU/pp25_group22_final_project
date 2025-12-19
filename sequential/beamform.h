// beamform.h
#pragma once
#include <vector>
#include "params.h"

// rf[tx][rx][sample] 皆為 float（已 bandpass）
void run_beamform(
    const std::vector<std::vector<std::vector<float>>>& rf,
    const BFParams& p,
    const char* beamfile);

// OpenMP parallelized version
void run_beamform_openmp(
    const std::vector<std::vector<std::vector<float>>>& rf,
    const BFParams& p,
    const char* beamfile);
