#pragma once
#include <vector>

// default 41-tap
void apply_fir_bp41(std::vector<float>& x);

// dynamic FIR
void apply_dynamic_fir(std::vector<float>& x,
                       float fs, float f1, float f2);

// rf[tx][rx][sample]
void bandpass_apply_all(
    std::vector<std::vector<std::vector<float>>>& rf,
    int Nchan, int Nsample,
    float fs_MHz,
    bool has_bp, float bp_low_MHz, float bp_high_MHz);

