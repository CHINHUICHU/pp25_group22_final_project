// bandpass.h
#pragma once
#include <vector>

void apply_fir_bp41(std::vector<float>& x);

// rf[tx][rx][sample]
void bandpass_apply_all(
    std::vector<std::vector<std::vector<float>>>& rf,
    int Nchan, int Nsample);
