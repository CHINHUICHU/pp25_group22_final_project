#pragma once
#include <string>

struct BFParams {
    int   Nchan;             // number of channels
    float fs;                // sampling rate (MHz)
    float fc;                // center frequency (MHz)
    float timeoffset;        // us
    int   Nsample;           // samples per trace
    int   bytes_per_sample;  // 2 = int16, 4 = float
    float pitch;             // mm
    float soundv;            // mm/us

    // ---- 新增：bandpass 參數 ----
    bool  has_bp = false;
    float bp_low = 0.0f;
    float bp_high = 0.0f;
};

BFParams load_params(const char* filename);
