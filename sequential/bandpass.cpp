// bandpass.cpp
#include "bandpass.h"
#include <cmath>

using namespace std;

// 41-tap Bandpass FIR 1.5–6 MHz @ fs=13.8889 MHz (Hamming window)
void apply_fir_bp41(vector<float>& x)
{
    static const double h[41] = {
        -0.002056037,  0.000924852, -0.001163920,  0.004113509,  0.001553636,
         0.003690527,  0.002282456, -0.010000161, -0.000456886, -0.026071766,
         0.007272336, -0.010216734,  0.027880677,  0.039142416,  0.010970782,
         0.060208140, -0.101972141,  0.006518833, -0.269375890, -0.067515217,
         0.647999482, -0.067515217, -0.269375890,  0.006518833, -0.101972141,
         0.060208140,  0.010970782,  0.039142416,  0.027880677, -0.010216734,
         0.007272336, -0.026071766, -0.000456886, -0.010000161,  0.002282456,
         0.003690527,  0.001553636,  0.004113509, -0.001163920,  0.000924852,
        -0.002056037
    };

    int N = (int)x.size();
    vector<float> y(N, 0.0f);

    for (int n = 0; n < N; n++) {
        double acc = 0.0;
        for (int k = 0; k < 41; k++) {
            int idx = n - k;
            if (idx >= 0)
                acc += h[k] * x[idx];
        }
        y[n] = (float)acc;
    }
    x.swap(y);
}

void bandpass_apply_all(
    vector<vector<vector<float>>>& rf,
    int Nchan, int Nsample)
{
    for (int tx = 0; tx < Nchan; ++tx) {
        for (int rx = 0; rx < Nchan; ++rx) {
            apply_fir_bp41(rf[tx][rx]);
        }
    }
}
