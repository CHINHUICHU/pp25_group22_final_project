#include "bandpass.h"
#include <cmath>

using namespace std;

// ============================================================
// 原本固定 41-tap FIR：1.5–6 MHz
// ============================================================
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

    int N = x.size();
    vector<float> y(N, 0.0f);

    for (int n = 0; n < N; n++) {
        double acc = 0.0;
        for (int k = 0; k < 41; k++) {
            int idx = n - k;
            if (idx >= 0)
                acc += h[k] * x[idx];
        }
        y[n] = acc;
    }
    x.swap(y);
}


// ============================================================
// 動態 FIR 產生器（Hamming window）
// taps = 41
// ============================================================
static vector<double> make_fir(int taps, double fs, double f1, double f2)
{
    vector<double> h(taps);
    int M = taps - 1;

    double w1 = 2.0 * M_PI * f1 / fs;
    double w2 = 2.0 * M_PI * f2 / fs;

    for (int n = 0; n < taps; n++) {
        int k = n - M/2;

        double ideal =
            (k == 0)
            ? (w2 - w1) / M_PI
            : (sin(w2*k) - sin(w1*k)) / (M_PI*k);

        double w = 0.54 - 0.46 * cos(2.0 * M_PI * n / M);
        h[n] = ideal * w;
    }
    return h;
}


// ============================================================
// 動態 FIR 應用
// ============================================================
void apply_dynamic_fir(std::vector<float>& x,
                       float fs_MHz, float f1_MHz, float f2_MHz)
{
    // FIR 設計必須使用 Hz，因此在這裡做內部轉換
    double fs = fs_MHz * 1e6;
    double f1 = f1_MHz * 1e6;
    double f2 = f2_MHz * 1e6;

    const int taps = 201;
    vector<double> h = make_fir(taps, fs, f1, f2);

    int N = x.size();
    vector<float> y(N);

    for (int n = 0; n < N; n++) {
        double acc = 0.0;
        for (int k = 0; k < taps; k++) {
            int idx = n - k;
            if (idx >= 0)
                acc += h[k] * x[idx];
        }
        y[n] = acc;
    }

    x.swap(y);
}



// ============================================================
// 統一 bandpass 入口：依照 has_bp 切換模式
// ============================================================
void bandpass_apply_all(
    vector<vector<vector<float>>>& rf,
    int Nchan, int Nsample,
    float fs_MHz,
    bool has_bp, float bp_low_MHz, float bp_high_MHz)
{
    for (int tx = 0; tx < Nchan; ++tx) {
        for (int rx = 0; rx < Nchan; ++rx) {

            if (has_bp)
                apply_dynamic_fir(rf[tx][rx],
                                  fs_MHz,       // 全部 MHz
                                  bp_low_MHz,
                                  bp_high_MHz);
            else
                apply_fir_bp41(rf[tx][rx]);
        }
    }
}


