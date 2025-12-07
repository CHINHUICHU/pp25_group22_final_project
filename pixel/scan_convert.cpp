// scan_convert.cpp
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "scan_convert.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

static void envelope_ma(vector<float>& v, int window = 8)
{
    int N = (int)v.size();
    vector<float> tmp(N);
    for (int i = 0; i < N; i++)
        tmp[i] = fabs(v[i]);

    for (int i = 0; i < N; i++) {
        float s = 0.0f; int c = 0;
        for (int j = -window; j <= window; j++) {
            int k = i + j;
            if (k >= 0 && k < N) {
                s += tmp[k];
                c++;
            }
        }
        v[i] = s / (float)c;
    }
}

void run_scan_conversion(
    const char* beamfile,
    const BFParams& p,
    const char* pngfile)
{
    cout << "[Scan] Loading beamformed data from " << beamfile << " ...\n";

    const int Nsample  = p.Nsample;
    const int upsamp   = 8;
    const int UNsample = Nsample * upsamp;

    // 讀檔大小 → Nbeam
    ifstream fin(beamfile, ios::binary | ios::ate);
    if (!fin) {
        cerr << "[Scan] Cannot open beamfile.\n";
        return;
    }
    streamsize fsize = fin.tellg();
    fin.seekg(0);

    if (fsize % (UNsample * sizeof(float)) != 0) {
        cerr << "[Scan] File size mismatch (UNsample).\n";
        return;
    }

    int Nbeam = (int)(fsize / (UNsample * sizeof(float)));
    cout << "[Scan] Detected Nbeam = " << Nbeam << endl;

    vector<float> bf(Nbeam * UNsample);
    fin.read((char*)bf.data(), bf.size() * sizeof(float));
    fin.close();

    // --------- envelope + global max + log (dB) ---------
    cout << "[Scan] Envelope + log compression (DR=60dB)...\n";

    const float DR  = 60.0f;
    const float eps = 1e-12f;

    float global_max = 0.0f;

    // envelope for each beam, 同時找 global max
    for (int b = 0; b < Nbeam; ++b) {
        vector<float> line(UNsample);
        for (int i = 0; i < UNsample; ++i)
            line[i] = bf[b * UNsample + i];

        envelope_ma(line);

        for (int i = 0; i < UNsample; ++i) {
            float a = line[i];
            if (a > global_max) global_max = a;
            bf[b * UNsample + i] = a;
        }
    }

    if (global_max < eps) global_max = 1.0f;

    // 轉 dB：20*log10(a / global_max)
    for (auto& x : bf) {
        float v = x / global_max;
        if (v < eps) v = eps;
        float db = 20.0f * log10f(v);      // 0 ~ -∞
        if (db < -DR) db = -DR;           // limit to DR
        if (db > 0.0f) db = 0.0f;
        x = db;                           // 存 dB 值
    }

    // --------- 幾何參數 (扇形) ---------
    float apersize = p.Nchan * p.pitch;
    float lambda   = p.soundv / p.fc;
    float dsin     = lambda / apersize / 2.0f;

    float sin_theta_max = dsin * (Nbeam - 1) / 2.0f;
    if (sin_theta_max > 1.0f) sin_theta_max = 1.0f;
    float theta_max = asinf(sin_theta_max);

    float drange      = p.soundv / p.fs / 2.0f / upsamp;
    float rangeoffset = p.timeoffset * p.soundv / 2.0f;
    float r_min       = rangeoffset;
    float r_max       = rangeoffset + (UNsample - 1) * drange;

    cout << "[Scan] theta_max(deg) = "
         << theta_max * 180.0f / 3.14159265f << endl;
    cout << "[Scan] r_min = " << r_min << " mm, r_max = " << r_max << " mm\n";

    const int outW = 2048;
    const int outH = 2048;

    float x_max = r_max * sin_theta_max;
    float dx = (2.0f * x_max) / outW;
    float dy = r_max / outH;

    vector<float> img(outW * outH, -DR);  // dB 空間，初始化為最低強度

    // --------- Scan Conversion (dB 空間) ---------
    cout << "[Scan] Scan converting...\n";

    for (int iy = 0; iy < outH; ++iy) {
        for (int ix = 0; ix < outW; ++ix) {
            float x = -x_max + (ix + 0.5f) * dx;
            float y = (iy + 0.5f) * dy;
            if (y <= 0.0f) continue;

            float r = sqrtf(x*x + y*y);
            if (r < r_min || r > r_max) continue;

            float theta = atan2f(x, y);
            if (fabs(theta) > theta_max) continue;

            float s = sinf(theta);
            float b_f = s / dsin + (Nbeam + 1)/2.0f - 1.0f;
            if (b_f < 0.0f || b_f > (float)(Nbeam - 1)) continue;

            float d_f = (r - rangeoffset) / drange;
            if (d_f < 0.0f || d_f > (float)(UNsample - 1)) continue;

            int b0 = (int)floorf(b_f);
            int b1 = min(b0 + 1, Nbeam - 1);
            float tb = b_f - b0;

            int d0 = (int)floorf(d_f);
            int d1 = min(d0 + 1, UNsample - 1);
            float td = d_f - d0;

            float v00 = bf[b0 * UNsample + d0];
            float v10 = bf[b1 * UNsample + d0];
            float v01 = bf[b0 * UNsample + d1];
            float v11 = bf[b1 * UNsample + d1];

            float v0 = v00 * (1.0f - tb) + v10 * tb;
            float v1 = v01 * (1.0f - tb) + v11 * tb;
            float v  = v0 * (1.0f - td) + v1 * td;

            // 仍然是 dB 空間，範圍大約 [-DR, 0]
            img[iy * outW + ix] = v;
        }
    }

    // --------- dB → 0~255 灰階 ---------
    vector<unsigned char> out(outW * outH);
    for (int i = 0; i < outW * outH; ++i) {
        float v = img[i];
        if (v < -DR) v = -DR;
        if (v > 0.0f) v = 0.0f;
        float x = (v + DR) / DR;            // -DR..0 → 0..1
        out[i] = (unsigned char)(x * 255.0f);
    }

    if (stbi_write_png(pngfile, outW, outH, 1, out.data(), outW))
        cout << "[Scan] Saved PNG: " << pngfile << endl;
    else
        cout << "[Scan] ERROR writing PNG\n";
}
