// ============================================================
//   beamform.cu  (GPU version — each thread handles one pixel)
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <chrono>          // ★ For timing

#include "beamform.h"

using std::vector;
using std::string;
using std::cout;
using std::endl;

// ============================================================
// Device Constants
// ============================================================

__constant__ float d_Interp[72];

// ============================================================
// GPU Kernel: One thread handles one pixel j
// ============================================================

__global__
void beamform_kernel(
    const float* __restrict__ rf,    // [Nchan*Nchan*Nsample]
    float* __restrict__ out,         // [UNsample]
    const float* __restrict__ xchan, // [Nchan]
    int   Nchan,
    int   Nsample,
    int   UNsample,
    int   upsamp,
    float fad,
    float timeoffset,
    float soundv,
    float sint,
    float cost,
    float rangeoffset,
    float drange)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j >= UNsample) return;

    // Compute pixel location
    float depth = rangeoffset + j * drange;
    float px    = depth * sint;
    float pz    = depth * cost;

    float acc = 0.0f;

    // -----------------------------------------------------
    // Loop over all tx,rx pairs
    // -----------------------------------------------------
    for (int tx = 0; tx < Nchan; tx++)
    {
        float x_tx = xchan[tx];

        for (int rx = 0; rx < Nchan; rx++)
        {
            float x_rx = xchan[rx];
            const float* rptr = rf + (tx * Nchan + rx) * Nsample;

            // Compute delay
            float dx_tx = px - x_tx;
            float dx_rx = px - x_rx;

            float d_tx = sqrtf(dx_tx * dx_tx + pz * pz);
            float d_rx = sqrtf(dx_rx * dx_rx + pz * pz);

            float t        = (d_tx + d_rx) / soundv;
            float sample_f = (t - timeoffset) * fad * upsamp;

            // CPU 版 index = round(sample_f)
            int m = (int)(sample_f + 0.5f);
            if (m < 0 || m >= UNsample) continue;

            // CPU: mm = m / upsamp, nn = m % upsamp
            int mm = m / upsamp;
            int nn = m % upsamp;
            if (nn < 0 || nn >= upsamp) continue;

            // interpolation
            float val = 0.0f;
            for (int k = 0; k < 9; ++k)
            {
                int idx_rf = mm - 4 + k;  // buff[mm+k] = rf[mm+k-4]
                if (idx_rf < 0 || idx_rf >= Nsample) continue;

                int coeff_idx = nn + 64 - 8 * k;  // (64,56,...,0)
                val += rptr[idx_rf] * d_Interp[coeff_idx];
            }

            acc += val;
        }
    }

    out[j] = acc;
}

// ============================================================
//  GPU Beamformer (single-beam loop inside)
// ============================================================

void run_beamform(
    
    const vector<vector<vector<float>>>& rf,
    const BFParams& p,
    const char* beamfile)
{
    // ======================================================
    //   ★ Beamforming Time Start
    // ======================================================


    auto T0 = std::chrono::high_resolution_clock::now();
    int Nchan    = p.Nchan;
    int Nsample  = p.Nsample;
    int upsamp   = 8;
    int UNsample = upsamp * Nsample;

    // ------------ Flatten RF to 1D array ------------
    vector<float> rf_flat(Nchan * Nchan * Nsample);
    for (int tx = 0; tx < Nchan; tx++)
        for (int rx = 0; rx < Nchan; rx++)
            for (int k = 0; k < Nsample; k++)
                rf_flat[(tx * Nchan + rx) * Nsample + k] = rf[tx][rx][k];

    // ------------ xchan ------------
    vector<float> xchan(Nchan);
    for (int i = 0; i < Nchan; i++)
        xchan[i] = (i + 1 - (float)(Nchan + 1) / 2.0f) * p.pitch;

    // ------------ Device memory ------------
    float *d_rf    = nullptr;
    float *d_xchan = nullptr;
    float *d_out   = nullptr;

    cudaMalloc(&d_rf,    rf_flat.size() * sizeof(float));
    cudaMalloc(&d_xchan, Nchan * sizeof(float));
    cudaMalloc(&d_out,   UNsample * sizeof(float));

    cudaMemcpy(d_rf,    rf_flat.data(), rf_flat.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_xchan, xchan.data(),   Nchan * sizeof(float),          cudaMemcpyHostToDevice);

    // ------------ Upload interpolation kernel ------------
    float Interp[72] = {
        0,-0.0024f,-0.0046f,-0.0061f,-0.0068f,-0.0065f,-0.0052f,-0.0029f,
        0,0.0136f,0.0258f,0.0349f,0.0395f,0.0384f,0.0312f,0.0181f,
        0,-0.045f,-0.0877f,-0.1222f,-0.1427f,-0.144f,-0.122f,-0.0743f,
        0,0.1370f,0.291f,0.4522f,0.6098f,0.753f,0.8713f,0.956f,
        1.0f,0.956f,0.8713f,0.753f,0.6098f,0.4522f,0.291f,0.137f,
        0,-0.0743f,-0.122f,-0.144f,-0.1427f,-0.1222f,-0.0877f,-0.045f,
        0,0.0181f,0.0312f,0.0384f,0.0395f,0.0349f,0.0258f,0.0136f,
        0,-0.0029f,-0.0052f,-0.0065f,-0.0068f,-0.0061f,-0.0046f,-0.0024f,
        0,0,0,0,0,0,0,0
    };
    cudaMemcpyToSymbol(d_Interp, Interp, 72 * sizeof(float));

    // ------------ Basic parameters ------------
    float fad        = p.fs;
    float soundv     = p.soundv;
    float timeoffset = p.timeoffset;

    float apersize = p.Nchan * p.pitch;
    float lambda   = soundv / p.fc;
    float dsin     = lambda / apersize / 2.0f;

    float drange      = soundv / fad / 2.0f / upsamp;
    float rangeoffset = timeoffset * soundv / 2.0f;

    int Nbeam = (int)(sqrt(2.0f) / dsin + 0.5f);

    std::ofstream fout(beamfile, std::ios::binary);

    dim3 block(256);
    dim3 grid((UNsample + block.x - 1) / block.x);



    for (int beam = 0; beam < Nbeam; beam++)
    {
        float sint = dsin * (beam + 1 - (float)(Nbeam + 1) / 2.0f);
        if (sint > 1.f)  sint = 1.f;
        if (sint < -1.f) sint = -1.f;
        float cost = sqrtf(1.f - sint * sint);

        beamform_kernel<<<grid, block>>>(
            d_rf, d_out, d_xchan,
            Nchan,
            Nsample,
            UNsample,
            upsamp,
            fad,
            timeoffset,
            soundv,
            sint,
            cost,
            rangeoffset,
            drange
        );
        cudaDeviceSynchronize();

        vector<float> beamsum(UNsample);
        cudaMemcpy(beamsum.data(), d_out, UNsample * sizeof(float), cudaMemcpyDeviceToHost);

        fout.write((char*)beamsum.data(), UNsample * sizeof(float));
    }

    // ======================================================
    //   ★ Beamforming Time End
    // ======================================================
    auto T1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(T1 - T0).count();

    cout << "\n===== GPU Beamforming Finished =====\n";
    cout << "Total beamforming time = " << sec << " sec\n";

    fout.close();

    cudaFree(d_rf);
    cudaFree(d_xchan);
    cudaFree(d_out);
}
