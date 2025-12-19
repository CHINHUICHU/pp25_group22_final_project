// ============================================================
// beamform_baseline.cu
// ------------------------------------------------------------
// BASELINE GPU VERSION - No optimization tricks
// This version uses straightforward parallelization without:
// - Shared memory optimization
// - Warp-level primitives
// - Constant memory
// - __ldg intrinsic
// - Tiling strategies
// - Register blocking
// - Loop unrolling
// - Dual-stream pipelining
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>

#include <cuda_runtime.h>
#include "beamform.h"

using namespace std;

#define UPSAMP   8
#define PAD      4

static inline void checkCuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        cerr << "[CUDA ERROR] " << msg << " : " << cudaGetErrorString(e) << endl;
        exit(EXIT_FAILURE);
    }
}

static inline void checkLastKernel(const char* msg) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        cerr << "[KERNEL ERROR] " << msg << " : " << cudaGetErrorString(e) << endl;
        exit(EXIT_FAILURE);
    }
}

// ------------------------------------------------------------
// BASELINE Kernel: Simple 1D parallelization over output samples
// Each thread computes ONE output sample across all TX/RX pairs
// ------------------------------------------------------------
__global__ void beamform_baseline_kernel(
    const float* __restrict__ rf_padded,   // [Tx][Rx][SamplePad]
    const float* __restrict__ xchan,       // [Nchan]
    const float* __restrict__ interp,      // [72]
    float* __restrict__ d_beam,            // [UNsample]
    int   Nchan,
    int   UNsample,
    int   SamplePad,
    float sf_scale,
    float sf_bias,
    float sint,
    float cost,
    float rangeoffset,
    float drange)
{
    // Simple 1D thread indexing - one thread per output sample
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= UNsample) return;

    // Compute pixel position
    float depth = rangeoffset + (float)j * drange;
    float px = depth * sint;
    float pz = depth * cost;

    float sum = 0.0f;

    // Loop over all TX-RX pairs
    for (int tx = 0; tx < Nchan; ++tx) {
        // Compute TX distance
        float dx_tx = px - xchan[tx];
        float dtx = sqrtf(dx_tx * dx_tx + pz * pz);

        for (int rx = 0; rx < Nchan; ++rx) {
            // Compute RX distance
            float dx_rx = px - xchan[rx];
            float drx = sqrtf(dx_rx * dx_rx + pz * pz);

            // Compute sample index with upsampling
            float sf = (dtx + drx) * sf_scale - sf_bias;
            int m = (int)(sf + 0.5f);
            if (m < 0 || m >= UNsample) continue;

            // Get integer and fractional parts
            int mm = m / 8;  // Integer division (no bit shift)
            int nn = m % 8;  // Modulo (no bit mask)

            // Access RF data: [tx][rx][sample]
            size_t stride_tx = (size_t)Nchan * (size_t)SamplePad;
            size_t stride_rx = (size_t)SamplePad;
            size_t base = (size_t)tx * stride_tx + (size_t)rx * stride_rx + (size_t)mm;

            // Simple interpolation (no loop unrolling)
            float val = 0.0f;
            for (int t = 0; t < 9; ++t) {
                float c = interp[nn + 64 - 8 * t];
                val += rf_padded[base + t] * c;  // Regular load (no __ldg)
            }
            sum += val;
        }
    }

    d_beam[j] = sum;  // Direct write (no atomic needed since one thread per output)
}

// ------------------------------------------------------------
// Host Code
// ------------------------------------------------------------
void run_beamform(const vector<vector<vector<float>>>& rf,
                  const BFParams& p,
                  const char* beamfile)
{
    auto T0 = chrono::high_resolution_clock::now();

    const int Nchan   = p.Nchan;
    const int Nsample = p.Nsample;
    const int UNsample = UPSAMP * Nsample;
    const int SamplePad = Nsample + 2 * PAD;

    // Prepare padded RF data [tx][rx][k+PAD]
    vector<float> rf_padded((size_t)Nchan * Nchan * SamplePad, 0.0f);
    for (int tx = 0; tx < Nchan; ++tx) {
        for (int rxch = 0; rxch < Nchan; ++rxch) {
            for (int k = 0; k < Nsample; ++k) {
                size_t dst = (size_t)tx * (size_t)Nchan * (size_t)SamplePad
                           + (size_t)rxch * (size_t)SamplePad
                           + (size_t)(k + PAD);
                rf_padded[dst] = rf[tx][rxch][k];
            }
        }
    }

    // Channel positions
    vector<float> xchan(Nchan);
    for (int i = 0; i < Nchan; ++i) {
        xchan[i] = (i + 1 - (Nchan + 1) / 2.0f) * p.pitch;
    }

    // Interpolation coefficients (in global memory, not constant)
    float Interp[72] = {
        0,-0.0024f,-0.0046f,-0.0061f,-0.0068f,-0.0065f,-0.0052f,-0.0029f, 0,0.0136f,0.0258f,0.0349f,0.0395f,0.0384f,0.0312f,0.0181f,
        0,-0.045f,-0.0877f,-0.1222f,-0.1427f,-0.144f,-0.122f,-0.0743f, 0,0.1370f,0.291f,0.4522f,0.6098f,0.753f,0.8713f,0.956f,
        1.0f,0.956f,0.8713f,0.753f,0.6098f,0.4522f,0.291f,0.137f, 0,-0.0743f,-0.122f,-0.144f,-0.1427f,-0.1222f,-0.0877f,-0.045f,
        0,0.0181f,0.0312f,0.0384f,0.0395f,0.0349f,0.0258f,0.0136f, 0,-0.0029f,-0.0052f,-0.0065f,-0.0068f,-0.0061f,-0.0046f,-0.0024f,
        0,0,0,0,0,0,0,0
    };

    // Allocate device memory
    float *d_rf = nullptr, *d_xchan = nullptr, *d_interp = nullptr, *d_beam = nullptr;

    size_t bytes_rf   = rf_padded.size() * sizeof(float);
    size_t bytes_beam = (size_t)UNsample * sizeof(float);

    checkCuda(cudaMalloc(&d_rf, bytes_rf), "Malloc RF");
    checkCuda(cudaMalloc(&d_xchan, (size_t)Nchan * sizeof(float)), "Malloc Xchan");
    checkCuda(cudaMalloc(&d_interp, 72 * sizeof(float)), "Malloc Interp");
    checkCuda(cudaMalloc(&d_beam, bytes_beam), "Malloc Beam");

    // Copy data to device (single stream, no pipelining)
    checkCuda(cudaMemcpy(d_rf, rf_padded.data(), bytes_rf, cudaMemcpyHostToDevice), "H2D RF");
    checkCuda(cudaMemcpy(d_xchan, xchan.data(), (size_t)Nchan * sizeof(float), cudaMemcpyHostToDevice), "H2D Xchan");
    checkCuda(cudaMemcpy(d_interp, Interp, 72 * sizeof(float), cudaMemcpyHostToDevice), "H2D Interp");

    // Beam parameters
    float apersize = Nchan * p.pitch;
    float lambda   = p.soundv / p.fc;
    float dsin     = lambda / apersize / 2.0f;
    float drange   = p.soundv / p.fs / 2.0f / (float)UPSAMP;
    float rangeoffset = p.timeoffset * p.soundv / 2.0f;
    int   Nbeam    = (int)(sqrtf(2.0f) / dsin + 0.5f);

    float sf_scale = (p.fs * UPSAMP) / p.soundv;
    float sf_bias  = p.timeoffset * p.fs * UPSAMP;

    // Simple 1D launch configuration (no tiling)
    int threadsPerBlock = 256;  // Simple choice
    int blocksPerGrid = (UNsample + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate host output (no pinned memory)
    vector<float> h_beam(UNsample);

    // Open output file
    ofstream fout(beamfile, ios::binary);
    if (!fout) {
        cerr << "Failed to open output file: " << beamfile << endl;
        exit(EXIT_FAILURE);
    }

    // Process each beam (no pipelining)
    for (int b = 0; b < Nbeam; ++b) {
        float sint = dsin * (b + 1 - (Nbeam + 1) / 2.0f);
        sint = fmaxf(-1.0f, fminf(1.0f, sint));
        float cost = sqrtf(1.0f - sint * sint);

        // Clear output buffer
        checkCuda(cudaMemset(d_beam, 0, bytes_beam), "Memset beam");

        // Launch kernel
        beamform_baseline_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_rf, d_xchan, d_interp, d_beam,
            Nchan, UNsample,
            SamplePad,
            sf_scale, sf_bias,
            sint, cost, rangeoffset, drange);

        checkLastKernel("beamform_baseline_kernel launch");

        // Wait for kernel to complete
        checkCuda(cudaDeviceSynchronize(), "DeviceSync");

        // Copy result back
        checkCuda(cudaMemcpy(h_beam.data(), d_beam, bytes_beam, cudaMemcpyDeviceToHost), "D2H beam");

        // Write to file
        fout.write((char*)h_beam.data(), (std::streamsize)bytes_beam);
    }

    fout.close();

    // Cleanup
    cudaFree(d_rf);
    cudaFree(d_xchan);
    cudaFree(d_interp);
    cudaFree(d_beam);

    auto T1 = chrono::high_resolution_clock::now();
    cout << "Total beamforming time (BASELINE GPU): "
         << chrono::duration<double>(T1 - T0).count()
         << " s" << endl;
}
