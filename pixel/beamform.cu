// ============================================================
// beamform_atomic.cu
//
// Optimizations:
// 1. Memory Coalescing: Transposed RF [Tx][Sample][Rx] (Inherited)
// 2. Kernel Fusion: Replaced d_partial write + Reduce Kernel
//    with direct atomicAdd() to d_beam. This saves massive bandwidth.
// 3. Strict Math: Standard sqrtf/sinf (No fast_math approximations).
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

using std::vector;
using std::cout;
using std::endl;

// Constant memory for interpolation coefficients
__constant__ float d_Interp[72];

static inline void checkCuda(cudaError_t e, const char* msg)
{
    if (e != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << msg << " : "
                  << cudaGetErrorString(e) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// ============================================================
// Kernel: Tx-based + Warp Rx Reduce + ATOMIC Global Accumulate
//
// Grid: (Tx, Pixel_Tiles)
// Block: (32, J_TILE)
//
// Instead of writing to d_partial, we atomicAdd to d_beam directly.
// ============================================================
__global__ __launch_bounds__(256)
void beamform_fused_atomic_kernel(
    const float* __restrict__ rf_transposed, // [Tx][Sample][Rx]
    const float* __restrict__ xchan,
    float* __restrict__ d_beam,              // Direct output
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
    int tx   = blockIdx.x;
    int tile = blockIdx.y;
    
    if (tx >= Nchan) return;

    int lane   = threadIdx.x; // rx lane (0..31)
    int jLocal = threadIdx.y;

    int jGlobal = tile * blockDim.y + jLocal;
    if (jGlobal >= UNsample) return;

    // --------------------------------------------------------
    // Geometry Calculation (High Precision)
    // --------------------------------------------------------
    float depth = rangeoffset + jGlobal * drange;
    float px    = depth * sint;
    float pz    = depth * cost;

    float x_tx  = xchan[tx];
    float dx_tx = px - x_tx;
    float d_tx  = sqrtf(dx_tx * dx_tx + pz * pz); // Standard sqrtf

    float sum_all_rx = 0.0f;
    
    // Base pointer for this Tx in the transposed array
    // [Tx][Sample][Rx]
    const float* tx_rf_base = rf_transposed + (size_t)tx * Nsample * Nchan;

    // --------------------------------------------------------
    // Loop over Rx (Warp Parallel)
    // --------------------------------------------------------
    for (int rxTile = 0; rxTile < Nchan; rxTile += 32)
    {
        int rx = rxTile + lane;
        float contrib = 0.0f;

        if (rx < Nchan)
        {
            float x_rx  = xchan[rx];
            float dx_rx = px - x_rx;
            float d_rx  = sqrtf(dx_rx * dx_rx + pz * pz); // Standard sqrtf

            float t        = (d_tx + d_rx) / soundv;
            float sample_f = (t - timeoffset) * fad * upsamp;

            int m = (int)(sample_f + 0.5f);
            
            // Boundary Check
            if ((unsigned)m < (unsigned)UNsample)
            {
                int mm = m / upsamp;
                int nn = m % upsamp;
                
                // Interpolation
                if ((unsigned)nn < (unsigned)upsamp)
                {
                    float val = 0.0f;
                    
                    // Unroll 9-tap filter
                    #pragma unroll
                    for (int k = 0; k < 9; ++k)
                    {
                        int idx = mm - 4 + k;
                        // Manual bounds check or assume padding? keeping check for safety
                        if ((unsigned)idx < (unsigned)Nsample)
                        {
                            int coeff_idx = nn + 64 - 8 * k;
                            float coeff = d_Interp[coeff_idx];

                            // Coalesced Read via Texture Cache (__ldg)
                            // tx_rf_base is offset by Tx.
                            // idx * Nchan selects the sample row.
                            // + rx selects the column.
                            float sample_val = __ldg(&tx_rf_base[idx * Nchan + rx]);
                            
                            val += sample_val * coeff;
                        }
                    }
                    contrib = val;
                }
            }
        }

        // Warp Reduction (Sum over 32 Rx lanes)
        for (int off = 16; off > 0; off >>= 1)
            contrib += __shfl_down_sync(0xffffffffu, contrib, off);

        if (lane == 0)
            sum_all_rx += contrib;
    }

    // --------------------------------------------------------
    // ATOMIC ACCUMULATION
    // Only Lane 0 writes. Multiple Tx blocks add to the same jGlobal.
    // --------------------------------------------------------
    if (lane == 0)
    {
        atomicAdd(&d_beam[jGlobal], sum_all_rx);
    }
}

// ============================================================
// Host Code
// ============================================================
void run_beamform(
    const vector<vector<vector<float>>>& rf,
    const BFParams& p,
    const char* beamfile)
{
    using clock = std::chrono::high_resolution_clock;
    auto T0 = clock::now();

    const int Nchan    = p.Nchan;
    const int Nsample  = p.Nsample;
    const int upsamp   = 8;
    const int UNsample = upsamp * Nsample;

    // --------------------------------------------------------
    // 1. Transpose RF Data [Tx][Rx][Sample] -> [Tx][Sample][Rx]
    // --------------------------------------------------------
    vector<float> rf_transposed((size_t)Nchan * Nchan * Nsample);
    
    // CPU Transpose
    for (int tx = 0; tx < Nchan; ++tx) {
        for (int k = 0; k < Nsample; ++k) {
            for (int rx = 0; rx < Nchan; ++rx) {
                size_t dst_idx = (size_t)tx * Nsample * Nchan + (size_t)k * Nchan + rx;
                rf_transposed[dst_idx] = rf[tx][rx][k];
            }
        }
    }

    vector<float> xchan(Nchan);
    for (int i = 0; i < Nchan; ++i)
        xchan[i] = (i + 1 - (float)(Nchan + 1) / 2.0f) * p.pitch;

    // --------------------------------------------------------
    // 2. Allocate Memory (NO d_partial needed anymore!)
    // --------------------------------------------------------
    float *d_rf = nullptr, *d_xchan = nullptr;
    float *d_beam[2] = {nullptr, nullptr}; // Only final beam buffer needed

    size_t bytes_rf    = rf_transposed.size() * sizeof(float);
    size_t bytes_xchan = Nchan * sizeof(float);
    size_t bytes_beam  = UNsample * sizeof(float);

    checkCuda(cudaMalloc(&d_rf, bytes_rf), "cudaMalloc d_rf");
    checkCuda(cudaMalloc(&d_xchan, bytes_xchan), "cudaMalloc d_xchan");
    // Only malloc beam buffers
    checkCuda(cudaMalloc(&d_beam[0], bytes_beam), "cudaMalloc beam0");
    checkCuda(cudaMalloc(&d_beam[1], bytes_beam), "cudaMalloc beam1");

    checkCuda(cudaMemcpy(d_rf, rf_transposed.data(), bytes_rf, cudaMemcpyHostToDevice), "Memcpy RF");
    checkCuda(cudaMemcpy(d_xchan, xchan.data(), bytes_xchan, cudaMemcpyHostToDevice), "Memcpy Xchan");

    // Host pinned memory
    float* h_beam[2] = {nullptr, nullptr};
    checkCuda(cudaMallocHost(&h_beam[0], bytes_beam), "AllocHost h0");
    checkCuda(cudaMallocHost(&h_beam[1], bytes_beam), "AllocHost h1");

    // Constants
    float Interp[72] = {
        0,-0.0024f,-0.0046f,-0.0061f,-0.0068f,-0.0065f,-0.0052f,-0.0029f,
        0,0.0136f,0.0258f,0.0349f,0.0395f,0.0384f,0.0312f,0.0181f,
        0,-0.045f,-0.0877f,-0.1222f,-0.1427f,-0.144f,-0.122f,-0.0743f,
        0,0.1370f,0.291f,0.4522f,0.6098f,0.753f,0.8713f,0.956f,
        1.0f,0.956f,0.8713f,0.753f,0.6098f,0.4522f,0.291f,0.137f,
        0,-0.0743f,-0.122f,-0.144f,-0.1427f,-0.1222f,-0.0877f,-0.045f,
        0,0.0181f,0.0312f,0.0384f,0.0395f,0.0349f,0.0258f,0.0136f,
        0,-0.0029f,-0.0052f,-0.0065f,-0.0068f,-0.0061f,-0.0046f,-0.0024f
    };
    checkCuda(cudaMemcpyToSymbol(d_Interp, Interp, 72 * sizeof(float)), "Const Memcpy");

    // Geometry
    float fad        = p.fs;
    float soundv     = p.soundv;
    float timeoffset = p.timeoffset;
    float apersize   = p.Nchan * p.pitch;
    float lambda     = soundv / p.fc;
    float dsin       = lambda / apersize / 2.0f;
    float drange     = soundv / fad / 2.0f / upsamp;
    float rangeoffset= timeoffset * soundv / 2.0f;
    int Nbeam        = (int)(std::sqrt(2.0f) / dsin + 0.5f);

    cudaStream_t stream[2];
    cudaEvent_t  done[2];
    for (int i = 0; i < 2; ++i) {
        checkCuda(cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking), "StreamCreate");
        checkCuda(cudaEventCreateWithFlags(&done[i], cudaEventDisableTiming), "EventCreate");
    }

    std::ofstream fout(beamfile, std::ios::binary);
    
    // Kernel Config
    const int R_TILE = 32;
    const int J_TILE = 8;
    dim3 block(R_TILE, J_TILE);
    dim3 grid_txj(Nchan, (UNsample + J_TILE - 1) / J_TILE);

    // ========================================================
    // Beam Loop
    // ========================================================
    for (int beam = 0; beam < Nbeam + 1; ++beam)
    {
        int cur  = beam & 1;
        int prev = cur ^ 1;

        if (beam < Nbeam)
        {
            float sint = dsin * (beam + 1 - (float)(Nbeam + 1) / 2.0f);
            if(sint > 1.0f) sint = 1.0f; 
            if(sint < -1.0f) sint = -1.0f;
            float cost = std::sqrt(1.f - sint * sint);

            // IMPORTANT: Memset d_beam to 0 before atomic accumulation
            checkCuda(cudaMemsetAsync(d_beam[cur], 0, bytes_beam, stream[cur]), "Memset Beam");

            // Fused Kernel
            beamform_fused_atomic_kernel<<<grid_txj, block, 0, stream[cur]>>>(
                d_rf, d_xchan, d_beam[cur], // Writing directly to d_beam
                Nchan, Nsample, UNsample, upsamp,
                fad, timeoffset, soundv,
                sint, cost, rangeoffset, drange
            );

            // No Reduce Kernel needed!
            
            checkCuda(cudaMemcpyAsync(h_beam[cur], d_beam[cur], bytes_beam,
                                      cudaMemcpyDeviceToHost, stream[cur]), "Memcpy DtoH");

            checkCuda(cudaEventRecord(done[cur], stream[cur]), "EventRecord");
        }

        if (beam > 0)
        {
            checkCuda(cudaEventSynchronize(done[prev]), "Sync");
            fout.write(reinterpret_cast<char*>(h_beam[prev]), bytes_beam);
        }
    }

    fout.close();

    for (int i = 0; i < 2; ++i) {
        cudaEventDestroy(done[i]);
        cudaStreamDestroy(stream[i]);
        cudaFree(d_beam[i]);
        cudaFreeHost(h_beam[i]);
    }
    cudaFree(d_rf);
    cudaFree(d_xchan);

    auto T1 = clock::now();
    double sec = std::chrono::duration<double>(T1 - T0).count();

    cout << "\n===== Atomic Fused GPU Beamforming (No Fast Math) =====\n";
    cout << "Total beamforming time = " << sec << " sec\n";
}
