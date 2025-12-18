// ============================================================
// beamform_pixel_parallel.cu
//
// Optimizations:
// 1. Pixel-Parallel: Each thread computes ONE pixel completely
//    - Eliminates ALL atomic operations (was 128 atomics per pixel!)
//    - Each thread loops over all 128 Tx × 128 Rx internally
// 2. Shared Memory: xchan loaded into shared memory for fast access
// 3. Memory Coalescing: Adjacent threads read adjacent pixels
// 4. Strict Math: Standard sqrtf (No fast_math approximations)
// 5. Loop Tiling: Tx loop tiled into chunks (16 Tx at a time)
//    - Reduces register pressure from 128×128 nested loops
//    - Enables shared memory caching of Tx geometry data
// 6. Multiple Pixels per Thread: Each thread handles 2 pixels
//    - Amortizes Tx geometry calculations across pixels
//    - Better register reuse
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
// Kernel: Pixel-Parallel Beamforming with Loop Tiling
//
// Grid: (pixel_tiles)
// Block: (BLOCK_SIZE)
//
// Each thread computes TWO pixels to amortize Tx geometry calculations.
// Tx loop is tiled into chunks of TX_TILE_SIZE to reduce register pressure.
// No atomics needed - direct write to d_beam[jGlobal].
// ============================================================
#define TX_TILE_SIZE 16
#define PIXELS_PER_THREAD 2

__global__ __launch_bounds__(256)
void beamform_pixel_parallel_kernel(
    const float* __restrict__ rf_transposed, // [Tx][Sample][Rx]
    const float* __restrict__ xchan,
    float* __restrict__ d_beam,
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
    // Load xchan into shared memory for fast repeated access
    __shared__ float s_xchan[128];

    // Shared memory for Tx geometry data within a tile
    __shared__ float s_tx_x[TX_TILE_SIZE];

    int tid = threadIdx.x;
    if (tid < Nchan) {
        s_xchan[tid] = xchan[tid];
    }
    __syncthreads();

    // Each thread handles PIXELS_PER_THREAD pixels
    int pixel_base = (blockIdx.x * blockDim.x + threadIdx.x) * PIXELS_PER_THREAD;

    // Pre-compute geometry for all pixels this thread handles
    float px[PIXELS_PER_THREAD];
    float pz[PIXELS_PER_THREAD];
    bool valid[PIXELS_PER_THREAD];

    #pragma unroll
    for (int p = 0; p < PIXELS_PER_THREAD; ++p) {
        int jGlobal = pixel_base + p;
        valid[p] = (jGlobal < UNsample);

        if (valid[p]) {
            float depth = rangeoffset + jGlobal * drange;
            px[p] = depth * sint;
            pz[p] = depth * cost;
        }
    }

    // Accumulation arrays for each pixel
    float total_sum[PIXELS_PER_THREAD] = {0.0f};

    // --------------------------------------------------------
    // Tiled loop over Tx (process TX_TILE_SIZE Tx at a time)
    // --------------------------------------------------------
    for (int tx_tile = 0; tx_tile < Nchan; tx_tile += TX_TILE_SIZE)
    {
        int tx_count = min(TX_TILE_SIZE, Nchan - tx_tile);

        // Load Tx positions for this tile into shared memory
        if (tid < tx_count) {
            s_tx_x[tid] = s_xchan[tx_tile + tid];
        }
        __syncthreads();

        // Process all Tx in this tile
        for (int tx_local = 0; tx_local < tx_count; ++tx_local)
        {
            int tx = tx_tile + tx_local;
            float x_tx = s_tx_x[tx_local];

            // Base pointer for this Tx: rf_transposed[tx][sample][rx]
            const float* tx_rf_base = rf_transposed + (size_t)tx * Nsample * Nchan;

            // Compute Tx geometry for each pixel
            float d_tx[PIXELS_PER_THREAD];
            #pragma unroll
            for (int p = 0; p < PIXELS_PER_THREAD; ++p) {
                if (valid[p]) {
                    float dx_tx = px[p] - x_tx;
                    d_tx[p] = sqrtf(dx_tx * dx_tx + pz[p] * pz[p]);
                }
            }

            // Loop over all Rx
            for (int rx = 0; rx < Nchan; ++rx)
            {
                float x_rx = s_xchan[rx];

                // Process each pixel
                #pragma unroll
                for (int p = 0; p < PIXELS_PER_THREAD; ++p)
                {
                    if (!valid[p]) continue;

                    // Rx geometry
                    float dx_rx = px[p] - x_rx;
                    float d_rx = sqrtf(dx_rx * dx_rx + pz[p] * pz[p]);

                    // Time-of-flight calculation
                    float t = (d_tx[p] + d_rx) / soundv;
                    float sample_f = (t - timeoffset) * fad * upsamp;

                    int m = (int)(sample_f + 0.5f);

                    // Boundary check
                    if ((unsigned)m < (unsigned)UNsample)
                    {
                        int mm = m / upsamp;
                        int nn = m % upsamp;

                        // 9-tap interpolation filter
                        float val = 0.0f;

                        #pragma unroll
                        for (int k = 0; k < 9; ++k)
                        {
                            int idx = mm - 4 + k;
                            if ((unsigned)idx < (unsigned)Nsample)
                            {
                                int coeff_idx = nn + 64 - 8 * k;
                                float coeff = d_Interp[coeff_idx];

                                // RF access: [tx][idx][rx]
                                float sample_val = __ldg(&tx_rf_base[idx * Nchan + rx]);
                                val += sample_val * coeff;
                            }
                        }

                        total_sum[p] += val;
                    }
                }
            }
        }
        __syncthreads();
    }

    // --------------------------------------------------------
    // Direct write - NO ATOMIC NEEDED!
    // Each pixel is computed by exactly one thread.
    // --------------------------------------------------------
    #pragma unroll
    for (int p = 0; p < PIXELS_PER_THREAD; ++p) {
        if (valid[p]) {
            int jGlobal = pixel_base + p;
            d_beam[jGlobal] = total_sum[p];
        }
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
    // 2. Allocate GPU Memory
    // --------------------------------------------------------
    float *d_rf = nullptr, *d_xchan = nullptr;
    float *d_beam[2] = {nullptr, nullptr};

    size_t bytes_rf    = rf_transposed.size() * sizeof(float);
    size_t bytes_xchan = Nchan * sizeof(float);
    size_t bytes_beam  = UNsample * sizeof(float);

    checkCuda(cudaMalloc(&d_rf, bytes_rf), "cudaMalloc d_rf");
    checkCuda(cudaMalloc(&d_xchan, bytes_xchan), "cudaMalloc d_xchan");
    checkCuda(cudaMalloc(&d_beam[0], bytes_beam), "cudaMalloc beam0");
    checkCuda(cudaMalloc(&d_beam[1], bytes_beam), "cudaMalloc beam1");

    checkCuda(cudaMemcpy(d_rf, rf_transposed.data(), bytes_rf, cudaMemcpyHostToDevice), "Memcpy RF");
    checkCuda(cudaMemcpy(d_xchan, xchan.data(), bytes_xchan, cudaMemcpyHostToDevice), "Memcpy Xchan");

    // Host pinned memory for async transfers
    float* h_beam[2] = {nullptr, nullptr};
    checkCuda(cudaMallocHost(&h_beam[0], bytes_beam), "AllocHost h0");
    checkCuda(cudaMallocHost(&h_beam[1], bytes_beam), "AllocHost h1");

    // Interpolation coefficients
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

    // Geometry parameters
    float fad        = p.fs;
    float soundv     = p.soundv;
    float timeoffset = p.timeoffset;
    float apersize   = p.Nchan * p.pitch;
    float lambda     = soundv / p.fc;
    float dsin       = lambda / apersize / 2.0f;
    float drange     = soundv / fad / 2.0f / upsamp;
    float rangeoffset= timeoffset * soundv / 2.0f;
    int Nbeam        = (int)(std::sqrt(2.0f) / dsin + 0.5f);

    // Streams for double-buffering
    cudaStream_t stream[2];
    cudaEvent_t  done[2];
    for (int i = 0; i < 2; ++i) {
        checkCuda(cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking), "StreamCreate");
        checkCuda(cudaEventCreateWithFlags(&done[i], cudaEventDisableTiming), "EventCreate");
    }

    std::ofstream fout(beamfile, std::ios::binary);

    // --------------------------------------------------------
    // Kernel Configuration: Pixel-Parallel with Tiling
    // --------------------------------------------------------
    const int BLOCK_SIZE = 256;
    // Each thread handles PIXELS_PER_THREAD pixels
    int total_pixel_threads = (UNsample + PIXELS_PER_THREAD - 1) / PIXELS_PER_THREAD;
    int numBlocks = (total_pixel_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cout << "\n===== Pixel-Parallel GPU Beamforming (Tiled) =====\n";
    cout << "UNsample = " << UNsample << ", Pixels/Thread = " << PIXELS_PER_THREAD << "\n";
    cout << "Blocks = " << numBlocks << ", Threads/Block = " << BLOCK_SIZE << "\n";
    cout << "Tx Tile Size = " << TX_TILE_SIZE << "\n";
    cout << "Nbeam = " << Nbeam << "\n";

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
            if (sint > 1.0f) sint = 1.0f;
            if (sint < -1.0f) sint = -1.0f;
            float cost = std::sqrt(1.f - sint * sint);

            // No memset needed! Each pixel is written exactly once.

            // Launch pixel-parallel kernel
            beamform_pixel_parallel_kernel<<<numBlocks, BLOCK_SIZE, 0, stream[cur]>>>(
                d_rf, d_xchan, d_beam[cur],
                Nchan, Nsample, UNsample, upsamp,
                fad, timeoffset, soundv,
                sint, cost, rangeoffset, drange
            );

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

    // Cleanup
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

    cout << "Total beamforming time = " << sec << " sec\n";
}
