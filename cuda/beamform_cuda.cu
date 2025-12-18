// beamform_cuda.cu
// CUDA implementation of beamforming with pre-interpolation (v7)
// Only the currently used kernels remain: preinterpolate_kernel and beamform_kernel_v7

#include "beamform_cuda.cuh"
#include "../sequential/params.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>

#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                 << cudaGetErrorString(err) << endl; \
            exit(1); \
        } \
    } while(0)

// Constant memory for interpolation kernel (288 bytes, heavily reused)
// Constant memory is optimal for broadcast access (all threads read same value)
__constant__ float c_Interp[72];

// =============================================================================
// Pre-interpolation Kernel
// =============================================================================
// Separates interpolation from beamforming for massive performance gain.
// Instead of each beamform thread computing 9-tap interpolation per RF access,
// we pre-compute ALL interpolated values once.
//
// Memory tradeoff:
//   Input:  rf[Nchan*Nchan][Nsample_padded] = 128*128*2056 * 4 = ~128 MB
//   Output: rf_interp[Nchan*Nchan][UNsample] = 128*128*16384 * 4 = ~1 GB
//
// But this eliminates 9 FMA operations per RF access in beamforming!
// Expected speedup: 3-5x in beamform kernel
// =============================================================================
__global__ void preinterpolate_kernel(
    const float* __restrict__ rf_padded,   // [Nchan*Nchan][Nsample_padded] (padded with ±4 zeros)
    float* __restrict__ rf_interp,          // [Nchan*Nchan][UNsample] output
    int Nchan,
    int Nsample,           // Original sample count
    int Nsample_padded,    // Padded sample count (Nsample + 8)
    int UNsample,          // Upsampled count (Nsample * upsamp)
    int upsamp)
{
    // 2D grid: blockIdx.x = tx_rx pair index, blockIdx.y = sample block
    int tx_rx = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;  // Upsampled index

    if (tx_rx >= Nchan * Nchan || j >= UNsample) return;

    int mm = j / upsamp;  // Original sample index
    int nn = j % upsamp;  // Sub-sample offset for interpolation

    // RF data pointer for this tx-rx pair (data starts at offset 4 due to padding)
    const float* rf_ptr = rf_padded + tx_rx * Nsample_padded + 4;

    // Compute 9-tap interpolation
    // No bounds checking needed - data is padded with zeros!
    float val = rf_ptr[mm - 4] * c_Interp[nn + 64] +
                rf_ptr[mm - 3] * c_Interp[nn + 56] +
                rf_ptr[mm - 2] * c_Interp[nn + 48] +
                rf_ptr[mm - 1] * c_Interp[nn + 40] +
                rf_ptr[mm]     * c_Interp[nn + 32] +
                rf_ptr[mm + 1] * c_Interp[nn + 24] +
                rf_ptr[mm + 2] * c_Interp[nn + 16] +
                rf_ptr[mm + 3] * c_Interp[nn + 8] +
                rf_ptr[mm + 4] * c_Interp[nn];

    // Store interpolated value
    rf_interp[tx_rx * UNsample + j] = val;
}

// =============================================================================
// Optimized Kernel v7 - Sample-Parallel with Pre-interpolated RF Data
// =============================================================================
// Key optimization: RF data is pre-interpolated, so beamforming is just
// delay computation + table lookup + summation.
//
// Eliminates: 9 FMA operations per RF access (was the bottleneck!)
//
// Grid: (Nbeam, numSampleBlocks)
// Each thread: processes ALL tx-rx pairs for ONE output sample
// No atomics needed - each thread owns its output sample
// =============================================================================
__global__ void beamform_kernel_v7(
    const float* __restrict__ rf_interp,   // [Nchan*Nchan][UNsample] pre-interpolated
    float* __restrict__ beamsum_out,       // [Nbeam][UNsample] output
    const float* __restrict__ xchan,       // [Nchan] channel x positions
    int Nchan,
    int UNsample,
    int Nbeam,
    float dsin,
    float drange,
    float rangeoffset,
    float soundv,
    float timeoffset,
    float fad,
    int upsamp)
{
    // 2D grid: blockIdx.x = beam, blockIdx.y = sample block index
    int beam = blockIdx.x;
    int sampleBlockIdx = blockIdx.y;

    if (beam >= Nbeam) return;

    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    // Calculate which output sample this thread handles
    int samplesPerBlock = blockSize;
    int sampleStart = sampleBlockIdx * samplesPerBlock;
    int j = sampleStart + tid;  // Output sample index for this thread

    if (j >= UNsample) return;

    // Shared memory for xchan (small, heavily reused)
    extern __shared__ float s_xchan[];

    // Cooperatively load xchan to shared memory
    for (int i = tid; i < Nchan; i += blockSize) {
        s_xchan[i] = xchan[i];
    }
    __syncthreads();

    // Compute beam geometry
    float sint = dsin * (beam + 1 - (float)(Nbeam + 1) / 2.0f);
    sint = fmaxf(-1.0f, fminf(1.0f, sint));
    float cost = sqrtf(1.0f - sint * sint);

    // Pre-compute point position for this thread's output sample
    float depth = rangeoffset + j * drange;
    float px = depth * sint;
    float pz = depth * cost;
    float pz_sq = pz * pz;

    // Accumulator - NO atomics needed!
    float sum = 0.0f;

    // Pre-compute constants
    float inv_soundv = 1.0f / soundv;
    float time_scale = fad * upsamp;

    // Loop over ALL tx-rx pairs
    for (int tx = 0; tx < Nchan; tx++) {
        float x_tx = s_xchan[tx];
        float dx_tx = px - x_tx;
        float dx_tx_sq = dx_tx * dx_tx;

        // Compute d_tx once per tx (hoisted from rx loop)
        float d_tx_sq = dx_tx_sq + pz_sq;
        float d_tx = sqrtf(d_tx_sq);

        // Base pointer for this tx row
        int tx_base = tx * Nchan;

        for (int rx = 0; rx < Nchan; rx++) {
            float x_rx = s_xchan[rx];
            float dx_rx = px - x_rx;

            // Compute d_rx
            float d_rx_sq = dx_rx * dx_rx + pz_sq;
            float d_rx = sqrtf(d_rx_sq);

            // Compute time and sample index
            float t = (d_tx + d_rx) * inv_soundv;
            float sample_f = (t - timeoffset) * time_scale;
            int idx = (int)(sample_f + 0.5f);

            if (idx >= 0 && idx < UNsample) {
                // Simple table lookup - NO interpolation computation!
                float val = rf_interp[(tx_base + rx) * UNsample + idx];
                sum += val;
            }
        }
    }

    // Direct write - no atomics needed!
    beamsum_out[beam * UNsample + j] = sum;
}

// =============================================================================
// Host function
// =============================================================================
void run_beamform_cuda(
    const vector<vector<vector<float>>>& rf,
    const BFParams& p,
    const char* beamfile)
{
    // Parameters
    const int   Nchan      = p.Nchan;
    const float fad        = p.fs;
    const float f0         = p.fc;
    const float timeoffset = p.timeoffset;
    const int   Nsample    = p.Nsample;
    const float pitch      = p.pitch;
    const float soundv     = p.soundv;
    const int   upsamp     = 8;

    const float apersize = Nchan * pitch;
    const float lambda   = soundv / f0;
    const float dsin     = lambda / apersize / 2.0f;

    const int   Nbeam    = static_cast<int>(sqrt(2.0f) / dsin + 0.5f);
    const int   UNsample = upsamp * Nsample;

    const float drange      = soundv / fad / 2.0f / upsamp;
    const float rangeoffset = timeoffset * soundv / 2.0f;

    cout << "===== CUDA Beamforming Parameters =====\n";
    cout << "Nchan      = " << Nchan      << "\n";
    cout << "Nbeam      = " << Nbeam      << "\n";
    cout << "Nsample    = " << Nsample    << "\n";
    cout << "UNsample   = " << UNsample   << "\n";
    cout << "fs (MHz)   = " << fad        << "\n";
    cout << "fc (MHz)   = " << f0         << "\n";
    cout << "timeoffset = " << timeoffset << " us\n";
    cout << "pitch      = " << pitch      << " mm\n";
    cout << "soundv     = " << soundv     << " mm/us\n";
    cout << "========================================\n";

    // Interpolation kernel
    const float h_Interp[72] = {
        0,-0.0024f,-0.0046f,-0.0061f,-0.0068f,-0.0065f,-0.0052f,-0.0029f,0,0.0136f,0.0258f,0.0349f,
        0.0395f,0.0384f,0.0312f,0.0181f,0,-0.045f,-0.0877f,-0.1222f,-0.1427f,-0.144f,-0.122f,-0.0743f,0,
        0.1370f,0.291f,0.4522f,0.6098f,0.753f,0.8713f,0.956f,1.0f,0.956f,0.8713f,0.753f,0.6098f,
        0.4522f,0.291f,0.137f,0,-0.0743f,-0.122f,-0.144f,-0.1427f,-0.1222f,-0.0877f,-0.045f,0,
        0.0181f,0.0312f,0.0384f,0.0395f,0.0349f,0.0258f,0.0136f,0,-0.0029f,-0.0052f,-0.0065f,
        -0.0068f,-0.0061f,-0.0046f,-0.0024f,0,0,0,0,0,0,0,0
    };

    // Copy interpolation kernel to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(c_Interp, h_Interp, sizeof(h_Interp)));

    // Prepare xchan array
    vector<float> h_xchan(Nchan);
    for (int i = 0; i < Nchan; ++i)
        h_xchan[i] = (i + 1 - (float)(Nchan + 1) / 2.0f) * pitch;

    // Flatten RF data with padding: rf[tx][rx][sample_padded]
    // Padding: add 4 zeros before and 4 zeros after each RF trace
    // This eliminates bounds checking in the kernel!
    const int PAD_SIZE = 4;
    const int Nsample_padded = Nsample + 2 * PAD_SIZE;
    size_t rf_size_padded = (size_t)Nchan * Nchan * Nsample_padded;
    vector<float> h_rf_padded(rf_size_padded, 0.0f);  // Initialize with zeros

    auto flatten_start = high_resolution_clock::now();
    for (int tx = 0; tx < Nchan; tx++) {
        for (int rx = 0; rx < Nchan; rx++) {
            size_t base_idx = (tx * Nchan + rx) * Nsample_padded;

            // First 4 elements are zero (already initialized)
            // Copy actual data starting at offset PAD_SIZE
            for (int s = 0; s < Nsample; s++) {
                h_rf_padded[base_idx + PAD_SIZE + s] = rf[tx][rx][s];
            }
            // Last 4 elements are zero (already initialized)
        }
    }
    auto flatten_end = high_resolution_clock::now();
    cout << "RF data flatten+padding time: "
         << duration<double, milli>(flatten_end - flatten_start).count() << " ms\n";
    cout << "Padding: " << PAD_SIZE << " zeros before and after each trace\n";
    cout << "Padded size: " << Nsample << " -> " << Nsample_padded << " samples\n";

    // Allocate device memory
    float* d_rf;
    float* d_rf_interp;  // Pre-interpolated RF data
    float* d_xchan;
    float* d_beamsum;

    size_t beamsum_size = (size_t)Nbeam * UNsample;
    size_t rf_interp_size = (size_t)Nchan * Nchan * UNsample;  // Pre-interpolated size

    CUDA_CHECK(cudaMalloc(&d_rf, rf_size_padded * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rf_interp, rf_interp_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_xchan, Nchan * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beamsum, beamsum_size * sizeof(float)));

    cout << "Memory allocation:\n";
    cout << "  d_rf (padded):    " << rf_size_padded * sizeof(float) / (1024.0 * 1024.0) << " MB\n";
    cout << "  d_rf_interp:      " << rf_interp_size * sizeof(float) / (1024.0 * 1024.0) << " MB\n";
    cout << "  d_beamsum:        " << beamsum_size * sizeof(float) / (1024.0 * 1024.0) << " MB\n";

    // Initialize beamsum to zero
    CUDA_CHECK(cudaMemset(d_beamsum, 0, beamsum_size * sizeof(float)));

    // Copy data to device
    auto copy_start = high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(d_rf, h_rf_padded.data(), rf_size_padded * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xchan, h_xchan.data(), Nchan * sizeof(float), cudaMemcpyHostToDevice));
    auto copy_end = high_resolution_clock::now();
    cout << "Host to Device copy time: "
         << duration<double, milli>(copy_end - copy_start).count() << " ms\n";

    // =========================================================================
    // PHASE 1: Pre-interpolation Kernel
    // =========================================================================
    // Compute interpolated RF values once, eliminating 9 FMA ops per beamform access
    cout << "\n--- Phase 1: Pre-interpolation ---\n";

    int interp_threads = 256;
    int numTxRxPairs = Nchan * Nchan;  // 16384 pairs
    int numSampleBlocksInterp = (UNsample + interp_threads - 1) / interp_threads;

    dim3 interpGrid(numTxRxPairs, numSampleBlocksInterp);
    dim3 interpBlock(interp_threads);

    cout << "Pre-interpolation kernel: " << numTxRxPairs << " x " << numSampleBlocksInterp
         << " = " << numTxRxPairs * numSampleBlocksInterp << " blocks, "
         << interp_threads << " threads/block\n";

    auto interp_start = high_resolution_clock::now();

    preinterpolate_kernel<<<interpGrid, interpBlock>>>(
        d_rf, d_rf_interp,
        Nchan, Nsample, Nsample_padded, UNsample, upsamp);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto interp_end = high_resolution_clock::now();
    cout << "Pre-interpolation time: "
         << duration<double, milli>(interp_end - interp_start).count() << " ms\n";

    // Free original RF data - no longer needed
    CUDA_CHECK(cudaFree(d_rf));
    d_rf = nullptr;

    // =========================================================================
    // PHASE 2: Beamforming Kernel v7 (with pre-interpolated data)
    // =========================================================================
    cout << "\n--- Phase 2: Beamforming (v7) ---\n";

    int threadsPerBlock = 256;
    int numSampleBlocks = (UNsample + threadsPerBlock - 1) / threadsPerBlock;

    dim3 gridDim(Nbeam, numSampleBlocks);
    dim3 blockDim(threadsPerBlock);

    size_t sharedMemSize = Nchan * sizeof(float);

    int totalBlocks = Nbeam * numSampleBlocks;
    cout << "Beamform kernel v7: " << Nbeam << " x " << numSampleBlocks
         << " = " << totalBlocks << " blocks, "
         << threadsPerBlock << " threads/block\n";
    cout << "Shared memory per block: " << sharedMemSize << " bytes\n";
    cout << "Optimization: pre-interpolated RF (simple table lookup)\n";

    auto kernel_start = high_resolution_clock::now();

    beamform_kernel_v7<<<gridDim, blockDim, sharedMemSize>>>(
        d_rf_interp, d_beamsum, d_xchan,
        Nchan, UNsample, Nbeam,
        dsin, drange, rangeoffset, soundv, timeoffset, fad, upsamp);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto kernel_end = high_resolution_clock::now();
    cout << "Beamform kernel time: "
         << duration<double, milli>(kernel_end - kernel_start).count() << " ms\n";

    // Total kernel time
    double total_kernel_ms = duration<double, milli>(interp_end - interp_start).count() +
                             duration<double, milli>(kernel_end - kernel_start).count();
    cout << "\nTotal kernel execution time (interp + beamform): " << total_kernel_ms << " ms\n";

    // Copy results back
    vector<float> h_beamsum(beamsum_size);
    auto copyback_start = high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(h_beamsum.data(), d_beamsum, beamsum_size * sizeof(float),
                          cudaMemcpyDeviceToHost));
    auto copyback_end = high_resolution_clock::now();
    cout << "Device to Host copy time: "
         << duration<double, milli>(copyback_end - copyback_start).count() << " ms\n";

    // Write output file
    ofstream fout(beamfile, ios::binary | ios::trunc);
    if (!fout) {
        cerr << "[CUDA Beamform] Cannot open output beam file: " << beamfile << endl;
        exit(1);
    }

    // Write beam by beam (same format as sequential version)
    for (int beam = 0; beam < Nbeam; beam++) {
        fout.write((char*)(h_beamsum.data() + beam * UNsample), UNsample * sizeof(float));
    }
    fout.close();

    // Cleanup
    CUDA_CHECK(cudaFree(d_rf));
    CUDA_CHECK(cudaFree(d_xchan));
    CUDA_CHECK(cudaFree(d_beamsum));

    auto total_end = high_resolution_clock::now();
    cout << "================ CUDA Beamforming Finished ================\n";
    cout << "Total CUDA beamform time: "
         << duration<double>(total_end - flatten_start).count() << " sec\n";
}
