// ============================================================
// beamform_v100_corrected.cu
// ------------------------------------------------------------
// 加速版 (保持數學結果不變的前提下)
//
// 主要改動：
// 1) [關鍵] 把 warp reduction 從「每個 rxTile 都做一次」改成「整個 rxTile loop 結束後只做一次」
//    -> Nchan=128 時，reduce 次數從 4 次降到 1 次（每像素、每txTile）
// 2) [關鍵] 預先計算 tx_base0 + i*strideTx，避免內圈每次做 64-bit tx*strideTx
// 3) 9-tap 係數用 index 取值 + fmaf，減少 pointer churn，利於編譯器排程
// 4) Host 端：建議 kernel 偏好 L1（shared 很小，Volta 常有感）
//
// 注意：你原本的資料 layout (rf_padded: [Tx][SamplePad][Rx]) 不變。
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
#define MAX_CHAN 128
#define TX_TILE  8
#define J_TILE   8

__constant__ float d_Interp[72];

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
// Kernel: Corrected & Faster (single reduction per pixel)
// ------------------------------------------------------------
__global__ __launch_bounds__(32 * J_TILE, 2)
void beamform_v100_corrected_kernel(
    const float* __restrict__ rf_padded,   // [Tx][SamplePad][Rx]
    const float* __restrict__ xchan,       // [Nchan]
    float* __restrict__ d_beam,            // [UNsample]
    int   Nchan,
    int   UNsample,
    int   SamplePad_x_Nchan,               // stride = SamplePad * Nchan
    float sf_scale,
    float sf_bias,
    float sint,
    float cost,
    float rangeoffset,
    float drange)
{
    __shared__ float s_xchan[MAX_CHAN];
    __shared__ float s_interp[72];

    const int lane   = threadIdx.x;  // 0..31
    const int jLocal = threadIdx.y;  // 0..J_TILE-1
    const int tid    = jLocal * 32 + lane;

    if (tid < Nchan) s_xchan[tid] = xchan[tid];
    if (tid < 72)    s_interp[tid] = d_Interp[tid];
    __syncthreads();

    const int jGlobal = (int)blockIdx.y * J_TILE + jLocal;
    if (jGlobal >= UNsample) return;

    // ---- 1) geometry (lane0 compute + warp broadcast) ----
    float px = 0.f, pz = 0.f;
    if (lane == 0) {
        float depth = rangeoffset + (float)jGlobal * drange;
        px = depth * sint;
        pz = depth * cost;
    }
    px = __shfl_sync(0xffffffff, px, 0);
    pz = __shfl_sync(0xffffffff, pz, 0);

    // ---- 2) TX distance (lane0 compute + broadcast) ----
    float dtx[TX_TILE];
    const int tx_start = (int)blockIdx.x * TX_TILE;

    if (lane == 0) {
        #pragma unroll
        for (int i = 0; i < TX_TILE; ++i) {
            int tx = tx_start + i;
            if (tx < Nchan) {
                float dx = px - s_xchan[tx];
                dtx[i] = sqrtf(dx * dx + pz * pz);
            } else {
                dtx[i] = 0.0f;
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < TX_TILE; ++i) {
        dtx[i] = __shfl_sync(0xffffffff, dtx[i], 0);
    }

    const size_t strideTx = (size_t)SamplePad_x_Nchan;
    const size_t tx_base0 = (size_t)tx_start * strideTx; // ★預先算 base

    // ★每個 lane 累加所有 rxTile 的結果，最後只做一次 reduction
    float lane_sum = 0.0f;

    for (int rxTile = 0; rxTile < Nchan; rxTile += 32) {
        int rx = rxTile + lane;
        if (rx >= Nchan) continue;

        float dx_rx = px - s_xchan[rx];
        float drx   = sqrtf(dx_rx * dx_rx + pz * pz);

        #pragma unroll
        for (int i = 0; i < TX_TILE; ++i) {
            int tx = tx_start + i;
            if (tx >= Nchan) break;

            float sf = (dtx[i] + drx) * sf_scale - sf_bias;
            int m = (int)(sf + 0.5f);
            if ((unsigned)m >= (unsigned)UNsample) continue;

            int mm = m >> 3;
            int nn = m & 7;

            // ptr 指向 [Tx][mm][rx]
            const float* ptr =
                rf_padded
                + (tx_base0 + (size_t)i * strideTx)
                + (size_t)mm * (size_t)Nchan
                + (size_t)rx;

            // 係數索引：tap t 的係數在 s_interp[nn + 64 - 8*t]
            float val = 0.0f;
            #pragma unroll
            for (int t = 0; t < 9; ++t) {
                float c = s_interp[nn + 64 - 8 * t];
                val = fmaf(__ldg(ptr), c, val);
                ptr += Nchan;
            }
            lane_sum += val;
        }
    }

    // ---- single warp reduction ----
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        lane_sum += __shfl_down_sync(0xffffffff, lane_sum, off);
    }

    if (lane == 0) atomicAdd(&d_beam[jGlobal], lane_sum);
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

    // ---- Transpose & Padding: rf_padded[tx][k+PAD][rx] ----
    vector<float> rf_padded((size_t)Nchan * SamplePad * Nchan, 0.0f);
    for (int tx = 0; tx < Nchan; ++tx) {
        for (int k = 0; k < Nsample; ++k) {
            for (int rxch = 0; rxch < Nchan; ++rxch) {
                size_t dst = (size_t)tx * (size_t)SamplePad * (size_t)Nchan
                           + (size_t)(k + PAD) * (size_t)Nchan
                           + (size_t)rxch;
                rf_padded[dst] = rf[tx][rxch][k];
            }
        }
    }

    vector<float> xchan(Nchan);
    for (int i = 0; i < Nchan; ++i) {
        xchan[i] = (i + 1 - (Nchan + 1) / 2.0f) * p.pitch;
    }

    float *d_rf = nullptr, *d_xchan = nullptr;
    float *d_beam[2] = {nullptr, nullptr};

    size_t bytes_rf   = rf_padded.size() * sizeof(float);
    size_t bytes_beam = (size_t)UNsample * sizeof(float);

    checkCuda(cudaMalloc(&d_rf, bytes_rf), "Malloc RF");
    checkCuda(cudaMalloc(&d_xchan, (size_t)Nchan * sizeof(float)), "Malloc Xchan");
    checkCuda(cudaMalloc(&d_beam[0], bytes_beam), "Malloc Beam0");
    checkCuda(cudaMalloc(&d_beam[1], bytes_beam), "Malloc Beam1");

    checkCuda(cudaMemcpy(d_rf, rf_padded.data(), bytes_rf, cudaMemcpyHostToDevice), "H2D RF");
    checkCuda(cudaMemcpy(d_xchan, xchan.data(), (size_t)Nchan * sizeof(float), cudaMemcpyHostToDevice), "H2D Xchan");

    // ---- Interp coefficients -> constant ----
    float Interp[72] = {
        0,-0.0024f,-0.0046f,-0.0061f,-0.0068f,-0.0065f,-0.0052f,-0.0029f, 0,0.0136f,0.0258f,0.0349f,0.0395f,0.0384f,0.0312f,0.0181f,
        0,-0.045f,-0.0877f,-0.1222f,-0.1427f,-0.144f,-0.122f,-0.0743f, 0,0.1370f,0.291f,0.4522f,0.6098f,0.753f,0.8713f,0.956f,
        1.0f,0.956f,0.8713f,0.753f,0.6098f,0.4522f,0.291f,0.137f, 0,-0.0743f,-0.122f,-0.144f,-0.1427f,-0.1222f,-0.0877f,-0.045f,
        0,0.0181f,0.0312f,0.0384f,0.0395f,0.0349f,0.0258f,0.0136f, 0,-0.0029f,-0.0052f,-0.0065f,-0.0068f,-0.0061f,-0.0046f,-0.0024f
    };
    checkCuda(cudaMemcpyToSymbol(d_Interp, Interp, sizeof(Interp)), "Copy Interp");

    // ---- beam parameters ----
    float apersize = Nchan * p.pitch;
    float lambda   = p.soundv / p.fc;
    float dsin     = lambda / apersize / 2.0f;
    float drange   = p.soundv / p.fs / 2.0f / (float)UPSAMP;
    float rangeoffset = p.timeoffset * p.soundv / 2.0f;
    int   Nbeam    = (int)(sqrtf(2.0f) / dsin + 0.5f);

    float sf_scale = (p.fs * UPSAMP) / p.soundv;
    float sf_bias  = p.timeoffset * p.fs * UPSAMP;

    // ---- launch config ----
    dim3 block(32, J_TILE); // 256 threads
    dim3 grid((Nchan + TX_TILE - 1) / TX_TILE,
              (UNsample + J_TILE - 1) / J_TILE);

    // 建議：偏好 L1（shared 很小）
    checkCuda(cudaFuncSetCacheConfig(beamform_v100_corrected_kernel, cudaFuncCachePreferL1),
              "Prefer L1");

    cudaStream_t stream[2];
    cudaEvent_t  done[2];
    float* h_beam[2] = {nullptr, nullptr};

    for (int i = 0; i < 2; ++i) {
        checkCuda(cudaStreamCreate(&stream[i]), "StreamCreate");
        checkCuda(cudaEventCreate(&done[i]), "EventCreate");
        checkCuda(cudaMallocHost(&h_beam[i], bytes_beam), "MallocHost h_beam");
    }

    ofstream fout(beamfile, ios::binary);
    if (!fout) {
        cerr << "Failed to open output file: " << beamfile << endl;
        exit(EXIT_FAILURE);
    }

    const int SamplePad_x_Nchan = SamplePad * Nchan;

    for (int b = 0; b < Nbeam + 1; ++b) {
        int cur  = b & 1;
        int prev = cur ^ 1;

        if (b < Nbeam) {
            float sint = dsin * (b + 1 - (Nbeam + 1) / 2.0f);
            sint = fmaxf(-1.0f, fminf(1.0f, sint));
            float cost = sqrtf(1.0f - sint * sint);

            checkCuda(cudaMemsetAsync(d_beam[cur], 0, bytes_beam, stream[cur]), "Memset beam");

            beamform_v100_corrected_kernel<<<grid, block, 0, stream[cur]>>>(
                d_rf, d_xchan, d_beam[cur],
                Nchan, UNsample,
                SamplePad_x_Nchan,
                sf_scale, sf_bias,
                sint, cost, rangeoffset, drange);

            checkLastKernel("beamform_v100_corrected_kernel launch");

            checkCuda(cudaMemcpyAsync(h_beam[cur], d_beam[cur], bytes_beam,
                                      cudaMemcpyDeviceToHost, stream[cur]),
                      "D2H beam");

            checkCuda(cudaEventRecord(done[cur], stream[cur]), "EventRecord");
        }

        if (b > 0) {
            checkCuda(cudaEventSynchronize(done[prev]), "EventSync");
            fout.write((char*)h_beam[prev], (std::streamsize)bytes_beam);
        }
    }

    fout.close();

    for (int i = 0; i < 2; ++i) {
        cudaFree(d_beam[i]);
        cudaFreeHost(h_beam[i]);
        cudaStreamDestroy(stream[i]);
        cudaEventDestroy(done[i]);
    }
    cudaFree(d_rf);
    cudaFree(d_xchan);

    auto T1 = chrono::high_resolution_clock::now();
    cout << "Total beamforming time: "
         << chrono::duration<double>(T1 - T0).count()
         << " s" << endl;
}


