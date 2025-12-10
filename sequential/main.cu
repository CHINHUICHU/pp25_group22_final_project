/**
 * Ultrasound Beamforming Pipeline (Timing + Debug + Fix)
 * 功能：
 * 1. [Timing] 輸出每一步驟的執行時間 (ms)。
 * 2. [Fix] 修復 RF 資料讀取格式 (int32 -> float)，解決 nan 問題。
 * 3. [Debug] 檢查每一步驟的訊號強度。
 * 4. [Robust] 使用 Global Memory 傳遞常數。
 */

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono> // 用於計時
#include <iomanip>

using namespace std;
using namespace std::chrono;

// =========================================================
// 0. CUDA Helper & Debug Tools
// =========================================================
#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// 檢查訊號健康度 (Debug用)
void check_signal_health(float* d_data, int N, string stage_name) {
    thrust::device_ptr<float> ptr(d_data);
    float max_val = *thrust::max_element(ptr, ptr + N);
    float min_val = *thrust::min_element(ptr, ptr + N);
    float abs_max = max(abs(max_val), abs(min_val));

    // 使用 \r 讓進度條或訊息看起來整齊，但在這裡我們直接印出
    // cout << "  [DEBUG] " << left << setw(15) << stage_name 
    //      << " | Range: [" << min_val << ", " << max_val << "]" << endl;

    if (isnan(abs_max)) {
        cout << "  \033[1;31m[ERROR] " << stage_name << " produced NaN values!\033[0m" << endl;
    } else if (abs_max == 0.0f) {
        cout << "  \033[1;33m[WARN]  " << stage_name << " produced All Zeros!\033[0m" << endl;
    }
}

// =========================================================
// 1. Structures
// =========================================================
struct BFParams {
    int   Nchan;
    float fs;
    float fc;
    float timeoffset;
    int   Nsample;
    int   bytes_per_sample;
    float pitch;
    float soundv;
};

static string trim(const string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

BFParams load_params(const char* filename) {
    BFParams p = {128, 13.8889f, 3.5f, 29.448f, 2048, 2, 0.22f, 1.48f}; 
    ifstream fin(filename);
    if (!fin) {
        cout << "[WARN] Params file not found, using defaults.\n";
        return p;
    }
    string line;
    while (getline(fin, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        auto pos = line.find('=');
        if (pos == string::npos) continue;
        string key = trim(line.substr(0, pos));
        string val = trim(line.substr(pos + 1));
        if (key == "Nchan")             p.Nchan = stoi(val);
        else if (key == "fs")          p.fs = stof(val);
        else if (key == "fc")          p.fc = stof(val);
        else if (key == "timeoffset")  p.timeoffset = stof(val);
        else if (key == "Nsample")     p.Nsample = stoi(val);
        else if (key == "bytes_per_sample") p.bytes_per_sample = stoi(val);
        else if (key == "pitch")       p.pitch = stof(val);
        else if (key == "soundv")      p.soundv = stof(val);
    }
    return p;
}

// [FIXED] 修正後的讀檔函式：正確處理 32-bit int -> float
float* load_rf_flattened(const char* filename, const BFParams& p) {
    size_t total_elements = (size_t)p.Nchan * p.Nchan * p.Nsample;
    float* data = new float[total_elements];
    
    ifstream fin(filename, ios::binary | ios::ate);
    if (!fin) { cerr << "[ERROR] Cannot open RF file: " << filename << endl; exit(1); }
    
    fin.seekg(0, ios::beg);

    if (p.bytes_per_sample == 2) {
        vector<short> tmp(total_elements);
        fin.read((char*)tmp.data(), total_elements * sizeof(short));
        for(size_t i=0; i<total_elements; ++i) data[i] = (float)tmp[i];
    } 
    else if (p.bytes_per_sample == 4) {
        // [FIX] 讀取 int32 並轉型為 float (解決 nan 問題)
        vector<int32_t> tmp(total_elements);
        fin.read((char*)tmp.data(), total_elements * sizeof(int32_t));
        for(size_t i=0; i<total_elements; ++i) data[i] = (float)tmp[i];
    } 
    else {
        // Fallback
        fin.read((char*)data, total_elements * sizeof(float));
    }
    return data;
}

// =========================================================
// 2. Kernels
// =========================================================

__global__ void k_bandpass(
    const float* __restrict__ d_in, float* __restrict__ d_out, 
    const float* __restrict__ d_FIR, int total_samples, int Nsample
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_samples) return;
    int sample_idx = idx % Nsample;
    double acc = 0.0;
    for (int k = 0; k < 41; k++) {
        int neighbor = sample_idx - k;
        if (neighbor >= 0) acc += d_FIR[k] * d_in[idx - k];
    }
    d_out[idx] = (float)acc;
}

__global__ void k_beamform(
    const float* __restrict__ d_rf, float* __restrict__ d_beam,
    const float* __restrict__ d_XChan, const float* __restrict__ d_Interp,
    int Nbeam, int UNsample, int Nsample, int Nchan,
    float dsin, float rangeoffset, float drange,
    float soundv, float fs, float timeoffset
) {
    int u_sample = blockIdx.x * blockDim.x + threadIdx.x; 
    int b = blockIdx.y; 
    if (u_sample >= UNsample || b >= Nbeam) return;

    float sint = dsin * (b + 1 - (Nbeam + 1)/2.0f);
    if (sint > 1.0f) sint = 1.0f; else if (sint < -1.0f) sint = -1.0f;
    float cost = sqrtf(1.0f - sint*sint);

    float depth = rangeoffset + u_sample * drange;
    float px = depth * sint;
    float pz = depth * cost;
    float sum_val = 0.0f;
    const int upsamp = 8; 

    for (int tx = 0; tx < Nchan; ++tx) {
        float x_tx = d_XChan[tx];
        float dx_tx = px - x_tx;
        float d_tx = sqrtf(dx_tx*dx_tx + pz*pz);
        for (int rx = 0; rx < Nchan; ++rx) {
            float x_rx = d_XChan[rx];
            float dx_rx = px - x_rx;
            float d_rx = sqrtf(dx_rx*dx_rx + pz*pz);
            float t = (d_tx + d_rx) / soundv;
            
            float sample_f_up = (t - timeoffset) * fs * upsamp;
            int idx_up = (int)(sample_f_up + 0.5f);

            if (idx_up >= 0 && idx_up < UNsample) {
                int mm = idx_up / upsamp; 
                int nn = idx_up % upsamp; 
                float val = 0.0f;
                int rf_base = (tx * Nchan + rx) * Nsample;
                for (int k = 0; k < 9; k++) {
                    int sample_idx = mm + k;
                    int weight_idx = nn + (8 - k) * 8;
                    if (sample_idx >= 0 && sample_idx < Nsample) {
                        val += d_rf[rf_base + sample_idx] * d_Interp[weight_idx];
                    }
                }
                sum_val += val;
            }
        }
    }
    d_beam[b * UNsample + u_sample] = sum_val;
}

__global__ void k_envelope(float* d_in, float* d_out, int total_len, int window) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_len) return;
    float s = 0.0f;
    int c = 0;
    for (int j = -window; j <= window; j++) {
        int k = idx + j;
        if (k >= 0 && k < total_len) {
            s += fabsf(d_in[k]);
            c++;
        }
    }
    d_out[idx] = s / (float)c;
}

__global__ void k_log_compress(float* d_data, int total_len, float global_max, float DR) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_len) return;
    float val = d_data[idx];
    float v = val / global_max;
    if (v < 1e-12f) v = 1e-12f;
    float db = 20.0f * log10f(v);
    if (db < -DR) db = -DR;
    if (db > 0.0f) db = 0.0f;
    d_data[idx] = db;
}

__global__ void k_scan_convert(
    cudaTextureObject_t tex_beam, unsigned char* d_out,
    int width, int height, float x_max, float r_min, float r_max, 
    float theta_max, float dsin, float Nbeam_f, float UNsample_f,
    float rangeoffset, float drange, float DR
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;

    float dx = (2.0f * x_max) / width;
    float dy = r_max / height;
    float x = -x_max + (ix + 0.5f) * dx;
    float y = (iy + 0.5f) * dy;

    d_out[iy * width + ix] = 0; 
    if (y <= 0.0f) return;
    float r = sqrtf(x*x + y*y);
    if (r < r_min || r > r_max) return;
    float theta = atan2f(x, y);
    if (fabsf(theta) > theta_max) return;

    float s = sinf(theta);
    float b_f = s / dsin + (Nbeam_f + 1)/2.0f - 1.0f;
    float d_f = (r - rangeoffset) / drange;

    float val_db = tex2D<float>(tex_beam, d_f + 0.5f, b_f + 0.5f);
    float gray = (val_db + DR) / DR;
    if (gray < 0.0f) gray = 0.0f;
    if (gray > 1.0f) gray = 1.0f;
    d_out[iy * width + ix] = (unsigned char)(gray * 255.0f);
}

// =========================================================
// 3. Main
// =========================================================
int main(int argc, char** argv) {
    cudaFree(0); // 強制驅動程式現在就初始化 Context
    auto total_pipeline_start = high_resolution_clock::now();

    if (argc != 5) {
        cout << "Usage: ./ultrasound_gpu input.dat params.txt beamout.dat output.png\n";
        return -1;
    }

    const char* rf_file = argv[1];
    const char* param_file = argv[2];
    const char* png_file = argv[4];

    // --- 1. Load Params ---
    BFParams p = load_params(param_file);
    size_t rf_elements = (size_t)p.Nchan * p.Nchan * p.Nsample;
    size_t rf_size = rf_elements * sizeof(float);

    cout << "===== Ultrasound GPU Pipeline (With Timing) =====" << endl;
    cout << "Params: Nchan=" << p.Nchan << " Nsample=" << p.Nsample << " fs=" << p.fs << endl;

    // --- 2. Load RF & Host Constants ---
    // [FIX] 這裡會使用修正後的 load_rf_flattened
    float* h_rf = load_rf_flattened(rf_file, p);
    
    // Check RF Health
    {
        float max_h = 0.0f;
        for(size_t i=0; i<rf_elements; i++) if(abs(h_rf[i]) > max_h) max_h = abs(h_rf[i]);
        if(isnan(max_h)) { cerr << "[ERROR] Input RF data contains NaN!" << endl; return -1; }
        // cout << "  [DEBUG] Host RF Max: " << max_h << endl;
    }

    // --- 3. GPU Allocation ---
    float *d_rf_in, *d_rf_bp;
    checkCuda(cudaMalloc(&d_rf_in, rf_size));
    checkCuda(cudaMalloc(&d_rf_bp, rf_size));
    checkCuda(cudaMemcpy(d_rf_in, h_rf, rf_size, cudaMemcpyHostToDevice));
    delete[] h_rf;

    // --- Constants Setup ---
    float h_FIR[41] = {-0.002056037f, 0.000924852f, -0.001163920f, 0.004113509f, 0.001553636f, 0.003690527f, 0.002282456f, -0.010000161f, -0.000456886f, -0.026071766f, 0.007272336f, -0.010216734f, 0.027880677f, 0.039142416f, 0.010970782f, 0.060208140f, -0.101972141f, 0.006518833f, -0.269375890f, -0.067515217f, 0.647999482f, -0.067515217f, -0.269375890f, 0.006518833f, -0.101972141f, 0.060208140f, 0.010970782f, 0.039142416f, 0.027880677f, -0.010216734f, 0.007272336f, -0.026071766f, -0.000456886f, -0.010000161f, 0.002282456f, 0.003690527f, 0.001553636f, 0.004113509f, -0.001163920f, 0.000924852f, -0.002056037f};
    vector<float> vec_XChan(p.Nchan);
    for(int i=0; i<p.Nchan; ++i) vec_XChan[i] = (i + 1 - (p.Nchan + 1) / 2.0f) * p.pitch;
    float h_Interp[72] = {0,-0.0024f,-0.0046f,-0.0061f,-0.0068f,-0.0065f,-0.0052f,-0.0029f,0,0.0136f,0.0258f,0.0349f,0.0395f,0.0384f,0.0312f,0.0181f,0,-0.045f,-0.0877f,-0.1222f,-0.1427f,-0.144f,-0.122f,-0.0743f,0,0.1370f,0.291f,0.4522f,0.6098f,0.753f,0.8713f,0.956f,1.0f,0.956f,0.8713f,0.753f,0.6098f,0.4522f,0.291f,0.137f,0,-0.0743f,-0.122f,-0.144f,-0.1427f,-0.1222f,-0.0877f,-0.045f,0,0.0181f,0.0312f,0.0384f,0.0395f,0.0349f,0.0258f,0.0136f,0,-0.0029f,-0.0052f,-0.0065f,-0.0068f,-0.0061f,-0.0046f,-0.0024f,0,0,0,0,0,0,0,0};

    float *d_FIR, *d_XChan, *d_Interp;
    checkCuda(cudaMalloc(&d_FIR, 41 * sizeof(float)));
    checkCuda(cudaMemcpy(d_FIR, h_FIR, 41 * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc(&d_XChan, p.Nchan * sizeof(float)));
    checkCuda(cudaMemcpy(d_XChan, vec_XChan.data(), p.Nchan * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc(&d_Interp, 72 * sizeof(float)));
    checkCuda(cudaMemcpy(d_Interp, h_Interp, 72 * sizeof(float), cudaMemcpyHostToDevice));

    // ==========================================
    // STEP 1: Bandpass
    // ==========================================
    auto t0 = high_resolution_clock::now();
    int blockSize = 256;
    int numBlocks = (rf_elements + blockSize - 1) / blockSize;
    k_bandpass<<<numBlocks, blockSize>>>(d_rf_in, d_rf_bp, d_FIR, rf_elements, p.Nsample);
    checkCuda(cudaDeviceSynchronize()); // Wait for kernel
    auto t1 = high_resolution_clock::now();
    
    double time_bp = duration<double, milli>(t1 - t0).count();
    printf("  [Time] Bandpass:         %7.2f ms\n", time_bp);
    check_signal_health(d_rf_bp, rf_elements, "Bandpass");
    checkCuda(cudaFree(d_rf_in));

    // ==========================================
    // STEP 2: Beamforming
    // ==========================================
    float apersize = p.Nchan * p.pitch;
    float lambda   = p.soundv / p.fc;
    float dsin     = lambda / apersize / 2.0f;
    int Nbeam      = static_cast<int>(sqrt(2.0f) / dsin + 0.5f);
    int UNsample   = p.Nsample * 8;
    float drange   = p.soundv / p.fs / 2.0f / 8.0f;
    float rangeoffset = p.timeoffset * p.soundv / 2.0f;
    size_t beam_size = Nbeam * UNsample * sizeof(float);
    size_t beam_elements = Nbeam * UNsample;
    float* d_beam;
    checkCuda(cudaMalloc(&d_beam, beam_size));
    checkCuda(cudaMemset(d_beam, 0, beam_size));

    t0 = high_resolution_clock::now();
    dim3 bf_block(256, 1);
    dim3 bf_grid((UNsample + 255)/256, Nbeam);
    k_beamform<<<bf_grid, bf_block>>>(d_rf_bp, d_beam, d_XChan, d_Interp, Nbeam, UNsample, p.Nsample, p.Nchan, dsin, rangeoffset, drange, p.soundv, p.fs, p.timeoffset);
    checkCuda(cudaDeviceSynchronize()); // Wait for kernel
    t1 = high_resolution_clock::now();

    double time_bf = duration<double, milli>(t1 - t0).count();
    printf("  [Time] Beamforming:      %7.2f ms\n", time_bf);
    check_signal_health(d_beam, beam_elements, "Beamform");
    checkCuda(cudaFree(d_rf_bp));

    // ==========================================
    // STEP 3: Envelope & Log
    // ==========================================
    float* d_env;
    checkCuda(cudaMalloc(&d_env, beam_size));
    int env_blocks = (beam_elements + 255) / 256;
    
    t0 = high_resolution_clock::now();
    k_envelope<<<env_blocks, 256>>>(d_beam, d_env, beam_elements, 8);
    thrust::device_ptr<float> t_ptr(d_env);
    float max_val = *thrust::max_element(t_ptr, t_ptr + beam_elements);
    if (max_val < 1e-6f) max_val = 1.0f;
    k_log_compress<<<env_blocks, 256>>>(d_env, beam_elements, max_val, 60.0f);
    checkCuda(cudaDeviceSynchronize()); // Wait for kernel
    t1 = high_resolution_clock::now();

    double time_env = duration<double, milli>(t1 - t0).count();
    printf("  [Time] Envelope & Log:   %7.2f ms\n", time_env);
    check_signal_health(d_env, beam_elements, "Envelope");
    checkCuda(cudaFree(d_beam));

    // ==========================================
    // STEP 4: Scan Conversion
    // ==========================================
    cudaArray* cuArray;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    checkCuda(cudaMallocArray(&cuArray, &channelDesc, UNsample, Nbeam));
    checkCuda(cudaMemcpy2DToArray(cuArray, 0, 0, d_env, UNsample*sizeof(float), UNsample*sizeof(float), Nbeam, cudaMemcpyDeviceToDevice));
    struct cudaResourceDesc resDesc2; memset(&resDesc2, 0, sizeof(resDesc2));
    resDesc2.resType = cudaResourceTypeArray; resDesc2.res.array.array = cuArray;
    struct cudaTextureDesc texDesc2; memset(&texDesc2, 0, sizeof(texDesc2));
    texDesc2.addressMode[0] = cudaAddressModeBorder; texDesc2.addressMode[1] = cudaAddressModeBorder;
    texDesc2.filterMode = cudaFilterModeLinear; texDesc2.readMode = cudaReadModeElementType;
    cudaTextureObject_t tex_beam = 0;
    checkCuda(cudaCreateTextureObject(&tex_beam, &resDesc2, &texDesc2, NULL));

    int outW = 2048, outH = 2048;
    unsigned char* d_img;
    checkCuda(cudaMalloc(&d_img, outW * outH));
    float sin_theta_max = dsin * (Nbeam - 1) / 2.0f;
    float theta_max = asinf(sin_theta_max > 1.0f ? 1.0f : sin_theta_max);
    float r_max = rangeoffset + (UNsample - 1) * drange;
    float x_max = r_max * sin_theta_max;

    t0 = high_resolution_clock::now();
    dim3 sc_block(16, 16);
    dim3 sc_grid((outW+15)/16, (outH+15)/16);
    k_scan_convert<<<sc_grid, sc_block>>>(tex_beam, d_img, outW, outH, x_max, rangeoffset, r_max, theta_max, dsin, (float)Nbeam, (float)UNsample, rangeoffset, drange, 60.0f);
    checkCuda(cudaDeviceSynchronize()); // Wait for kernel
    t1 = high_resolution_clock::now();

    double time_sc = duration<double, milli>(t1 - t0).count();
    printf("  [Time] Scan Conversion:  %7.2f ms\n", time_sc);

    // Save
    vector<unsigned char> h_img(outW * outH);
    checkCuda(cudaMemcpy(h_img.data(), d_img, outW * outH, cudaMemcpyDeviceToHost));
    if (stbi_write_png(png_file, outW, outH, 1, h_img.data(), outW))
        cout << "[System] Saved PNG: " << png_file << endl;

    // Cleanup
    cudaFree(d_env); cudaFree(d_img); cudaFree(d_FIR); cudaFree(d_XChan); cudaFree(d_Interp);
    cudaFreeArray(cuArray); cudaDestroyTextureObject(tex_beam);

    auto total_pipeline_end = high_resolution_clock::now();
    double time_total = duration<double, milli>(total_pipeline_end - total_pipeline_start).count();

    cout << "=======================================" << endl;
    printf("  [Time] TOTAL EXECUTION:  %7.2f ms\n", time_total);
    cout << "=======================================" << endl;

    return 0;
}