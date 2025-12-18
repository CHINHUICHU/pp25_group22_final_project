// main_cuda.cu
// Main entry point for CUDA ultrasound reconstruction
// Uses CUDA-accelerated beamforming with CPU bandpass and scan conversion

#include <iostream>
#include <fstream>
#include <vector>

#include "../sequential/params.h"
#include "../sequential/bandpass.h"
#include "beamform_cuda.cuh"
#include "../sequential/scan_convert.h"

using namespace std;

// ---------------------------------------------------------
// Load RF file to rf[tx][rx][sample], all as float
// ---------------------------------------------------------
static vector<vector<vector<float>>> load_rf_cube(
    const char* filename,
    const BFParams& p)
{
    int Nchan   = p.Nchan;
    int Nsample = p.Nsample;
    int bps     = p.bytes_per_sample;

    vector<vector<vector<float>>> rf(
        Nchan, vector<vector<float>>(Nchan, vector<float>(Nsample, 0.0f))
    );

    ifstream fin(filename, ios::binary | ios::ate);
    if (!fin) {
        cerr << "[RF] Cannot open " << filename << endl;
        exit(1);
    }

    streamsize fsize = fin.tellg();
    fin.seekg(0);

    long long expected = (long long)Nchan * Nchan * Nsample * bps;
    if (fsize != expected) {
        cerr << "[RF] File size mismatch: got " << fsize
             << ", expected " << expected << endl;
    }

    cout << "[RF] Reading " << filename
         << " as " << (bps == 2 ? "int16" : "int32")
         << " RF data...\n";

    if (bps == 2) {
        vector<short> tmp(Nsample);
        for (int tx = 0; tx < Nchan; ++tx) {
            for (int rx = 0; rx < Nchan; ++rx) {
                fin.read((char*)tmp.data(), Nsample * sizeof(short));
                for (int i = 0; i < Nsample; ++i)
                    rf[tx][rx][i] = (float)tmp[i];
            }
        }
    } else if (bps == 4) {
        vector<int32_t> tmp(Nsample);
        for (int tx = 0; tx < Nchan; ++tx) {
            for (int rx = 0; rx < Nchan; ++rx) {
                fin.read((char*)tmp.data(), Nsample * sizeof(int32_t));
                for (int i = 0; i < Nsample; ++i)
                    rf[tx][rx][i] = (float)tmp[i];
            }
        }
    } else {
        cerr << "[RF] Unsupported bytes_per_sample = " << bps << endl;
        exit(1);
    }

    fin.close();
    return rf;
}

int main(int argc, char** argv)
{
    if (argc != 5) {
        cout << "Usage: ultrasound_cuda input.dat params.txt beamout.dat output.png\n";
        return -1;
    }

    const char* rf_file   = argv[1];
    const char* txt_file  = argv[2];
    const char* beam_file = argv[3];
    const char* png_file  = argv[4];

    // 1) Load parameters
    BFParams p = load_params(txt_file);

    // 2) Load RF data (float cube)
    auto rf = load_rf_cube(rf_file, p);

    // 3) Bandpass FIR (1.5–6 MHz, 41-tap) - CPU version
    cout << "[Main] Applying 41-tap bandpass (1.5–6 MHz)...\n";
    bandpass_apply_all(rf, p.Nchan, p.Nsample,
                       p.fs,
                       p.has_bp,
                       p.bp_low,
                       p.bp_high);

    // 4) CUDA Beamforming
    cout << "[Main] Starting CUDA beamforming...\n";
    run_beamform_cuda(rf, p, beam_file);

    // 5) Scan conversion + B-mode PNG - CPU version
    run_scan_conversion(beam_file, p, png_file);

    cout << "All done (CUDA version)." << endl;
    return 0;
}
