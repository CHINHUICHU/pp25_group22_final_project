// main.cpp
#include <iostream>
#include <fstream>
#include <vector>

#include "params.h"
#include "bandpass.h"
#include "beamform.h"
#include "scan_convert.h"

using namespace std;

// ---------------------------------------------------------
// 讀 RF 檔案到 rf[tx][rx][sample]，全部用 float 表示
// 不做 normalize，只是 short→float / float→float
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
        // 不 exit，先試著讀
    }

    cout << "[RF] Reading " << filename
         << " as " << (bps == 2 ? "int16" : "float")
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
    }else if (bps == 4) {
        vector<int32_t> tmp(Nsample);
        for (int tx = 0; tx < Nchan; ++tx) {
            for (int rx = 0; rx < Nchan; ++rx) {
                fin.read((char*)tmp.data(), Nsample * sizeof(int32_t));
                for (int i = 0; i < Nsample; ++i)
                    rf[tx][rx][i] = (float)tmp[i];   // 直接轉 float
            }
        }
    }else {
        cerr << "[RF] Unsupported bytes_per_sample = " << bps << endl;
        exit(1);
    }

    fin.close();
    return rf;
}

int main(int argc, char** argv)
{
    if (argc != 5) {
        cout << "Usage: ultrasound input.dat params.txt beamout.dat output.png\n";
        return -1;
    }

    const char* rf_file   = argv[1];
    const char* txt_file  = argv[2];
    const char* beam_file = argv[3];
    const char* png_file  = argv[4];

    // 1) 讀取參數
    BFParams p = load_params(txt_file);

    // 2) 讀取 RF (float cube，不做 normalize)
    auto rf = load_rf_cube(rf_file, p);

    // 3) Bandpass FIR (1.5–6 MHz, 41-tap)
    cout << "[Main] Applying 41-tap bandpass (1.5–6 MHz)...\n";
    bandpass_apply_all(rf, p.Nchan, p.Nsample,
                   p.fs,                 // MHz
                   p.has_bp,
                   p.bp_low,            // MHz
                   p.bp_high);

    // 4) Beamforming
    run_beamform(rf, p, beam_file);

    // 5) Scan conversion + B-mode PNG
    run_scan_conversion(beam_file, p, png_file);

    cout << "All done." << endl;
    return 0;
}
