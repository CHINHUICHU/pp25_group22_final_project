// main_openmp.cpp - uses OpenMP beamforming
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

#include "params.h"
#include "bandpass.h"
#include "beamform.h"
#include "scan_convert.h"

using namespace std;
using namespace std::chrono;

// ---------------------------------------------------------
// Load RF file into rf[tx][rx][sample], all as float
// No normalization, just short→float / int→float conversion
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
                    rf[tx][rx][i] = (float)tmp[i];
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
        cout << "Usage: ultrasound_openmp input.dat params.txt beamout.dat output.png\n";
        return -1;
    }

    const char* rf_file   = argv[1];
    const char* txt_file  = argv[2];
    const char* beam_file = argv[3];
    const char* png_file  = argv[4];

    auto pipeline_start = high_resolution_clock::now();

    // 1) Load parameters
    BFParams p = load_params(txt_file);

    // 2) Load RF data (float cube, no normalization)
    auto load_start = high_resolution_clock::now();
    auto rf = load_rf_cube(rf_file, p);
    auto load_end = high_resolution_clock::now();
    double load_time = duration<double>(load_end - load_start).count();

    // 3) Bandpass FIR (1.5-6 MHz, 41-tap)
    cout << "[Main] Applying 41-tap bandpass (1.5-6 MHz)...\n";
    auto bandpass_start = high_resolution_clock::now();
    bandpass_apply_all(rf, p.Nchan, p.Nsample,
                   p.fs,
                   p.has_bp,
                   p.bp_low,
                   p.bp_high);
    auto bandpass_end = high_resolution_clock::now();
    double bandpass_time = duration<double>(bandpass_end - bandpass_start).count();

    // 4) Beamforming with OpenMP
    auto beamform_start = high_resolution_clock::now();
    run_beamform_openmp(rf, p, beam_file);
    auto beamform_end = high_resolution_clock::now();
    double beamform_time = duration<double>(beamform_end - beamform_start).count();

    // 5) Scan conversion + B-mode PNG
    auto scan_start = high_resolution_clock::now();
    run_scan_conversion(beam_file, p, png_file);
    auto scan_end = high_resolution_clock::now();
    double scan_time = duration<double>(scan_end - scan_start).count();

    auto pipeline_end = high_resolution_clock::now();
    double total_time = duration<double>(pipeline_end - pipeline_start).count();

    cout << "\n========== Performance Summary ==========\n";
    cout << "RF Load Time:        " << load_time << " sec\n";
    cout << "Bandpass Time:       " << bandpass_time << " sec\n";
    cout << "Beamforming Time:    " << beamform_time << " sec\n";
    cout << "Scan Conversion Time:" << scan_time << " sec\n";
    cout << "Total Pipeline Time: " << total_time << " sec\n";
    cout << "=========================================\n";

    cout << "All done." << endl;
    return 0;
}
