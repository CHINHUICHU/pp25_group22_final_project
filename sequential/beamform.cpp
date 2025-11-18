// beamform.cpp
#include "beamform.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>

using namespace std;
using namespace std::chrono;

void run_beamform(
    const vector<vector<vector<float>>>& rf,
    const BFParams& p,
    const char* beamfile)
{
    // --------------------------- Parameters ----------------------------------
    const int   Nchan      = p.Nchan;
    const float fad        = p.fs;         // MHz
    const float f0         = p.fc;         // MHz
    const float timeoffset = p.timeoffset; // us
    const int   Nsample    = p.Nsample;
    const float pitch      = p.pitch;      // mm
    const float soundv     = p.soundv;     // mm/us
    const int   upsamp     = 8;            // fixed
    //const float normal     = 200000.0f;    // 和你原本一樣

    const float apersize = Nchan * pitch;
    const float lambda   = soundv / f0;
    const float dsin     = lambda / apersize / 2.0f;

    const int   Nbeam    = static_cast<int>(sqrt(2.0f) / dsin + 0.5f);
    const int   UNsample = upsamp * Nsample;

    const float drange      = soundv / fad / 2.0f / upsamp;
    const float rangeoffset = timeoffset * soundv / 2.0f;

    cout << "===== Beamforming Parameters =====\n";
    cout << "Nchan      = " << Nchan      << "\n";
    cout << "Nbeam      = " << Nbeam      << "\n";
    cout << "Nsample    = " << Nsample    << "\n";
    cout << "UNsample   = " << UNsample   << "\n";
    cout << "fs (MHz)   = " << fad        << "\n";
    cout << "fc (MHz)   = " << f0         << "\n";
    cout << "timeoffset = " << timeoffset << " us\n";
    cout << "pitch      = " << pitch      << " mm\n";
    cout << "soundv     = " << soundv     << " mm/us\n";
    //cout << "normal     = " << normal     << "\n";
    cout << "=================================\n";

    // ------------------ Interpolation Kernel (8x) ------------------
    const float Interp[72] = {
        0,-0.0024f,-0.0046f,-0.0061f,-0.0068f,-0.0065f,-0.0052f,-0.0029f,0,0.0136f,0.0258f,0.0349f,
        0.0395f,0.0384f,0.0312f,0.0181f,0,-0.045f,-0.0877f,-0.1222f,-0.1427f,-0.144f,-0.122f,-0.0743f,0,
        0.1370f,0.291f,0.4522f,0.6098f,0.753f,0.8713f,0.956f,1.0f,0.956f,0.8713f,0.753f,0.6098f,
        0.4522f,0.291f,0.137f,0,-0.0743f,-0.122f,-0.144f,-0.1427f,-0.1222f,-0.0877f,-0.045f,0,
        0.0181f,0.0312f,0.0384f,0.0395f,0.0349f,0.0258f,0.0136f,0,-0.0029f,-0.0052f,-0.0065f,
        -0.0068f,-0.0061f,-0.0046f,-0.0024f,0,0,0,0,0,0,0,0
    };

    // ------------------ Allocate Working Buffers ------------------
    vector<float> buff(Nsample + 8);
    vector<float> buff2(UNsample + 2);
    vector<float> beamsum(UNsample);

    vector<float> xchan(Nchan);
    for (int i = 0; i < Nchan; ++i)
        xchan[i] = (i + 1 - (float)(Nchan + 1) / 2.0f) * pitch;

    // ------------------------- Prepare Output -------------------------
    ofstream fout(beamfile, ios::binary | ios::trunc);
    if (!fout) {
        cerr << "[Beamform] Cannot open output beam file: " << beamfile << endl;
        exit(1);
    }

    auto total_t0 = high_resolution_clock::now();

    // ------------------------- Beamforming Loop -------------------------
    for (int beam = 0; beam < Nbeam; beam++)
    {
        auto t0 = high_resolution_clock::now();
        cout << "Beam " << beam << "/" << (Nbeam-1) << endl;

        fill(beamsum.begin(), beamsum.end(), 0.0f);

        float sint = dsin * (beam + 1 - (float)(Nbeam + 1)/2.0f);
        sint = max(-1.0f, min(1.0f, sint));
        float cost = sqrt(1.0f - sint*sint);

        vector<float> px(UNsample), pz(UNsample);
        for (int j = 0; j < UNsample; j++) {
            float depth = rangeoffset + j*drange;
            px[j] = depth * sint;
            pz[j] = depth * cost;
        }

        for (int tx = 0; tx < Nchan; tx++)
        {
            float x_tx = xchan[tx];

            for (int rx = 0; rx < Nchan; rx++)
            {
                float x_rx = xchan[rx];

                // interpolation input
                fill(buff.begin(), buff.end(), 0.0f);
                for (int k = 0; k < Nsample; k++)
                    buff[k+4] = rf[tx][rx][k];

                // 8× interpolation
                for (int m = 0; m < UNsample; m++) {
                    int mm = m / upsamp;
                    int nn = m % upsamp;
                    buff2[m] =
                        buff[mm]   * Interp[nn + 64] +
                        buff[mm+1] * Interp[nn + 56] +
                        buff[mm+2] * Interp[nn + 48] +
                        buff[mm+3] * Interp[nn + 40] +
                        buff[mm+4] * Interp[nn + 32] +
                        buff[mm+5] * Interp[nn + 24] +
                        buff[mm+6] * Interp[nn + 16] +
                        buff[mm+7] * Interp[nn +  8] +
                        buff[mm+8] * Interp[nn];
                }

                // Delay and Sum
                for (int j = 0; j < UNsample; j++)
                {
                    float dx_tx = px[j] - x_tx;
                    float dx_rx = px[j] - x_rx;
                    float dz    = pz[j];

                    float d_tx = sqrt(dx_tx*dx_tx + dz*dz);
                    float d_rx = sqrt(dx_rx*dx_rx + dz*dz);

                    float t = (d_tx + d_rx) / soundv;
                    float sample_f = (t - timeoffset) * fad * upsamp;
                    int idx = (int)(sample_f + 0.5f);

                    if (idx >= 0 && idx < UNsample)
                        beamsum[j] += buff2[idx];
                }
            }
        }

        fout.write((char*)beamsum.data(), UNsample*sizeof(float));

        auto t1 = high_resolution_clock::now();
        cout << "Beam time = "
             << duration<double,milli>(t1 - t0).count()
             << " ms\n";
    }

    fout.close();

    auto total_t1 = high_resolution_clock::now();
    cout << "================ Beamforming Finished ================\n";
    cout << "Total time = "
         << duration<double>(total_t1 - total_t0).count()
         << " sec\n";
}
