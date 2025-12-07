#include "params.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <algorithm>

using namespace std;

static string trim(const string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

BFParams load_params(const char* filename)
{
    BFParams p{};
    p.Nchan = 128;
    p.fs = 13.8889f;
    p.fc = 3.5f;
    p.timeoffset = 29.448f;
    p.Nsample = 2048;
    p.bytes_per_sample = 2;
    p.pitch = 0.22f;
    p.soundv = 1.48f;

    ifstream fin(filename);
    if (!fin) {
        cerr << "[Params] Cannot open " << filename
             << ", using defaults.\n";
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
        else if (key == "fs")           p.fs = stof(val);
        else if (key == "fc")           p.fc = stof(val);
        else if (key == "timeoffset")   p.timeoffset = stof(val);
        else if (key == "Nsample")      p.Nsample = stoi(val);
        else if (key == "bytes_per_sample") p.bytes_per_sample = stoi(val);
        else if (key == "pitch")        p.pitch = stof(val);
        else if (key == "soundv")       p.soundv = stof(val);

        // ---- 新增的兩個 fields ----
        else if (key == "bandpass_low") {
            p.bp_low = stof(val);
            p.has_bp = true;
        }
        else if (key == "bandpass_high") {
            p.bp_high = stof(val);
            p.has_bp = true;
        }
    }

    cout << "[Params] Loaded from " << filename << ":\n";
    cout << "  Nchan           = " << p.Nchan << "\n";
    cout << "  fs (MHz)        = " << p.fs << "\n";
    cout << "  fc (MHz)        = " << p.fc << "\n";
    cout << "  timeoffset (us) = " << p.timeoffset << "\n";
    cout << "  Nsample         = " << p.Nsample << "\n";
    cout << "  bytes_per_sample= " << p.bytes_per_sample << "\n";
    cout << "  pitch (mm)      = " << p.pitch << "\n";
    cout << "  soundv (mm/us)  = " << p.soundv << "\n";

    if (p.has_bp) {
        cout << "  Custom bandpass = "
             << p.bp_low << " – " << p.bp_high << " Hz\n\n";
    } else {
        cout << "  Bandpass        = default 41-tap (1.5–6 MHz)\n\n";
    }

    return p;
}
