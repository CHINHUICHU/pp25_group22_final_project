// #define STB_IMAGE_IMPLEMENTATION
// #include "stb_image.h"

// #include <iostream>
// #include <cmath>

// using namespace std;

// int main(int argc, char** argv)
// {
//     if (argc != 3) {
//         cout << "Usage: validate result.png truth.png\n";
//         return 0;
//     }

//     const char* f_res   = argv[1];
//     const char* f_truth = argv[2];

//     int w1, h1, c1;
//     int w2, h2, c2;

//     unsigned char* img1 = stbi_load(f_res, &w1, &h1, &c1, 1);  // force gray
//     unsigned char* img2 = stbi_load(f_truth, &w2, &h2, &c2, 1);

//     if (!img1 || !img2) {
//         cout << "Error: cannot load images." << endl;
//         return -1;
//     }

//     if (w1 != w2 || h1 != h2) {
//         cout << "Error: image size mismatch." << endl;
//         stbi_image_free(img1);
//         stbi_image_free(img2);
//         return -1;
//     }

//     long total = (long)w1 * h1;
//     long same  = 0;

//     for (long i = 0; i < total; i++) {
//         int diff = abs((int)img1[i] - (int)img2[i]);

//         if (diff <= 1)  // tolerance: ±1
//             same++;
//     }

//     double acc = (double)same / (double)total * 100.0;

//     cout << "Accuracy = " << acc << "%" << endl;

//     if (acc >= 98.0)
//         cout << "accepted" << endl;
//     else
//         cout << "wrong" << endl;

//     stbi_image_free(img1);
//     stbi_image_free(img2);

//     return 0;
// }
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>

using namespace std;

int main(int argc, char** argv)
{
    if (argc != 3) {
        cout << "Usage: validate result.png truth.png\n";
        return 0;
    }

    const char* f_res   = argv[1];
    const char* f_truth = argv[2];

    int w1, h1, c1;
    int w2, h2, c2;

    unsigned char* img1 = stbi_load(f_res, &w1, &h1, &c1, 1);  // force gray
    unsigned char* img2 = stbi_load(f_truth, &w2, &h2, &c2, 1);

    if (!img1 || !img2) {
        cout << "Error: cannot load images." << endl;
        return -1;
    }

    if (w1 != w2 || h1 != h2) {
        cout << "Error: image size mismatch." << endl;
        return -1;
    }

    long total = (long)w1 * h1;
    long same_strict = 0;
    long same_relaxed = 0;
    double total_diff = 0.0;
    double mse = 0.0;

    for (long i = 0; i < total; i++) {
        int v1 = (int)img1[i];
        int v2 = (int)img2[i];
        int diff = abs(v1 - v2);

        total_diff += diff;
        mse += diff * diff;

        if (diff <= 1) same_strict++;   // Strict tolerance
        if (diff <= 5) same_relaxed++;  // Relaxed tolerance (acceptable for GPU vs CPU)
    }

    mse /= total;
    double psnr = 10.0 * log10((255.0 * 255.0) / mse);
    double avg_diff = total_diff / total;

    cout << "----------------------------------------" << endl;
    cout << "Validation Report:" << endl;
    cout << "----------------------------------------" << endl;
    cout << "Strict Accuracy (diff<=1):  " << (double)same_strict / total * 100.0 << "%" << endl;
    cout << "Relaxed Accuracy (diff<=5): " << (double)same_relaxed / total * 100.0 << "%" << endl;
    cout << "Average Pixel Error (MAE):  " << avg_diff << " / 255" << endl;
    cout << "PSNR (Quality Metric):      " << psnr << " dB" << endl;
    cout << "----------------------------------------" << endl;

    // Typically, PSNR > 30dB or Relaxed Accuracy > 95% is considered a success for GPU porting
    if (psnr > 30.0 || (double)same_relaxed / total > 0.95)
        cout << "Result: ACCEPTED (Visual Match Confirmed)" << endl;
    else
        cout << "Result: REJECTED (Significant Deviation)" << endl;

    stbi_image_free(img1);
    stbi_image_free(img2);

    return 0;
}