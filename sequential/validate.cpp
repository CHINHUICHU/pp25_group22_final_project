#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>
#include <cmath>

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
        stbi_image_free(img1);
        stbi_image_free(img2);
        return -1;
    }

    long total = (long)w1 * h1;
    long same  = 0;

    for (long i = 0; i < total; i++) {
        int diff = abs((int)img1[i] - (int)img2[i]);

        if (diff <= 1)  // tolerance: ±1
            same++;
    }

    double acc = (double)same / (double)total * 100.0;

    cout << "Accuracy = " << acc << "%" << endl;

    if (acc >= 98.0)
        cout << "accepted" << endl;
    else
        cout << "wrong" << endl;

    stbi_image_free(img1);
    stbi_image_free(img2);

    return 0;
}
