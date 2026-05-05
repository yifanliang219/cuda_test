#include <stdio.h>
#include <stdlib.h>
#include "cxtimers.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "thrust/device_vector.h"
#include "image_lib.h"
#include "matrix_lib.h"

using namespace std;

int main(int argc, char *argv[])
{
    size_t num = 20;
    size_t width = 512;

    const vector<float> A = loadMatrix("data/A_f32_K20_N512.npy", num, width);
    const vector<float> B = loadMatrix("data/B_f32_K20_N512.npy", num, width);
    const vector<float> C = loadMatrix("data/C_f32_K20_N512.npy", num, width);

    vector<float> firstRowInA(A.begin(), A.begin() + width);
    vector<float> firstColInB(width);

    for (int i = 0; i < width; i++)
    {
        firstColInB[i] = *(B.begin() + i * width);
    }

    float dot = 0.0;

    for (int i = 0; i < width; i++)
    {
        dot += firstRowInA[i] * firstColInB[i];
    }

    printf("A * B = %.3f, C = %.3f\n", dot, C[0]);

    return 0;
}