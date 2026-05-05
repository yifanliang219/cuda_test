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

    vector<float> A = loadMatrix("A_f32_K20_N512.npy", num, width);
    vector<float> B = loadMatrix("B_f32_K20_N512.npy", num, width);

}