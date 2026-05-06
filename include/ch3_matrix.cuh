#include <stdio.h>
#include <stdlib.h>
#include "cxtimers.h"
#include "cuda_runtime.h"
#include "thrust/device_vector.h"
#include <vector>
#include <random>
#include <tuple>

using namespace std;

__global__ void singleMatrixMulCuda(float *dA, float *dB, float *dC, size_t width)
{
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width)
    {
        float sum = 0.0f;
        for (int i = 0; i < width; i++)
        {
            sum += dA[row * width + i] * dB[col + width * i];
        }
        dC[row * width + col] = sum;
    }
}

vector<float> singleMatrixMul(const vector<float> &A, const vector<float> &B, size_t width)
{
    size_t size = width * width * sizeof(float);
    float *dA;
    float *dB;
    float *dC;
    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC, size);
    cudaMemcpy(dA, A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), size, cudaMemcpyHostToDevice);
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(ceil(width / (float)dimBlock.x), ceil(width / (float)dimBlock.y), 1);
    singleMatrixMulCuda<<<dimGrid, dimBlock>>>(dA, dB, dC, width);
    vector<float> C(width * width);
    cudaMemcpy(C.data(), dC, size, cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return C;
}