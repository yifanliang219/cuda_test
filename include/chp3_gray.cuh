#include <stdio.h>
#include <stdlib.h>
#include "cxtimers.h"
#include "cuda_runtime.h"
#include "thrust/device_vector.h"
#include <vector>
#include <random>
#include <tuple>

using namespace std;

__global__ void colorToGrayCuda(unsigned char *dout, unsigned char *din, size_t y, size_t x)
{
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < y && col < x)
    {
        size_t pixel_offset = row * x + col;
        size_t rgb_offset = pixel_offset * 3;
        unsigned char r = din[rgb_offset];
        unsigned char g = din[rgb_offset + 1];
        unsigned char b = din[rgb_offset + 2];
        dout[pixel_offset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}

vector<unsigned char> colorToGray(const vector<unsigned char> &img_in, size_t y, size_t x)
{
    size_t size_in = x * y * 3 * sizeof(unsigned char);
    size_t size_out = x * y * sizeof(unsigned char);
    unsigned char *din, *dout;
    cudaMalloc(&din, size_in);
    cudaMalloc(&dout, size_out);
    cudaMemcpy(din, img_in.data(), size_in, cudaMemcpyHostToDevice);
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(ceil(x / (float)dimBlock.x), ceil(y / (float)dimBlock.y), 1);
    colorToGrayCuda<<<dimGrid, dimBlock>>>(dout, din, y, x);
    vector<unsigned char> out(size_out);
    cudaMemcpy(out.data(), dout, size_out, cudaMemcpyDeviceToHost);
    cudaFree(din);
    cudaFree(dout);
    return out;
}