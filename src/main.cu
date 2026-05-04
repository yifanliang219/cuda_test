#include <stdio.h>
#include <stdlib.h>
#include "cxtimers.h"
#include "cuda_runtime.h"
#include "thrust/device_vector.h"

template <typename T>
void cudaMallocErrorCheck(T **p, size_t size, const char *file, int line)
{
    cudaError_t err = ::cudaMalloc(reinterpret_cast<void **>(p), size);

    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

#define cudaMalloc(p, size) cudaMallocErrorCheck((p), (size), __FILE__, __LINE__)

__global__ void vecAddKernel(float *da, float *db, float *dc, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        dc[i] = da[i] + db[i];
    }
}

void vecAdd(float *ha, float *hb, float *hc, int n)
{
    size_t size = n * sizeof(float);
    float *da, *db, *dc;

    cudaMalloc(&da, size);
    cudaMalloc(&db, size);
    cudaMalloc(&dc, size);

    cudaMemcpy(da, ha, size, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, size, cudaMemcpyHostToDevice);

    vecAddKernel<<<ceil(n / 256.0), 256>>>(da, db, dc, n);

    cudaMemcpy(hc, dc, size, cudaMemcpyDeviceToHost);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
}

int main(int argc, char *argv[])
{
    return 0;
}