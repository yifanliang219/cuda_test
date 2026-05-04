#include <stdio.h>
#include <stdlib.h>
#include "cxtimers.h"
#include "cuda_runtime.h"
#include "thrust/device_vector.h"
#include <vector>
#include <random>
#include <tuple>

using namespace std;

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

__global__ void vecAddKernel(float *da, float *db, float *dc, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        dc[i] = da[i] + db[i];
    }
}

void vecAdd(const float *ha, const float *hb, float *hc, size_t n)
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

void vecAdd_cpu(const float *ha, const float *hb, float *hc, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        hc[i] = ha[i] + hb[i];
    }
}

tuple<vector<vector<float>>, vector<vector<float>>, vector<vector<float>>> initialize(size_t num_v, size_t size)
{
    vector<vector<float>> a(num_v, vector<float>(size));
    vector<vector<float>> b(num_v, vector<float>(size));
    vector<vector<float>> c(num_v, vector<float>(size));

    mt19937 rng(42);
    uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < num_v; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            a[i][j] = dist(rng);
            b[i][j] = dist(rng);
        }
    }

    return {a, b, c};
}
