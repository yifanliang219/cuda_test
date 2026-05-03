#include <stdio.h>
#include <stdlib.h>
#include "cxtimers.h"
#include "cuda_runtime.h"
#include "thrust/device_vector.h"

__host__ __device__ inline float sinsum(float x, int terms)
{
    float term = x;
    float sum = term;
    float x2 = x * x;
    for (int n = 1; n < terms; n++)
    {
        term *= -x2 / (float)(2 * n * (2 * n + 1));
        sum += term;
    }
    return sum;
}

__global__ void gpu_sin(float *sums, int steps, int terms, float step_size)
{
    int step = blockIdx.x * blockDim.x + threadIdx.x;
    if (step < steps)
    {
        float x = step_size * step;
        sums[step] = sinsum(x, terms);
    }
}

int main(int argc, char *argv[])
{
    int steps = (argc > 1) ? atoi(argv[1]) : 10000000;
    int terms = (argc > 2) ? atoi(argv[2]) : 1000;
    int threads = 256;
    int blocks = (steps + threads - 1) / threads;

    double pi = 3.14159265358979323;
    double step_size = pi / (steps - 1);

    thrust::device_vector<float> dsums(steps);
    float *dptr = thrust::raw_pointer_cast(&dsums[0]);

    cx::timer tim;
    
    gpu_sin<<<blocks, threads>>>(dptr, steps, terms, (float)step_size);
    double gpu_sum = thrust::reduce(dsums.begin(), dsums.end());

    double gpu_time = tim.lap_ms();

    gpu_sum -= 0.5 * (sinsum(0.0f, terms) + sinsum(pi, terms));
    gpu_sum *= step_size;

    printf("gpu sum = %.10f, steps %d, terms %d, time %.3f ms, threads %d\n", gpu_sum, steps, terms, gpu_time, threads);
    return 0;
}