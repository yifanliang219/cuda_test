#include <stdio.h>
#include <stdlib.h>
#include "cxtimers.h"
#include "cuda_runtime.h"
#include "thrust/device_vector.h"
#include "exercise2.cuh"

int main(int argc, char *argv[])
{
    size_t num_v = 32;
    size_t size = 1 << 23;
    auto [a, b, c] = initialize(num_v, size);

    cx::timer tim;

    for (size_t i = 0; i < num_v; i++)
    {
        vecAdd_cpu(a[i].data(), b[i].data(), c[i].data(), size);
    }

    double cpu_time = tim.lap_ms();

    printf("vector additions completed, cpu time %.3f ms.\n", cpu_time);

    tim.reset();
    tim.start();

    for (size_t i = 0; i < num_v; i++)
    {
        vecAdd(a[i].data(), b[i].data(), c[i].data(), size);
    }
    double gpu_time = tim.lap_ms();

    printf("vector additions completed, gpu time %.3f ms.\n", gpu_time);

    return 0;
}