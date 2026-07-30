#include <stdio.h>
#include <stdlib.h>
#include "cxtimers.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "thrust/device_vector.h"
#include "image_lib.h"
#include "matrix_lib.h"
#include "ch3_matrix.cuh"
#include "mdp_csr.h"
#include "policy_iter.cuh"

using namespace std;

void loadMatrices(vector<float> &A, vector<float> &B, vector<float> &C)
{
    size_t num = 20;
    size_t width = 512;

    A = loadMatrix("data/A_f32_K20_N512.npy", num, width);
    B = loadMatrix("data/B_f32_K20_N512.npy", num, width);
    C = loadMatrix("data/C_f32_K20_N512.npy", num, width);
}

bool checkEqual(PolicyIteration iter1, PolicyIteration iter2)
{
    for (size_t i = 0; i < iter1.state_values.size(); i++)
    {
        if (fabs(iter1.state_values[i] - iter2.state_values[i]) > 0.0001f || (iter1.policy[i] != iter2.policy[i]))
        {
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[])
{

    vector<float> A, B, C;
    loadMatrices(A, B, C);

    const vector<float> A1(A.begin(), A.begin() + 512 * 512);
    const vector<float> B1(B.begin(), B.begin() + 512 * 512);
    const vector<float> C_cuda = singleMatrixMul(A1, B1, 512);
    const vector<float> C_eigen = eigenRefMatrixMul(A1, B1, 1, 512);

    vector<MDP> mdps = generate_random_MDPs(1, 5096, 12, 0.95f, 123);
    // print_MDP(mdps[0]);
    cx::timer tim;
    PolicyIteration iter_cpu = policy_iter_cpu(mdps[0], 1e-6f);
    double cpu_time = tim.lap_ms();
    tim.reset();
    tim.start();
    PolicyIteration iter_gpu = policy_iter_gpu(mdps[0], 1e-6f);
    double gpu_time = tim.lap_ms();
    tim.reset();
    tim.start();
    PolicyIteration iter_better_gpu = policy_iter_gpu_better(mdps[0], 1e-6f);
    double gpu_better_time = tim.lap_ms();
    
    //printPolicyIter(iter_cpu);
    //printPolicyIter(iter_gpu);

    cout << "cpu time: " << cpu_time << ", gpu time: " << gpu_time << ", gpu better time: " << gpu_better_time << endl;

    cout << "same policy and same values: " << (checkEqual(iter_cpu, iter_gpu) && checkEqual(iter_gpu, iter_better_gpu)) << endl;

    cout << "converged: " << iter_cpu.converged << ", " << iter_gpu.converged << ", " << iter_better_gpu.converged << endl;
    return 0;
}