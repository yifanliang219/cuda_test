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
#include "policy_iter_fp.cuh"
#include "policy_iter_matrix.cuh"

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

    // vector<float> A, B, C;
    // loadMatrices(A, B, C);

    // const vector<float> A1(A.begin(), A.begin() + 512 * 512);
    // const vector<float> B1(B.begin(), B.begin() + 512 * 512);
    // const vector<float> C_cuda = singleMatrixMul(A1, B1, 512);
    // const vector<float> C_eigen = eigenRefMatrixMul(A1, B1, 1, 512);

    // vector<MDP> mdps = generate_random_MDPs(1, 50960, 64, 0.95f, 123);
    // save_mdp(mdps[0], "data/50960_64.npz");

    MDP loaded = load_mdp("data/5096_16.npz");
    // // print_MDP(mdps[0]);

    cudaFree(0);

    cx::timer tim;

    tim.start();
    PolicyIteration iter_cpu = policy_iter_cpu(loaded, 1e-6f);
    double cpu_time = tim.lap_ms();

    tim.reset();
    tim.start();
    PolicyIteration iter_LU_cpu = policy_iter_matrix_LU_cpu(loaded);
    cudaDeviceSynchronize();
    double cpu_LU_time = tim.lap_ms();

    tim.reset();
    tim.start();
    PolicyIteration iter_better_gpu = policy_iter_gpu_better(loaded, 1e-6f);
    cudaDeviceSynchronize();
    double gpu_better_time = tim.lap_ms();

    // printPolicyIter(iter_cpu);
    // printPolicyIter(iter_gpu);

    cout << "cpu time: " << cpu_time << ", cpu LU time: " << cpu_LU_time << ", gpu better time: " << gpu_better_time << endl;

    cout << "same policy and same values: " << (checkEqual(iter_cpu, iter_LU_cpu) && checkEqual(iter_cpu, iter_better_gpu)) << endl;

    cout << "converged: " << iter_cpu.converged << ", " << iter_LU_cpu.converged << ", " << iter_better_gpu.converged << endl;
    return 0;
}