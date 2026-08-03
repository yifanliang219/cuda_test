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

bool checkSamePolicy(PolicyIteration iter1, PolicyIteration iter2)
{
    for (size_t i = 0; i < iter1.state_values.size(); i++)
    {
        if (iter1.policy[i] != iter2.policy[i])
        {
            return false;
        }
    }
    return true;
}

bool checkSameValues(PolicyIteration iter1, PolicyIteration iter2)
{
    for (size_t i = 0; i < iter1.state_values.size(); i++)
    {
        if (fabs(iter1.state_values[i] - iter2.state_values[i]) > 0.0001f)
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

    // vector<MDP> mdps = generate_random_MDPs(1, 4096, 64, 0.95f, 123);
    // save_mdp(mdps[0], "data/4096_64.npz");

    string num_states = "4096";
    string num_actions = "16";
    if (argc >= 2)
        num_states = argv[1];
    if (argc >= 3)
        num_actions = argv[2];

    string mdp_file = num_states + "_" + num_actions;

    MDP loaded = load_mdp("data/" + mdp_file + ".npz");
    // // print_MDP(mdps[0]);

    cudaFree(0);

    cx::timer tim;

    tim.start();
    PolicyIteration iter_cpu = policy_iter_cpu(loaded, 1e-6f);
    double cpu_time = tim.lap_ms();

    tim.reset();
    tim.start();
    PolicyIteration iter_BiCGSTAB_cpu = policy_iter_matrix_BiCGSTAB_cpu(loaded, 1e-6f);
    // cudaDeviceSynchronize();
    double cpu_BiCGSTAB_time = tim.lap_ms();

    tim.reset();
    tim.start();
    PolicyIteration iter_better_gpu = policy_iter_gpu_better(loaded, 1e-6f);
    cudaDeviceSynchronize();
    double gpu_better_time = tim.lap_ms();

    // printPolicyIter(iter_cpu);
    // printPolicyIter(iter_gpu);

    cout << "cpu time: " << cpu_time << ", cpu BiCGSTAB time: " << cpu_BiCGSTAB_time << ", gpu better time: " << gpu_better_time << endl;

    cout << "same policy: " << (checkSamePolicy(iter_cpu, iter_BiCGSTAB_cpu) && checkSamePolicy(iter_cpu, iter_better_gpu)) << endl;

    cout << "same values: " << (checkSameValues(iter_cpu, iter_BiCGSTAB_cpu) && checkSameValues(iter_cpu, iter_better_gpu)) << endl;

    cout << "converged: " << iter_cpu.converged << ", " << iter_BiCGSTAB_cpu.converged << ", " << iter_better_gpu.converged << endl;

    return 0;
}