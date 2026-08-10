#include <stdio.h>
#include <stdlib.h>
#include <iomanip>
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

void generate_all_test_mdps()
{
    vector<int> num_states = {4096, 40960, 409600};
    vector<int> num_actions = {16, 64};
    for (int s : num_states)
    {
        for (int a : num_actions)
        {
            string path = "data/" + to_string(s) + "_" + to_string(a) + ".npz";
            vector<MDP> mdps = generate_random_MDPs(1, s, a, 0.95f, 123);
            save_mdp(mdps[0], path);
        }
    }
}

void analysis(int argc, char *argv[])
{
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
    PolicyIteration iter_cpu = policy_iter_FP_cpu(loaded, 1e-6f);
    double cpu_FP_time = tim.lap_ms();

    // tim.reset();
    // tim.start();
    // PolicyIteration iter_matrix_sparse_LU_cpu = policy_iter_matrix_sparse_LU_cpu(loaded);
    // double cpu_matrix_sparse_LU_time = tim.lap_ms();
    tim.reset();
    tim.start();
    PolicyIteration iter_matrix_custom_sparse_LU_cpu = policy_iter_matrix_custom_sparse_LU_cpu(loaded);
    double cpu_matrix_custom_sparse_LU_time = tim.lap_ms();

    tim.reset();
    tim.start();
    PolicyIteration iter_matrix_BiCGSTAB_cpu = policy_iter_matrix_BiCGSTAB_cpu(loaded, 1e-6f);
    double cpu_matrix_BiCGSTAB_time = tim.lap_ms();

    tim.reset();
    tim.start();
    PolicyIteration iter_FP_gpu = policy_iter_gpu_better(loaded, 1e-6f);
    cudaDeviceSynchronize();
    double gpu_FP_time = tim.lap_ms();

    // tim.reset();
    // tim.start();
    // PolicyIteration iter_matrix_sparse_LU_gpu = policy_iter_matrix_sparse_LU_gpu(loaded);
    // cudaDeviceSynchronize();
    // double gpu_sparse_LU_time = tim.lap_ms();

    tim.reset();
    tim.start();
    PolicyIteration iter_BiCGSTAB_gpu = policy_iter_matrix_sparse_BiCGSTAB_gpu(loaded, 1e-6f);
    cudaDeviceSynchronize();
    double gpu_BiCGSTAB_time = tim.lap_ms();

    // printPolicyIter(iter_cpu);
    // printPolicyIter(iter_gpu);

    struct ResultRow
    {
        string name;
        const PolicyIteration *iter;
        double time_ms;
    };

    vector<ResultRow> results = {
        {"CPU Fixed-Point", &iter_cpu, cpu_FP_time},
        //{"CPU Sparse LU", &iter_matrix_sparse_LU_cpu, cpu_matrix_sparse_LU_time},
        {"CPU Custom Sparse LU", &iter_matrix_custom_sparse_LU_cpu, cpu_matrix_custom_sparse_LU_time},
        {"CPU Sparse BiCGSTAB", &iter_matrix_BiCGSTAB_cpu, cpu_matrix_BiCGSTAB_time},
        {"GPU Fixed-Point", &iter_FP_gpu, gpu_FP_time},
        //{"GPU Sparse LU", &iter_matrix_sparse_LU_gpu, gpu_sparse_LU_time},
        {"GPU Sparse BiCGSTAB", &iter_BiCGSTAB_gpu, gpu_BiCGSTAB_time},
    };

    const PolicyIteration &baseline = iter_cpu;

    const int nameWidth = 22, numWidth = 12, boolWidth = 13;
    const int totalWidth = nameWidth + numWidth + numWidth + boolWidth + boolWidth + boolWidth;

    cout << "\n"
         << string(totalWidth, '=') << "\n";
    cout << left << setw(nameWidth) << "Method"
         << right << setw(numWidth) << "Time (ms)"
         << setw(numWidth) << "Iterations"
         << setw(boolWidth) << "Converged"
         << setw(boolWidth) << "Same Policy"
         << setw(boolWidth) << "Same Values" << "\n";
    cout << string(totalWidth, '-') << "\n";

    cout << fixed << setprecision(3);
    for (const ResultRow &row : results)
    {
        bool isBaseline = (row.iter == &baseline);
        string samePolicy = isBaseline ? "-" : (checkSamePolicy(baseline, *row.iter) ? "yes" : "NO");
        string sameValues = isBaseline ? "-" : (checkSameValues(baseline, *row.iter) ? "yes" : "NO");

        cout << left << setw(nameWidth) << row.name
             << right << setw(numWidth) << row.time_ms
             << setw(numWidth) << row.iter->num_iterations
             << setw(boolWidth) << (row.iter->converged ? "yes" : "NO")
             << setw(boolWidth) << samePolicy
             << setw(boolWidth) << sameValues << "\n";
    }
    cout << string(totalWidth, '=') << "\n"
         << endl;
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

    // generate_all_test_mdps();

    analysis(argc, argv);

    return 0;
}