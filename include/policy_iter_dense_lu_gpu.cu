#include <stdio.h>
#include <vector>
#include "cuda_runtime.h"
#include <cusolverDn.h>
#include "mdp_csr.h"
#include "policy_iter.h"
#include "policy_improvement_gpu.cuh"

using namespace std;

__global__ void generate_dense_matrix_A_and_vector_R_kernel(float *A, float *R, const size_t *policy, size_t num_states, size_t num_actions, float gamma, const float *prob,
                                                            const float *reward, const size_t *next_state, const size_t *row_ptr)
{
    size_t s = blockIdx.x * blockDim.x + threadIdx.x;

    if (s < num_states)
    {
        size_t n = num_states;

        // Column-major index: A[col * n + row]
        A[s * n + s] = 1.0f;

        size_t action = policy[s];
        size_t row = s * num_actions + action;
        size_t begin = row_ptr[row];
        size_t end = row_ptr[row + 1];

        float rhs = 0.0f;

        for (size_t offset = begin; offset < end; offset++)
        {
            size_t next = next_state[offset];

            float p = prob[offset];
            float r = reward[offset];

            // Column-major: A[next * n + s]
            A[next * n + s] += -gamma * p;

            rhs += p * r;
        }

        R[s] = rhs;
    }
}

void policy_eval_matrix_dense_LU_gpu(const MDP &mdp, const vector<size_t> &policy, vector<float> &state_values, float *A_d, float *R_d, size_t *policy_d, float *prob_d,
                                     float *reward_d, size_t *next_state_d, size_t *row_ptr_d, cusolverDnHandle_t solver_handle, float *work_d, int *ipiv_d, int *info_d)
{
    size_t n = mdp.num_states;
    size_t size_A = n * n * sizeof(float);
    size_t size_R = n * sizeof(float);
    size_t size_policy = n * sizeof(size_t);

    cudaMemcpy(policy_d, policy.data(), size_policy, cudaMemcpyHostToDevice);
    cudaMemset(A_d, 0, size_A);
    cudaMemset(R_d, 0, size_R);

    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);

    generate_dense_matrix_A_and_vector_R_kernel<<<blocks, threads>>>(A_d, R_d, policy_d, mdp.num_states, mdp.num_actions, mdp.gamma, prob_d, reward_d, next_state_d, row_ptr_d);

    int n_int = static_cast<int>(n);
    int lda = n_int;
    int ldb = n_int;
    int nrhs = 1;

    cusolverDnSgetrf(
        solver_handle,
        n_int,
        n_int,
        A_d,
        lda,
        work_d,
        ipiv_d,
        info_d);

    cusolverDnSgetrs(
        solver_handle,
        CUBLAS_OP_N,
        n_int,
        nrhs,
        A_d,
        lda,
        ipiv_d,
        R_d,
        ldb,
        info_d);

    cudaMemcpy(state_values.data(), R_d, size_R, cudaMemcpyDeviceToHost);
}

PolicyIteration policy_iter_matrix_dense_LU_gpu(const MDP &mdp)
{
    size_t n = mdp.num_states;
    size_t num_transitions = mdp.prob.size();

    PolicyIteration iter = {vector<size_t>(n, 0), vector<float>(n, 0.0f), false, 0};

    size_t size_A = n * n * sizeof(float);
    size_t size_R = n * sizeof(float);
    size_t size_policy = n * sizeof(size_t);
    size_t size_prob_reward = num_transitions * sizeof(float);
    size_t size_next_state = num_transitions * sizeof(size_t);
    size_t size_row_ptr = mdp.row_ptr.size() * sizeof(size_t);

    float *A_d;
    float *R_d;
    float *prob_d;
    float *reward_d;
    float *work_d;
    size_t *policy_d;
    size_t *next_state_d;
    size_t *row_ptr_d;
    int *ipiv_d;
    int *info_d;

    cudaMalloc(&A_d, size_A);
    cudaMalloc(&R_d, size_R);
    cudaMalloc(&policy_d, size_policy);
    cudaMalloc(&prob_d, size_prob_reward);
    cudaMalloc(&reward_d, size_prob_reward);
    cudaMalloc(&next_state_d, size_next_state);
    cudaMalloc(&row_ptr_d, size_row_ptr);
    cudaMalloc(&ipiv_d, n * sizeof(int));
    cudaMalloc(&info_d, sizeof(int));

    cudaMemcpy(prob_d, mdp.prob.data(), size_prob_reward, cudaMemcpyHostToDevice);
    cudaMemcpy(reward_d, mdp.reward.data(), size_prob_reward, cudaMemcpyHostToDevice);
    cudaMemcpy(next_state_d, mdp.next_state.data(), size_next_state, cudaMemcpyHostToDevice);
    cudaMemcpy(row_ptr_d, mdp.row_ptr.data(), size_row_ptr, cudaMemcpyHostToDevice);

    cusolverDnHandle_t solver_handle;
    cusolverDnCreate(&solver_handle);

    int n_int = static_cast<int>(n);
    int lda = n_int;
    int lwork = 0;

    cusolverDnSgetrf_bufferSize(solver_handle, n_int, n_int, A_d, lda, &lwork);

    cudaMalloc(&work_d, lwork * sizeof(float));

    for (int i = 0; i < 50; i++)
    {
        iter.num_iterations++;

        cout << "policy iteration GPU dense LU loop " << iter.num_iterations << endl;

        policy_eval_matrix_dense_LU_gpu(mdp, iter.policy, iter.state_values, A_d, R_d, policy_d, prob_d, reward_d, next_state_d, row_ptr_d, solver_handle, work_d, ipiv_d, info_d);

        if (policy_improvement_gpu(mdp, iter.policy, iter.state_values))
        {
            iter.converged = true;
            cout << "policy iteration GPU dense LU completed successfully." << endl;
            break;
        }
    }

    cudaFree(A_d);
    cudaFree(R_d);
    cudaFree(policy_d);
    cudaFree(prob_d);
    cudaFree(reward_d);
    cudaFree(next_state_d);
    cudaFree(row_ptr_d);
    cudaFree(work_d);
    cudaFree(ipiv_d);
    cudaFree(info_d);

    cusolverDnDestroy(solver_handle);

    return iter;
}
