#pragma once

#include <vector>
#include "cuda_runtime.h"
#include "thrust/device_vector.h"
#include "thrust/scan.h"
#include "mdp_csr.h"

using namespace std;

static __global__ void build_sparse_A_row_counts_kernel(int *A_row_ptr, const size_t *policy, size_t num_states, size_t num_actions, const size_t *next_state, const size_t *mdp_row_ptr)
{
    size_t s = blockIdx.x * blockDim.x + threadIdx.x;

    if (s == 0)
    {
        A_row_ptr[0] = 0;
    }

    if (s < num_states)
    {
        size_t action = policy[s];
        size_t row = s * num_actions + action;
        size_t begin = mdp_row_ptr[row];
        size_t end = mdp_row_ptr[row + 1];

        int count = 1;

        for (size_t offset = begin; offset < end; offset++)
        {
            size_t next = next_state[offset];

            if (next != s)
            {
                count++;
            }
        }

        A_row_ptr[s + 1] = count;
    }
}

static __global__ void fill_sparse_A_and_R_kernel(int *A_col_ind, float *A_values, float *R, const int *A_row_ptr, const size_t *policy, size_t num_states,
                                           size_t num_actions, float gamma, const float *prob, const float *reward, const size_t *next_state, const size_t *mdp_row_ptr)
{
    size_t s = blockIdx.x * blockDim.x + threadIdx.x;

    if (s < num_states)
    {
        size_t action = policy[s];
        size_t row = s * num_actions + action;
        size_t begin = mdp_row_ptr[row];
        size_t end = mdp_row_ptr[row + 1];

        int out = A_row_ptr[s];

        int diagonal_pos = out;

        A_col_ind[diagonal_pos] = static_cast<int>(s);
        A_values[diagonal_pos] = 1.0f;

        out++;

        float rhs = 0.0f;

        for (size_t offset = begin; offset < end; offset++)
        {
            size_t next = next_state[offset];

            float p = prob[offset];
            float r = reward[offset];

            rhs += p * r;

            if (next == s)
            {
                A_values[diagonal_pos] += -gamma * p;
            }
            else
            {
                A_col_ind[out] = static_cast<int>(next);
                A_values[out] = -gamma * p;
                out++;
            }
        }

        R[s] = rhs;
    }
}

inline int generate_sparse_A_and_R_gpu(const MDP &mdp, const vector<size_t> &policy, size_t *policy_d, int *A_row_ptr_d, int *A_col_ind_d, float *A_values_d, float *R_d, const float *prob_d, const float *reward_d, const size_t *next_state_d, const size_t *mdp_row_ptr_d)
{
    size_t n = mdp.num_states;

    cudaMemcpy(policy_d, policy.data(), n * sizeof(size_t), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);

    build_sparse_A_row_counts_kernel<<<blocks, threads>>>(A_row_ptr_d,  policy_d, mdp.num_states, mdp.num_actions, next_state_d, mdp_row_ptr_d);

    thrust::device_ptr<int> row_ptr_thrust = thrust::device_pointer_cast(A_row_ptr_d);

    thrust::inclusive_scan(row_ptr_thrust, row_ptr_thrust + n + 1, row_ptr_thrust);

    int nnz_A = 0;

    cudaMemcpy(&nnz_A, A_row_ptr_d + n, sizeof(int), cudaMemcpyDeviceToHost);

    fill_sparse_A_and_R_kernel<<<blocks, threads>>>(A_col_ind_d, A_values_d, R_d, A_row_ptr_d, policy_d, mdp.num_states, mdp.num_actions, mdp.gamma, prob_d, reward_d, next_state_d, mdp_row_ptr_d);

    return nnz_A;
}
