#pragma once

#include <vector>
#include "cuda_runtime.h"
#include "mdp_csr.h"

using namespace std;

static __global__ void policy_improvement_kernel(size_t *policy, const float *state_values, const size_t num_states, const size_t num_actions, const float gamma,
                                          const float *probs, const float *rewards, const size_t *next_states, const size_t *row_ptr, int *not_converged_d)
{
    size_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < num_states)
    {
        size_t action = policy[s];
        float best_q = -INFINITY;
        size_t best_action = action;
        for (size_t a = 0; a < num_actions; a++)
        {
            size_t row = s * num_actions + a;
            size_t begin = row_ptr[row];
            size_t end = row_ptr[row + 1];
            float q_value = 0.0f;
            for (size_t offset = begin; offset < end; offset++)
            {
                q_value += probs[offset] * (rewards[offset] + gamma * state_values[next_states[offset]]);
            }
            if (q_value > best_q)
            {
                best_q = q_value;
                best_action = a;
            }
        }
        policy[s] = best_action;
        if (best_action != action)
        {
            atomicExch(not_converged_d, 1);
        }
    }
}

inline bool policy_improvement_gpu(const MDP &mdp, vector<size_t> &policy, const vector<float> &state_values)
{
    size_t num_states = state_values.size();
    size_t num_trans = mdp.prob.size();
    size_t size_state_values = num_states * sizeof(float);
    size_t size_p_r = num_trans * sizeof(float);
    size_t size_policy = num_states * sizeof(size_t);
    size_t size_next_states = num_trans * sizeof(size_t);
    size_t size_rowptr = mdp.row_ptr.size() * sizeof(size_t);
    float *values, *probs, *rewards;
    size_t *policy_d, *next_states, *rowPtr;
    cudaMalloc(&values, size_state_values);
    cudaMalloc(&policy_d, size_policy);
    cudaMalloc(&probs, size_p_r);
    cudaMalloc(&rewards, size_p_r);
    cudaMalloc(&next_states, size_next_states);
    cudaMalloc(&rowPtr, size_rowptr);
    cudaMemcpy(values, state_values.data(), size_state_values, cudaMemcpyHostToDevice);
    cudaMemcpy(policy_d, policy.data(), size_policy, cudaMemcpyHostToDevice);
    cudaMemcpy(probs, mdp.prob.data(), size_p_r, cudaMemcpyHostToDevice);
    cudaMemcpy(rewards, mdp.reward.data(), size_p_r, cudaMemcpyHostToDevice);
    cudaMemcpy(next_states, mdp.next_state.data(), size_next_states, cudaMemcpyHostToDevice);
    cudaMemcpy(rowPtr, mdp.row_ptr.data(), size_rowptr, cudaMemcpyHostToDevice);
    int threads = 256;
    int blocks = (num_states + threads - 1) / threads;
    dim3 dimGrid(blocks, 1, 1);
    dim3 dimBlock(threads, 1, 1);
    int *not_converged_d;
    cudaMalloc(&not_converged_d, sizeof(int));
    cudaMemset(not_converged_d, 0, sizeof(int));
    int not_converged_h = 1;
    policy_improvement_kernel<<<dimGrid, dimBlock>>>(policy_d, values, mdp.num_states, mdp.num_actions, mdp.gamma, probs, rewards, next_states, rowPtr, not_converged_d);
    cudaMemcpy(&not_converged_h, not_converged_d, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(policy.data(), policy_d, size_policy, cudaMemcpyDeviceToHost);
    cudaFree(values);
    cudaFree(policy_d);
    cudaFree(probs);
    cudaFree(rewards);
    cudaFree(next_states);
    cudaFree(rowPtr);
    cudaFree(not_converged_d);
    return (not_converged_h == 0) ? true : false;
}
