#include <stdio.h>
#include "cuda_runtime.h"
#include <vector>
#include "mdp_csr.h"
#include "policy_iter.h"
#include "policy_improvement_gpu.cuh"

using namespace std;

// Jacobi style
__global__ void policy_eval_kernel(float *newV, const float *oldV, const size_t *policy_d, const size_t num_states, const size_t num_actions, const float gamma, const float tolerance,
                                   const float *probs, const float *rewards, const size_t *next_states, const size_t *row_ptr, int *not_converged_d)
{
    size_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < num_states)
    {
        size_t action = policy_d[s];
        size_t row = s * num_actions + action;
        size_t begin = row_ptr[row];
        size_t end = row_ptr[row + 1];
        float state_value = 0.0f;
        for (size_t offset = begin; offset < end; offset++)
        {
            state_value += probs[offset] * (rewards[offset] + gamma * oldV[next_states[offset]]);
        }
        if (fabsf(state_value - oldV[s]) > tolerance)
        {
            atomicExch(not_converged_d, 1);
        }
        newV[s] = state_value;
    }
}

bool policy_eval_gpu(const MDP &mdp, const vector<size_t> &policy, vector<float> &state_values, const float tolerance)
{
    size_t num_states = state_values.size();
    size_t num_trans = mdp.prob.size();
    size_t size_state_values = num_states * sizeof(float);
    size_t size_p_r = num_trans * sizeof(float);
    size_t size_policy = num_states * sizeof(size_t);
    size_t size_next_states = num_trans * sizeof(size_t);
    size_t size_rowptr = mdp.row_ptr.size() * sizeof(size_t);
    float *newV, *oldV, *probs, *rewards;
    size_t *next_states, *policy_d, *rowPtr;
    cudaMalloc(&newV, size_state_values);
    cudaMalloc(&oldV, size_state_values);
    cudaMalloc(&policy_d, size_policy);
    cudaMalloc(&probs, size_p_r);
    cudaMalloc(&rewards, size_p_r);
    cudaMalloc(&next_states, size_next_states);
    cudaMalloc(&rowPtr, size_rowptr);
    cudaMemcpy(oldV, state_values.data(), size_state_values, cudaMemcpyHostToDevice);
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
    int not_converged_h = 1;
    for (size_t iter = 0; iter < 10000; iter++)
    {
        cudaMemset(not_converged_d, 0, sizeof(int));
        policy_eval_kernel<<<dimGrid, dimBlock>>>(newV, oldV, policy_d, mdp.num_states, mdp.num_actions, mdp.gamma, tolerance, probs, rewards, next_states, rowPtr, not_converged_d);
        cudaMemcpy(&not_converged_h, not_converged_d, sizeof(int), cudaMemcpyDeviceToHost);
        swap(newV, oldV);
        if (not_converged_h == 0)
        {
            break;
        }
    }
    cudaMemcpy(state_values.data(), oldV, size_state_values, cudaMemcpyDeviceToHost);
    cudaFree(newV);
    cudaFree(oldV);
    cudaFree(policy_d);
    cudaFree(probs);
    cudaFree(rewards);
    cudaFree(next_states);
    cudaFree(rowPtr);
    cudaFree(not_converged_d);
    return (not_converged_h == 0) ? true : false;
}

PolicyIteration policy_iter_gpu_better(const MDP &mdp, float tolerance)
{

    // init
    PolicyIteration iter = {vector<size_t>(mdp.num_states, 0), vector<float>(mdp.num_states, 0.0f), false, 0};
    size_t num_states = mdp.num_states;
    size_t num_trans = mdp.prob.size();
    size_t size_state_values = num_states * sizeof(float);
    size_t size_p_r = num_trans * sizeof(float);
    size_t size_policy = num_states * sizeof(size_t);
    size_t size_next_states = num_trans * sizeof(size_t);
    size_t size_rowptr = mdp.row_ptr.size() * sizeof(size_t);
    float *newV, *oldV, *probs, *rewards;
    size_t *next_states, *policy_d, *rowPtr;

    cudaMalloc(&newV, size_state_values);
    cudaMalloc(&oldV, size_state_values);
    cudaMalloc(&policy_d, size_policy);
    cudaMalloc(&probs, size_p_r);
    cudaMalloc(&rewards, size_p_r);
    cudaMalloc(&next_states, size_next_states);
    cudaMalloc(&rowPtr, size_rowptr);
    cudaMemcpy(oldV, iter.state_values.data(), size_state_values, cudaMemcpyHostToDevice);
    cudaMemcpy(policy_d, iter.policy.data(), size_policy, cudaMemcpyHostToDevice);
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

    for (int i = 0; i < 50; i++)
    {
        iter.num_iterations++;
        cout << "policy iteration GPU FP loop " << iter.num_iterations << endl;
        // policy evaluation
        int not_converged_h = 1;
        for (size_t l = 0; l < 10000; l++)
        {
            cudaMemset(not_converged_d, 0, sizeof(int));
            policy_eval_kernel<<<dimGrid, dimBlock>>>(newV, oldV, policy_d, mdp.num_states, mdp.num_actions, mdp.gamma, tolerance, probs, rewards, next_states, rowPtr, not_converged_d);
            cudaMemcpy(&not_converged_h, not_converged_d, sizeof(int), cudaMemcpyDeviceToHost);
            swap(newV, oldV);
            if (not_converged_h == 0)
            {
                break;
            }
        }
        if (not_converged_h == 1)
        {
            break;
        }

        // policy improvement
        cudaMemset(not_converged_d, 0, sizeof(int));
        not_converged_h = 1;
        policy_improvement_kernel<<<dimGrid, dimBlock>>>(policy_d, oldV, mdp.num_states, mdp.num_actions, mdp.gamma, probs, rewards, next_states, rowPtr, not_converged_d);
        cudaMemcpy(&not_converged_h, not_converged_d, sizeof(int), cudaMemcpyDeviceToHost);
        if (not_converged_h == 0)
        {
            iter.converged = true;
            //cout << "policy iteration GPU FP completed successfully." << endl;
            break;
        }
    }

    cudaMemcpy(iter.state_values.data(), oldV, size_state_values, cudaMemcpyDeviceToHost);
    cudaMemcpy(iter.policy.data(), policy_d, size_policy, cudaMemcpyDeviceToHost);
    cudaFree(newV);
    cudaFree(oldV);
    cudaFree(policy_d);
    cudaFree(probs);
    cudaFree(rewards);
    cudaFree(next_states);
    cudaFree(rowPtr);
    cudaFree(not_converged_d);

    return iter;
}

PolicyIteration policy_iter_gpu(const MDP &mdp, float tolerance)
{
    PolicyIteration iter = {vector<size_t>(mdp.num_states, 0), vector<float>(mdp.num_states, 0.0f), false, 0};
    for (int i = 0; i < 50; i++)
    {
        iter.num_iterations++;
        cout << "policy iteration GPU FP loop " << iter.num_iterations << endl;
        if (!policy_eval_gpu(mdp, iter.policy, iter.state_values, tolerance))
        {
            break;
        }
        if (policy_improvement_gpu(mdp, iter.policy, iter.state_values))
        {
            iter.converged = true;
            cout << "policy iteration GPU FP completed successfully." << endl;
            break;
        }
    }
    return iter;
}
