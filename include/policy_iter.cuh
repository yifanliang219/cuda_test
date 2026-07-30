#pragma once

#include <stdio.h>
#include "cuda_runtime.h"
#include <vector>
#include "mdp_csr.h"

using namespace std;

struct PolicyIteration
{
    vector<size_t> policy;
    vector<float> state_values;
    bool converged;
    size_t num_iterations;
};

// Gauss–Seidel style
bool policy_eval_cpu(const MDP &mdp, const vector<size_t> &policy, vector<float> &state_values, float tolerance)
{
    for (int i = 0; i < 10000; i++)
    {
        float delta = 0.0f;
        for (size_t s = 0; s < mdp.num_states; s++)
        {
            size_t action = policy[s];
            size_t row = s * mdp.num_actions + action;
            size_t begin = mdp.row_ptr[row];
            size_t end = mdp.row_ptr[row + 1];
            float state_value = 0.0f;
            for (size_t offset = begin; offset < end; offset++)
            {
                state_value += mdp.prob[offset] * (mdp.reward[offset] + mdp.gamma * state_values[mdp.next_state[offset]]);
            }
            delta = max(delta, fabsf(state_value - state_values[s]));
            state_values[s] = state_value;
        }
        if (delta < tolerance)
        {
            return true;
        }
    }
    return false;
}

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

bool policy_improvement_cpu(const MDP &mdp, vector<size_t> &policy, const vector<float> &state_values)
{
    bool converged = true;
    for (size_t s = 0; s < mdp.num_states; s++)
    {
        size_t action = policy[s];
        float best_q = -INFINITY;
        size_t best_action = action;
        for (size_t a = 0; a < mdp.num_actions; a++)
        {
            size_t row = s * mdp.num_actions + a;
            size_t begin = mdp.row_ptr[row];
            size_t end = mdp.row_ptr[row + 1];
            float q_value = 0.0f;
            for (size_t offset = begin; offset < end; offset++)
            {
                q_value += mdp.prob[offset] * (mdp.reward[offset] + mdp.gamma * state_values[mdp.next_state[offset]]);
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
            converged = false;
        }
    }
    return converged;
}

__global__ void policy_improvement_kernel(size_t *policy, const float *state_values, const size_t num_states, const size_t num_actions, const float gamma,
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

bool policy_improvement_gpu(const MDP &mdp, vector<size_t> &policy, const vector<float> &state_values)
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

PolicyIteration policy_iter_cpu(const MDP &mdp, float tolerance)
{
    PolicyIteration iter = {vector<size_t>(mdp.num_states, 0), vector<float>(mdp.num_states, 0.0f), false, 0};
    for (int i = 0; i < 10000; i++)
    {
        iter.num_iterations++;
        if (!policy_eval_cpu(mdp, iter.policy, iter.state_values, tolerance))
        {
            break;
        }
        if (policy_improvement_cpu(mdp, iter.policy, iter.state_values))
        {
            iter.converged = true;
            break;
        }
    }
    return iter;
}

PolicyIteration policy_iter_gpu_better(const MDP &mdp, float tolerance)
{

    // init
    PolicyIteration iter = {vector<size_t>(mdp.num_states, 0), vector<float>(mdp.num_states, 0.0f), false, 0};
    size_t num_states = iter.state_values.size();
    size_t num_trans = mdp.prob.size();
    size_t size_state_values = num_states * sizeof(float);
    size_t size_p_r = num_trans * sizeof(float);
    size_t size_policy = num_states * sizeof(size_t);
    size_t size_next_states = num_trans * sizeof(size_t);
    size_t size_rowptr = mdp.row_ptr.size() * sizeof(size_t);

    for (int i = 0; i < 10000; i++)
    {
        iter.num_iterations++;

        // policy evaluation
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
        int not_converged_h = 1;
        // loop until state values converge
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
        cudaMemcpy(iter.state_values.data(), oldV, size_state_values, cudaMemcpyDeviceToHost);
        if (not_converged_h == 1)
        {
            break;
        }

        // policy improvement
        cudaMemset(not_converged_d, 0, sizeof(int));
        not_converged_h = 1;
        policy_improvement_kernel<<<dimGrid, dimBlock>>>(policy_d, newV, mdp.num_states, mdp.num_actions, mdp.gamma, probs, rewards, next_states, rowPtr, not_converged_d);
        cudaMemcpy(&not_converged_h, not_converged_d, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(iter.policy.data(), policy_d, size_policy, cudaMemcpyDeviceToHost);
        cudaFree(newV);
        cudaFree(oldV);
        cudaFree(policy_d);
        cudaFree(probs);
        cudaFree(rewards);
        cudaFree(next_states);
        cudaFree(rowPtr);
        cudaFree(not_converged_d);
        if (not_converged_h == 0)
        {
            iter.converged = true;
            break;
        }
    }
    return iter;
}

PolicyIteration policy_iter_gpu(const MDP &mdp, float tolerance)
{
    PolicyIteration iter = {vector<size_t>(mdp.num_states, 0), vector<float>(mdp.num_states, 0.0f), false, 0};
    for (int i = 0; i < 10000; i++)
    {
        iter.num_iterations++;
        if (!policy_eval_gpu(mdp, iter.policy, iter.state_values, tolerance))
        {
            break;
        }
        if (policy_improvement_gpu(mdp, iter.policy, iter.state_values))
        {
            iter.converged = true;
            break;
        }
    }
    return iter;
}

void printPolicyIter(PolicyIteration iter)
{
    if (iter.converged)
    {
        cout << "policy iteration succeed in iteration " << iter.num_iterations << ".\n";
    }
    else
    {
        cout << "policy evaluation failed to converge in iteration " << iter.num_iterations << ".\n";
    }
    cout << "optimal policy is (";
    for (size_t a : iter.policy)
    {
        cout << a << " ";
    }
    cout << ")\n";
    cout << "optimal state values are (";
    for (float s : iter.state_values)
    {
        cout << s << " ";
    }
    cout << ")\n";
}