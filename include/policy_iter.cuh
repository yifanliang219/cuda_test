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
            delta = max(delta, fabs(state_value - state_values[s]));
            state_values[s] = state_value;
        }
        if (delta < tolerance)
        {
            return true;
        }
    }
    return false;
}

__global__ void policyEvalCuda(float* state_vals, float* probs, float* rewards, size_t* next_states, size_t* row_ptr, int *not_converged)
{
}

bool policy_eval_gpu(const MDP &mdp, const vector<size_t> &policy, vector<float> &state_values, float tolerance)
{
    size_t num_states = state_values.size();
    size_t num_trans = mdp.prob.size();
    size_t size_state_values = num_states * sizeof(float);
    size_t size_p_r = num_trans * sizeof(float);
    size_t size_next_states = num_trans * sizeof(size_t);
    size_t size_rowptr = num_states * sizeof(size_t);
    float *state_vals, *probs, *rewards;
    size_t *next_states, *rowPtr;
    cudaMalloc(&state_vals, size_state_values);
    cudaMalloc(&probs, size_p_r);
    cudaMalloc(&rewards, size_p_r);
    cudaMalloc(&next_states, size_next_states);
    cudaMalloc(&rowPtr, size_rowptr);
    cudaMemcpy(state_vals, state_values.data(), size_state_values, cudaMemcpyHostToDevice);
    cudaMemcpy(probs, mdp.prob.data(), size_p_r, cudaMemcpyHostToDevice);
    cudaMemcpy(rewards, mdp.reward.data(), size_p_r, cudaMemcpyHostToDevice);
    cudaMemcpy(next_states, mdp.next_state.data(), size_next_states, cudaMemcpyHostToDevice);
    cudaMemcpy(rowPtr, mdp.row_ptr.data(), size_rowptr, cudaMemcpyHostToDevice);
    int threads = 256;
    int blocks = (num_states + threads - 1) / threads;
    dim3 dimGrid(blocks, 1, 1);
    dim3 dimBlock(threads, 1, 1);
    int* not_converged;
    cudaMalloc(&not_converged, sizeof(int));
    cudaMemset(not_converged, 0, sizeof(int));
    for (size_t s = 0; s < num_states; s++)
    {
        policyEvalCuda<<<dimGrid, dimBlock>>>(state_vals, probs, rewards, next_states, rowPtr, not_converged);
    }
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
        cout << "policy iteration succeed in iteration " << iter.num_iterations << ".\n"
             << endl;
    }
    else
    {
        cout << "policy evaluation failed to converge in iteration " << iter.num_iterations << ".\n"
             << endl;
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