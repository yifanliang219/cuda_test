#include <stdio.h>
#include <vector>
#include "cuda_runtime.h"
#include "sparse_gpu_common.cuh"
#include <cudss.h>
#include "policy_iter.h"
#include "policy_improvement_gpu.cuh"

using namespace std;

void solve_sparse_AV_equals_R_cuDSS(cudssHandle_t handle, cudssConfig_t config, int n, int nnz, int *A_row_ptr_d, int *A_col_ind_d, float *A_values_d, float *R_d, float *V_d)
{
    cudssData_t data;

    cudssMatrix_t A;
    cudssMatrix_t R;
    cudssMatrix_t V;

    cudssDataCreate(handle, &data);

    cudssMatrixCreateCsr(&A, n, n, nnz, A_row_ptr_d, NULL, A_col_ind_d, A_values_d, CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_32F, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO);

    cudssMatrixCreateDn(&R, n, 1, n, R_d, CUDSS_R_32F, CUDSS_LAYOUT_COL_MAJOR);
    cudssMatrixCreateDn(&V, n, 1, n, V_d, CUDSS_R_32F, CUDSS_LAYOUT_COL_MAJOR);

    cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config, data, A, V, R);
    cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, config, data, A, V, R);
    cudssExecute(handle, CUDSS_PHASE_SOLVE, config, data, A, V, R);

    cudssMatrixDestroy(A);
    cudssMatrixDestroy(R);
    cudssMatrixDestroy(V);

    cudssDataDestroy(handle, data);
}

void policy_eval_matrix_sparse_LU_gpu(const MDP &mdp, const vector<size_t> &policy, vector<float> &state_values, cudssHandle_t cudss_handle, cudssConfig_t cudss_config, size_t *policy_d, int *A_row_ptr_d, int *A_col_ind_d, float *A_values_d, float *R_d, float *V_d, const float *prob_d, const float *reward_d, const size_t *next_state_d, const size_t *mdp_row_ptr_d)
{
    int n = static_cast<int>(mdp.num_states);

    int nnz_A = generate_sparse_A_and_R_gpu(mdp, policy, policy_d, A_row_ptr_d, A_col_ind_d, A_values_d, R_d, prob_d, reward_d, next_state_d, mdp_row_ptr_d);

    cudaMemset(V_d, 0, n * sizeof(float));

    solve_sparse_AV_equals_R_cuDSS(cudss_handle, cudss_config, n, nnz_A, A_row_ptr_d, A_col_ind_d, A_values_d, R_d, V_d);

    cudaMemcpy(state_values.data(), V_d, n * sizeof(float), cudaMemcpyDeviceToHost);
}

static PolicyIteration policy_iter_matrix_sparse_LU_gpu_impl(const MDP &mdp, cudssPivotType_t pivot_type)
{
    size_t n = mdp.num_states;
    size_t num_transitions = mdp.prob.size();

    PolicyIteration iter = {vector<size_t>(n, 0), vector<float>(n, 0.0f), false, 0};

    size_t max_nnz_A = mdp.num_states + mdp.prob.size();
    size_t *policy_d;
    size_t *next_state_d;
    size_t *mdp_row_ptr_d;

    float *prob_d;
    float *reward_d;
    int *A_row_ptr_d;
    int *A_col_ind_d;
    float *A_values_d;
    float *R_d;
    float *V_d;

    cudaMalloc(&policy_d, n * sizeof(size_t));
    cudaMalloc(&next_state_d, num_transitions * sizeof(size_t));
    cudaMalloc(&mdp_row_ptr_d, mdp.row_ptr.size() * sizeof(size_t));
    cudaMalloc(&prob_d, num_transitions * sizeof(float));
    cudaMalloc(&reward_d, num_transitions * sizeof(float));
    cudaMalloc(&A_row_ptr_d, (n + 1) * sizeof(int));
    cudaMalloc(&A_col_ind_d, max_nnz_A * sizeof(int));
    cudaMalloc(&A_values_d, max_nnz_A * sizeof(float));
    cudaMalloc(&R_d, n * sizeof(float));
    cudaMalloc(&V_d, n * sizeof(float));

    cudaMemcpy(prob_d, mdp.prob.data(), num_transitions * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(reward_d, mdp.reward.data(), num_transitions * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(next_state_d, mdp.next_state.data(), num_transitions * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(mdp_row_ptr_d, mdp.row_ptr.data(), mdp.row_ptr.size() * sizeof(size_t), cudaMemcpyHostToDevice);

    cudssHandle_t cudss_handle;
    cudssConfig_t cudss_config;

    cudssCreate(&cudss_handle);
    cudssConfigCreate(&cudss_config);
    cudssConfigSet(cudss_config, CUDSS_CONFIG_PIVOT_TYPE, &pivot_type, sizeof(pivot_type));

    for (int i = 0; i < 50; i++)
    {
        iter.num_iterations++;

        cout << "policy iteration GPU cuDSS loop " << iter.num_iterations << endl;

        policy_eval_matrix_sparse_LU_gpu(mdp, iter.policy, iter.state_values, cudss_handle, cudss_config, policy_d, A_row_ptr_d, A_col_ind_d, A_values_d, R_d, V_d, prob_d, reward_d, next_state_d, mdp_row_ptr_d);

        if (policy_improvement_gpu(mdp, iter.policy, iter.state_values))
        {
            iter.converged = true;
            cout << "policy iteration GPU cuDSS completed successfully." << endl;
            break;
        }
    }

    cudssConfigDestroy(cudss_config);
    cudssDestroy(cudss_handle);

    cudaFree(policy_d);
    cudaFree(next_state_d);
    cudaFree(mdp_row_ptr_d);
    cudaFree(prob_d);
    cudaFree(reward_d);
    cudaFree(A_row_ptr_d);
    cudaFree(A_col_ind_d);
    cudaFree(A_values_d);
    cudaFree(R_d);
    cudaFree(V_d);

    return iter;
}

PolicyIteration policy_iter_matrix_sparse_LU_gpu(const MDP &mdp)
{
    return policy_iter_matrix_sparse_LU_gpu_impl(mdp, CUDSS_PIVOT_AUTO);
}

PolicyIteration policy_iter_matrix_sparse_LU_no_pivot_gpu(const MDP &mdp)
{
    return policy_iter_matrix_sparse_LU_gpu_impl(mdp, CUDSS_PIVOT_NONE);
}
