#pragma once
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include "mdp_csr.h"
#include "policy_iter_fp.cuh"

using namespace std;

__global__ void build_sparse_A_row_counts_kernel(int *A_row_ptr, const size_t *policy, size_t num_states, size_t num_actions, const size_t *next_state, const size_t *mdp_row_ptr)
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

__global__ void fill_sparse_A_and_R_kernel(int *A_col_ind, float *A_values, float *R, const int *A_row_ptr, const size_t *policy, size_t num_states,
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


int generate_sparse_A_and_R_gpu(const MDP &mdp, const vector<size_t> &policy, size_t *policy_d, int *A_row_ptr_d, int *A_col_ind_d, float *A_values_d, float *R_d, const float *prob_d, const float *reward_d, const size_t *next_state_d, const size_t *mdp_row_ptr_d)
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

__global__ void residual_kernel(float *r, const float *b, const float *Ax, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        r[i] = b[i] - Ax[i];
    }
}

__global__ void update_p_kernel(float *p, const float *r, const float *v, float beta, float omega, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        p[i] = r[i] + beta * (p[i] - omega * v[i]);
    }
}

__global__ void compute_s_kernel(float *s, const float *r, const float *v, float alpha, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        s[i] = r[i] - alpha * v[i];
    }
}

__global__ void update_x_alpha_p_kernel(float *x, const float *p, float alpha, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        x[i] += alpha * p[i];
    }
}

__global__ void update_x_and_r_kernel(float *x, float *r, const float *p, const float *s, const float *t, float alpha, float omega, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        x[i] = x[i] + alpha * p[i] + omega * s[i];
        r[i] = s[i] - omega * t[i];
    }
}

void spmv_csr_gpu(cusparseHandle_t cusparse_handle, cusparseSpMatDescr_t matA, cusparseDnVecDescr_t vecX, cusparseDnVecDescr_t vecY, const float *x_d, float *y_d, void *spmv_buffer)
{
    float alpha = 1.0f;
    float beta = 0.0f;

    cusparseDnVecSetValues(vecX, const_cast<float *>(x_d));
    cusparseDnVecSetValues(vecY, y_d);

    cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, spmv_buffer);
}

bool solve_BiCGSTAB_gpu(int n, int nnz_A, const int *A_row_ptr_d, const int *A_col_ind_d, const float *A_values_d, const float *R_d, float *V_d, float tolerance, int max_iterations, cusparseHandle_t cusparse_handle, cublasHandle_t cublas_handle, float *r_d, float *r_hat_d, float *p_d, float *v_d, float *s_d, float *t_d, float *Ax_d)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX;
    cusparseDnVecDescr_t vecY;

    cusparseCreateCsr(&matA, n, n, nnz_A, const_cast<int *>(A_row_ptr_d), const_cast<int *>(A_col_ind_d), const_cast<float *>(A_values_d), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseCreateDnVec(&vecX, n, V_d, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, n, Ax_d, CUDA_R_32F);

    float spmv_alpha = 1.0f;
    float spmv_beta = 0.0f;
    size_t spmv_buffer_size = 0;

    cusparseSpMV_bufferSize(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &spmv_alpha, matA, vecX, &spmv_beta, vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &spmv_buffer_size);

    void *spmv_buffer;
    cudaMalloc(&spmv_buffer, spmv_buffer_size);

    spmv_csr_gpu(cusparse_handle, matA, vecX, vecY, V_d, Ax_d, spmv_buffer);

    residual_kernel<<<blocks, threads>>>(r_d, R_d, Ax_d, n);

    cublasScopy(cublas_handle, n, r_d, 1, r_hat_d, 1);

    cudaMemset(p_d, 0, n * sizeof(float));
    cudaMemset(v_d, 0, n * sizeof(float));

    float norm_R = 0.0f;
    float norm_r = 0.0f;

    cublasSnrm2(cublas_handle, n, R_d, 1, &norm_R);
    cublasSnrm2(cublas_handle, n, r_d, 1, &norm_r);

    if (norm_R < 1.0f)
    {
        norm_R = 1.0f;
    }

    float target = tolerance * norm_R;

    bool converged = false;

    if (norm_r < target)
    {
        converged = true;
    }

    float rho_old = 1.0f;
    float alpha = 1.0f;
    float omega = 1.0f;

    for (int iter = 0; iter < max_iterations && !converged; iter++)
    {
        float rho_new = 0.0f;

        cublasSdot(cublas_handle, n, r_hat_d, 1, r_d, 1, &rho_new);

        float beta = (rho_new / rho_old) * (alpha / omega);

        update_p_kernel<<<blocks, threads>>>(p_d, r_d, v_d, beta, omega, n);

        spmv_csr_gpu(cusparse_handle, matA, vecX, vecY, p_d, v_d, spmv_buffer);

        float rhat_dot_v = 0.0f;

        cublasSdot(cublas_handle, n, r_hat_d, 1, v_d, 1, &rhat_dot_v);

        alpha = rho_new / rhat_dot_v;

        compute_s_kernel<<<blocks, threads>>>(s_d, r_d, v_d, alpha, n);

        float norm_s = 0.0f;

        cublasSnrm2(cublas_handle, n, s_d, 1, &norm_s);

        if (norm_s < target)
        {
            update_x_alpha_p_kernel<<<blocks, threads>>>(V_d, p_d, alpha, n);
            converged = true;
        }
        else
        {
            spmv_csr_gpu(cusparse_handle, matA, vecX, vecY, s_d, t_d, spmv_buffer);

            float t_dot_s = 0.0f;
            float t_dot_t = 0.0f;

            cublasSdot(cublas_handle, n, t_d, 1, s_d, 1, &t_dot_s);
            cublasSdot(cublas_handle, n, t_d, 1, t_d, 1, &t_dot_t);

            omega = t_dot_s / t_dot_t;

            update_x_and_r_kernel<<<blocks, threads>>>(V_d, r_d, p_d, s_d, t_d, alpha, omega, n);

            cublasSnrm2(cublas_handle, n, r_d, 1, &norm_r);

            if (norm_r < target)
            {
                converged = true;
            }

            rho_old = rho_new;
        }
    }

    cudaFree(spmv_buffer);

    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroySpMat(matA);

    return converged;
}

bool policy_eval_matrix_sparse_BiCGSTAB_gpu(const MDP &mdp, const vector<size_t> &policy, vector<float> &state_values, size_t *policy_d, int *A_row_ptr_d, int *A_col_ind_d, float *A_values_d, float *R_d, float *V_d, const float *prob_d, const float *reward_d, const size_t *next_state_d, const size_t *mdp_row_ptr_d, cusparseHandle_t cusparse_handle, cublasHandle_t cublas_handle, float *r_d, float *r_hat_d, float *p_d, float *v_d, float *s_d, float *t_d, float *Ax_d, float tolerance)
{
    size_t n = mdp.num_states;

    int nnz_A = generate_sparse_A_and_R_gpu(mdp, policy, policy_d, A_row_ptr_d, A_col_ind_d, A_values_d, R_d, prob_d, reward_d, next_state_d, mdp_row_ptr_d);

    cudaMemcpy(V_d, state_values.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    bool converged = solve_BiCGSTAB_gpu(static_cast<int>(n), nnz_A, A_row_ptr_d, A_col_ind_d, A_values_d, R_d, V_d, tolerance, 10000, cusparse_handle, cublas_handle, r_d, r_hat_d, p_d, v_d, s_d, t_d, Ax_d);

    cudaMemcpy(state_values.data(), V_d, n * sizeof(float), cudaMemcpyDeviceToHost);

    return converged;
}

PolicyIteration policy_iter_matrix_sparse_BiCGSTAB_gpu(const MDP &mdp, float tolerance)
{
    size_t n = mdp.num_states;
    size_t num_transitions = mdp.prob.size();

    PolicyIteration iter = {vector<size_t>(n, 0), vector<float>(n, 0.0f), false, 0};

    size_t max_nnz_A = mdp.num_states + mdp.prob.size();
    size_t size_policy = n * sizeof(size_t);
    size_t size_values = n * sizeof(float);
    size_t size_prob_reward = num_transitions * sizeof(float);
    size_t size_next_state = num_transitions * sizeof(size_t);
    size_t size_mdp_row_ptr = mdp.row_ptr.size() * sizeof(size_t);
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
    float *r_d;
    float *r_hat_d;
    float *p_d;
    float *v_d;
    float *s_d;
    float *t_d;
    float *Ax_d;

    cudaMalloc(&policy_d, size_policy);
    cudaMalloc(&next_state_d, size_next_state);
    cudaMalloc(&mdp_row_ptr_d, size_mdp_row_ptr);
    cudaMalloc(&prob_d, size_prob_reward);
    cudaMalloc(&reward_d, size_prob_reward);
    cudaMalloc(&A_row_ptr_d, (n + 1) * sizeof(int));
    cudaMalloc(&A_col_ind_d, max_nnz_A * sizeof(int));
    cudaMalloc(&A_values_d, max_nnz_A * sizeof(float));
    cudaMalloc(&R_d, size_values);
    cudaMalloc(&V_d, size_values);
    cudaMalloc(&r_d, size_values);
    cudaMalloc(&r_hat_d, size_values);
    cudaMalloc(&p_d, size_values);
    cudaMalloc(&v_d, size_values);
    cudaMalloc(&s_d, size_values);
    cudaMalloc(&t_d, size_values);
    cudaMalloc(&Ax_d, size_values);

    cudaMemcpy(prob_d, mdp.prob.data(), size_prob_reward, cudaMemcpyHostToDevice);
    cudaMemcpy(reward_d, mdp.reward.data(), size_prob_reward, cudaMemcpyHostToDevice);
    cudaMemcpy(next_state_d, mdp.next_state.data(), size_next_state, cudaMemcpyHostToDevice);
    cudaMemcpy(mdp_row_ptr_d, mdp.row_ptr.data(), size_mdp_row_ptr, cudaMemcpyHostToDevice);

    cusparseHandle_t cusparse_handle;
    cublasHandle_t cublas_handle;

    cusparseCreate(&cusparse_handle);
    cublasCreate(&cublas_handle);

    for (int i = 0; i < 10000; i++)
    {
        iter.num_iterations++;

        //cout << "policy iteration GPU sparse BiCGSTAB loop " << iter.num_iterations << endl;

        bool eval_converged = policy_eval_matrix_sparse_BiCGSTAB_gpu(mdp, iter.policy, iter.state_values, policy_d, A_row_ptr_d, A_col_ind_d, A_values_d, R_d, V_d, prob_d, reward_d, next_state_d, mdp_row_ptr_d, cusparse_handle, cublas_handle, r_d, r_hat_d, p_d, v_d, s_d, t_d, Ax_d, tolerance);

        if (!eval_converged)
        {
            break;
        }

        if (policy_improvement_cpu(mdp, iter.policy, iter.state_values))
        {
            iter.converged = true;
            //cout << "policy iteration GPU sparse BiCGSTAB completed successfully." << endl;
            break;
        }
    }

    cusparseDestroy(cusparse_handle);
    cublasDestroy(cublas_handle);

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
    cudaFree(r_d);
    cudaFree(r_hat_d);
    cudaFree(p_d);
    cudaFree(v_d);
    cudaFree(s_d);
    cudaFree(t_d);
    cudaFree(Ax_d);

    return iter;
}
