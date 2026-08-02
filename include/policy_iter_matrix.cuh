#pragma once

#include <stdio.h>
#include "cuda_runtime.h"
#include <vector>
#include "mdp_csr.h"
#include "policy_iter_fp.cuh"
#include "matrix_lib.h"

using namespace std;

void generate_matrix_A_and_vector_R_cpu(const MDP &mdp, const vector<size_t> &policy, vector<float> &A, vector<float> &R)
{
    size_t n = mdp.num_states;

    fill(A.begin(), A.end(), 0.0f);
    fill(R.begin(), R.end(), 0.0f);

    for (size_t s = 0; s < n; s++)
    {
        A[s * n + s] = 1.0f;

        size_t action = policy[s];
        size_t row = s * mdp.num_actions + action;
        size_t begin = mdp.row_ptr[row];
        size_t end = mdp.row_ptr[row + 1];

        for (size_t offset = begin; offset < end; offset++)
        {
            size_t next = mdp.next_state[offset];

            float p = mdp.prob[offset];
            float r = mdp.reward[offset];

            A[s * n + next] += -mdp.gamma * p;
            R[s] += p * r;
        }
    }
}

void policy_eval_matrix_LU_cpu(const MDP &mdp, const vector<size_t> &policy, vector<float> &state_values, vector<float> &A, vector<float> &R)
{
    size_t n = mdp.num_states;

    generate_matrix_A_and_vector_R_cpu(mdp, policy, A, R);

    using RowMajorMatrixXf =
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    Eigen::Map<RowMajorMatrixXf> A_eigen(A.data(), n, n);
    Eigen::Map<Eigen::VectorXf> R_eigen(R.data(), n);
    Eigen::Map<Eigen::VectorXf> V_eigen(state_values.data(), n);

    V_eigen = A_eigen.partialPivLu().solve(R_eigen);
}

PolicyIteration policy_iter_matrix_LU_cpu(const MDP &mdp)
{
    size_t n = mdp.num_states;
    PolicyIteration iter = {vector<size_t>(n, 0), vector<float>(n, 0.0f), false, 0};
    vector<float> A(n * n, 0.0f);
    vector<float> R(n, 0.0f);

    for (int i = 0; i < 10000; i++)
    {
        iter.num_iterations++;
        cout << "policy iteration CPU LU loop " << iter.num_iterations << endl;
        policy_eval_matrix_LU_cpu(mdp, iter.policy, iter.state_values, A, R);

        if (policy_improvement_cpu(mdp, iter.policy, iter.state_values))
        {
            iter.converged = true;
            cout << "policy iteration CPU LU completed successfully." << endl;
            break;
        }
    }
    return iter;
}

using SpMat = Eigen::SparseMatrix<float>;
using Triplet = Eigen::Triplet<float>;

void generate_sparse_matrix_A_and_vector_R_cpu(const MDP &mdp, const vector<size_t> &policy, SpMat &A, Eigen::VectorXf &R, vector<Triplet> &triplets)
{
    size_t n = mdp.num_states;

    triplets.clear();
    R.setZero(n);

    for (size_t s = 0; s < n; s++)
    {
        triplets.emplace_back(s, s, 1.0f);

        size_t action = policy[s];
        size_t row = s * mdp.num_actions + action;
        size_t begin = mdp.row_ptr[row];
        size_t end = mdp.row_ptr[row + 1];

        for (size_t offset = begin; offset < end; offset++)
        {
            size_t next = mdp.next_state[offset];

            float p = mdp.prob[offset];
            float r = mdp.reward[offset];

            triplets.emplace_back(s, next, -mdp.gamma * p);
            R[s] += p * r;
        }
    }

    A.resize(n, n);
    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();
}

void policy_eval_matrix_sparse_LU_cpu(const MDP &mdp, const vector<size_t> &policy, vector<float> &state_values, SpMat &A, Eigen::VectorXf &R, vector<Triplet> &triplets, Eigen::SparseLU<SpMat> &solver)
{
    size_t n = mdp.num_states;

    generate_sparse_matrix_A_and_vector_R_cpu(mdp, policy, A, R, triplets);

    solver.compute(A);

    Eigen::Map<Eigen::VectorXf> V_eigen(state_values.data(), n);
    V_eigen = solver.solve(R);
}

PolicyIteration policy_iter_matrix_sparse_LU_cpu(const MDP &mdp)
{
    size_t n = mdp.num_states;
    PolicyIteration iter = {vector<size_t>(n, 0), vector<float>(n, 0.0f), false, 0};
    SpMat A(n, n);
    Eigen::VectorXf R(n);
    vector<Triplet> triplets;
    triplets.reserve(mdp.num_states + mdp.prob.size());
    Eigen::SparseLU<SpMat> solver;

    for (int i = 0; i < 10000; i++)
    {
        iter.num_iterations++;
        cout << "policy iteration CPU Sparse LU loop " << iter.num_iterations << endl;
        policy_eval_matrix_sparse_LU_cpu(mdp, iter.policy, iter.state_values, A, R, triplets, solver);

        if (policy_improvement_cpu(mdp, iter.policy, iter.state_values))
        {
            iter.converged = true;
            cout << "policy iteration CPU Sparse LU completed successfully." << endl;
            break;
        }
    }
    return iter;
}

void policy_eval_matrix_BiCGSTAB_cpu(const MDP &mdp, const vector<size_t> &policy, vector<float> &state_values, SpMat &A, Eigen::VectorXf &R, vector<Triplet> &triplets, Eigen::BiCGSTAB<SpMat, Eigen::DiagonalPreconditioner<float>> &solver)
{
    size_t n = mdp.num_states;

    generate_sparse_matrix_A_and_vector_R_cpu(mdp, policy, A, R, triplets);

    solver.compute(A);

    Eigen::Map<Eigen::VectorXf> V_eigen(state_values.data(), n);
    Eigen::VectorXf V = solver.solveWithGuess(R, V_eigen);

    V_eigen = V;
}

PolicyIteration policy_iter_matrix_BiCGSTAB_cpu(const MDP &mdp, float tolerance)
{
    size_t n = mdp.num_states;
    PolicyIteration iter = {vector<size_t>(n, 0), vector<float>(n, 0.0f), false, 0};
    SpMat A(n, n);
    Eigen::VectorXf R(n);
    vector<Triplet> triplets;
    triplets.reserve(mdp.num_states + mdp.prob.size());
    Eigen::BiCGSTAB<SpMat, Eigen::DiagonalPreconditioner<float>> solver;
    solver.setTolerance(tolerance);
    solver.setMaxIterations(10000);

    for (int i = 0; i < 10000; i++)
    {
        iter.num_iterations++;
        cout << "policy iteration CPU BiCGSTAB loop " << iter.num_iterations << endl;
        policy_eval_matrix_BiCGSTAB_cpu(mdp, iter.policy, iter.state_values, A, R, triplets, solver);

        if (policy_improvement_cpu(mdp, iter.policy, iter.state_values))
        {
            iter.converged = true;
            cout << "policy iteration CPU BiCGSTAB completed successfully." << endl;
            break;
        }
    }
    return iter;
}
