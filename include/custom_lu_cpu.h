#include <stdio.h>
#include <cstdlib>
#include <vector>
#include "mdp_csr.h"
#include <map>

using namespace std;

using SparseRow = map<size_t, float>;
using SparseMatrix = vector<SparseRow>;

void generate_matrix_sparse_A_and_vector_R_cpu(const MDP &mdp, const vector<size_t> &policy, SparseMatrix &A, vector<float> &R)
{
    size_t n = mdp.num_states;

    for (size_t s = 0; s < n; s++)
        A[s].clear();

    fill(R.begin(), R.end(), 0.0f);

    for (size_t s = 0; s < n; s++)
    {
        A[s][s] = 1.0f;

        size_t action = policy[s];
        size_t row = s * mdp.num_actions + action;
        size_t begin = mdp.row_ptr[row];
        size_t end = mdp.row_ptr[row + 1];

        for (size_t offset = begin; offset < end; offset++)
        {
            size_t next = mdp.next_state[offset];

            float p = mdp.prob[offset];
            float r = mdp.reward[offset];

            A[s][next] += -mdp.gamma * p;
            R[s] += p * r;
        }
    }
}

void custom_sparse_LU_factorisation_cpu(SparseMatrix &U, SparseMatrix &L, vector<size_t> &permutation)
{
    size_t n = U.size();

    for (size_t i = 0; i < n; i++)
    {
        L[i].clear();
        permutation[i] = i;
    }

    for (size_t k = 0; k < n; k++)
    {
        size_t pivot_row = k;
        float pivot_abs = 0.0f;

        for (size_t i = k; i < n; i++)
        {
            auto it = U[i].find(k);

            if (it != U[i].end())
            {
                float value = abs(it->second);

                if (value > pivot_abs)
                {
                    pivot_abs = value;
                    pivot_row = i;
                }
            }
        }

        if (pivot_row != k)
        {
            swap(U[k], U[pivot_row]);
            swap(L[k], L[pivot_row]);
            swap(permutation[k], permutation[pivot_row]);
        }

        float pivot = U[k].find(k)->second;

        for (size_t i = k + 1; i < n; i++)
        {
            auto ik = U[i].find(k);

            if (ik == U[i].end())
            {
                continue;
            }

            float multiplier = ik->second / pivot;

            L[i][k] = multiplier;

            U[i].erase(ik);

            for (auto kj = U[k].upper_bound(k); kj != U[k].end(); ++kj)
            {
                size_t j = kj->first;
                float ukj = kj->second;
                U[i][j] -= multiplier * ukj;
            }
        }
    }
}

void custom_sparse_LU_solve_cpu(const SparseMatrix &L, const SparseMatrix &U, const vector<size_t> &permutation, const vector<float> &R, vector<float> &state_values)
{
    size_t n = U.size();
    vector<float> y(n);

    // Forward substitution: Ly = PR
    for (size_t i = 0; i < n; i++)
    {
        float sum = R[permutation[i]];
        for (const auto &[j, lij] : L[i])
        {
            sum -= lij * y[j];
        }

        y[i] = sum;
    }

    // Back substitution: UV = y
    for (size_t ii = n; ii-- > 0;)
    {
        float sum = y[ii];
        for (auto it = U[ii].upper_bound(ii); it != U[ii].end(); ++it)
        {
            size_t j = it->first;
            sum -= it->second * state_values[j];
        }
        state_values[ii] = sum / U[ii].find(ii)->second;
    }
}

void policy_eval_matrix_sparse_LU_cpu(const MDP &mdp, const vector<size_t> &policy, vector<float> &state_values, SparseMatrix &A, SparseMatrix &L, vector<float> &R, vector<size_t> &permutation)
{
    generate_matrix_sparse_A_and_vector_R_cpu(mdp, policy, A, R);

    // A is modified in-place and becomes U
    custom_sparse_LU_factorisation_cpu(A, L, permutation);
    custom_sparse_LU_solve_cpu(L, A, permutation, R, state_values);
}

PolicyIteration policy_iter_matrix_custom_sparse_LU_cpu(const MDP &mdp)
{
    size_t n = mdp.num_states;
    PolicyIteration iter = {vector<size_t>(n, 0), vector<float>(n, 0.0f), false, 0};
    SparseMatrix A(n);
    SparseMatrix L(n);
    vector<float> R(n);
    vector<size_t> permutation(n);

    for (int i = 0; i < 10000; i++)
    {
        iter.num_iterations++;
        cout << "policy iteration CPU Custom Sparse LU loop " << iter.num_iterations << endl;
        policy_eval_matrix_sparse_LU_cpu(mdp, iter.policy, iter.state_values, A, L, R, permutation);
        if (policy_improvement_cpu(mdp, iter.policy, iter.state_values))
        {
            iter.converged = true;
            break;
        }
    }

    return iter;
}