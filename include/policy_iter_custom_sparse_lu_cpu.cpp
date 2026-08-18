#include <stdio.h>
#include <cstdlib>
#include <vector>
#include "policy_iter.h"
#include "sparse_custom_common.h"
#include "policy_improvement_cpu.h"

using namespace std;

void custom_sparse_LU_factorisation_cpu(SparseMatrix &U, SparseMatrix &L, vector<size_t> &permutation)
{
    size_t n = U.size();

    for (size_t i = 0; i < n; i++)
    {
        L[i].clear();
        permutation[i] = i;
    }

    vector<pair<size_t, float>> merged;

    for (size_t k = 0; k < n; k++)
    {
        size_t pivot_row = k;
        float pivot_abs = 0.0f;

        for (size_t i = k; i < n; i++)
        {
            auto it = findInRow(U[i], k);

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

        auto pivot_it = findInRow(U[k], k);
        float pivot = pivot_it->second;

        for (size_t i = k + 1; i < n; i++)
        {
            auto ik = findInRow(U[i], k);

            if (ik == U[i].end())
            {
                continue;
            }

            float multiplier = ik->second / pivot;

            L[i].emplace_back(k, multiplier);

            auto a = U[i].begin();
            auto a_end = U[i].end();
            auto b = pivot_it + 1;
            auto b_end = U[k].end();

            merged.clear();
            merged.reserve(U[i].size() + static_cast<size_t>(b_end - b));

            while (a != a_end || b != b_end)
            {
                if (a != a_end && a == ik)
                {
                    ++a;
                    continue;
                }

                if (b == b_end || (a != a_end && a->first < b->first))
                {
                    merged.push_back(*a);
                    ++a;
                }
                else if (a == a_end || b->first < a->first)
                {
                    merged.emplace_back(b->first, -multiplier * b->second);
                    ++b;
                }
                else
                {
                    merged.emplace_back(a->first, a->second - multiplier * b->second);
                    ++a;
                    ++b;
                }
            }

            U[i].swap(merged);
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

        auto diag_it = findInRow(U[ii], ii);

        for (auto it = diag_it + 1; it != U[ii].end(); ++it)
        {
            sum -= it->second * state_values[it->first];
        }

        state_values[ii] = sum / diag_it->second;
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

    for (int i = 0; i < 50; i++)
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
