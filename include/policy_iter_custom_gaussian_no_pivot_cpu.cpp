#include <stdio.h>
#include <cstdlib>
#include <vector>
#include "policy_iter.h"
#include "sparse_custom_common.h"
#include "policy_improvement_cpu.h"

using namespace std;

void gaussian_eliminate_no_pivot_cpu(SparseMatrix &A, vector<float> &R)
{
    size_t n = A.size();
    vector<pair<size_t, float>> merged;

    for (size_t k = 0; k < n; k++)
    {
        auto pivot_it = findInRow(A[k], k);
        float pivot = pivot_it->second;

        for (size_t i = k + 1; i < n; i++)
        {
            auto ik = findInRow(A[i], k);

            if (ik == A[i].end())
            {
                continue;
            }

            float multiplier = ik->second / pivot;

            R[i] -= multiplier * R[k];

            auto a = A[i].begin();
            auto a_end = A[i].end();
            auto b = pivot_it + 1;
            auto b_end = A[k].end();

            merged.clear();
            merged.reserve(A[i].size() + static_cast<size_t>(b_end - b));

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

            A[i].swap(merged);
        }
    }
}

void gaussian_back_substitute_cpu(const SparseMatrix &U, const vector<float> &y, vector<float> &state_values)
{
    size_t n = U.size();

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

void policy_eval_gaussian_no_pivot_cpu(const MDP &mdp, const vector<size_t> &policy, vector<float> &state_values, SparseMatrix &A, vector<float> &R)
{
    generate_matrix_sparse_A_and_vector_R_cpu(mdp, policy, A, R);

    gaussian_eliminate_no_pivot_cpu(A, R);
    gaussian_back_substitute_cpu(A, R, state_values);
}

PolicyIteration policy_iter_gaussian_no_pivot_cpu(const MDP &mdp)
{
    size_t n = mdp.num_states;
    PolicyIteration iter = {vector<size_t>(n, 0), vector<float>(n, 0.0f), false, 0};
    SparseMatrix A(n);
    vector<float> R(n);

    for (int i = 0; i < 50; i++)
    {
        iter.num_iterations++;
        cout << "policy iteration CPU Gaussian No Pivot loop " << iter.num_iterations << endl;
        policy_eval_gaussian_no_pivot_cpu(mdp, iter.policy, iter.state_values, A, R);
        if (policy_improvement_cpu(mdp, iter.policy, iter.state_values))
        {
            iter.converged = true;
            break;
        }
    }

    return iter;
}
