#pragma once

#include <vector>
#include <algorithm>
#include <utility>
#include "mdp_csr.h"

using namespace std;

using SparseRow = vector<pair<size_t, float>>;
using SparseMatrix = vector<SparseRow>;

inline SparseRow::iterator findInRow(SparseRow &row, size_t col)
{
    auto it = lower_bound(row.begin(), row.end(), col,
                          [](const pair<size_t, float> &entry, size_t c) { return entry.first < c; });
    return (it != row.end() && it->first == col) ? it : row.end();
}

inline SparseRow::const_iterator findInRow(const SparseRow &row, size_t col)
{
    auto it = lower_bound(row.begin(), row.end(), col,
                          [](const pair<size_t, float> &entry, size_t c) { return entry.first < c; });
    return (it != row.end() && it->first == col) ? it : row.end();
}

inline void generate_matrix_sparse_A_and_vector_R_cpu(const MDP &mdp, const vector<size_t> &policy, SparseMatrix &A, vector<float> &R)
{
    size_t n = mdp.num_states;

    fill(R.begin(), R.end(), 0.0f);

    vector<pair<size_t, float>> scratch;

    for (size_t s = 0; s < n; s++)
    {
        size_t action = policy[s];
        size_t row = s * mdp.num_actions + action;
        size_t begin = mdp.row_ptr[row];
        size_t end = mdp.row_ptr[row + 1];

        scratch.clear();
        scratch.emplace_back(s, 1.0f);

        for (size_t offset = begin; offset < end; offset++)
        {
            size_t next = mdp.next_state[offset];

            float p = mdp.prob[offset];
            float r = mdp.reward[offset];

            scratch.emplace_back(next, -mdp.gamma * p);
            R[s] += p * r;
        }

        sort(scratch.begin(), scratch.end(),
             [](const pair<size_t, float> &a, const pair<size_t, float> &b) { return a.first < b.first; });

        SparseRow &out = A[s];
        out.clear();

        for (const auto &entry : scratch)
        {
            if (!out.empty() && out.back().first == entry.first)
            {
                out.back().second += entry.second;
            }
            else
            {
                out.push_back(entry);
            }
        }
    }
}
