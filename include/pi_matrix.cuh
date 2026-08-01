#pragma once

#include <stdio.h>
#include "cuda_runtime.h"
#include <vector>
#include "mdp_csr.h"

using namespace std;

vector<float> generate_matrix_I(size_t n)
{
    vector I(n * n, 0.0f);
    for (size_t i = 0; i < n; i++)
    {
        I[n * i + i] = 1.0f;
    }
    return I;
}

vector<float> generate_matrix_P(const MDP &mdp, const vector<size_t> &policy)
{
    size_t n = mdp.num_states;
    vector P(n * n, 0.0f);
    for (size_t s = 0; s < n; s++)
    {
        size_t action = policy[s];
        size_t row = s * n + action;
        size_t begin = mdp.row_ptr[row];
        size_t end = mdp.row_ptr[row + 1];
        for (size_t offset = begin; offset < end; offset++)
        {
            size_t next = mdp.next_state[offset];
            float p = mdp.prob[offset];
            P[s * n + next] += p;
        }
    }
    return P;
}

// vector<float> generate_matrix_A(const MDP &mdp, const vector<size_t> &policy)
// {
//     vector<float> I = generate_matrix_I(mdp.num_states);
//     vector<float> P = generate_matrix_P(mdp, policy);
//     vector<float> A(mdp.num_states * mdp.num_states, 0.0f);
//     for (size_t s = 0; s < A.size(); s++)
//     {
//         A[s] = I[s] - mdp.gamma * P[s];
//     }
//     return A;
// }

// vector<float> generate_matrix_A(const MDP &mdp, const vector<size_t> &policy)
// {
//     size_t n = mdp.num_states;

//     vector<float> A(n * n, 0.0f);

//     for (size_t s = 0; s < n; s++)
//     {
//         A[n * s + s] = 1.0f;
//         size_t action = policy[s];
//         size_t row = s * mdp.num_actions + action;
//         size_t begin = mdp.row_ptr[row];
//         size_t end = mdp.row_ptr[row + 1];
//         for (size_t offset = begin; offset < end; offset++)
//         {
//             size_t next = mdp.next_state[offset];
//             float p = mdp.prob[offset];
//             A[s * n + next] += -mdp.gamma * p;
//         }
//     }

//     return A;
// }

// vector<float> generate_vector_R(const MDP &mdp, const vector<size_t> &policy)
// {
//     size_t n = mdp.num_states;

//     vector<float> R(n, 0.0f);

//     for (size_t s = 0; s < n; s++)
//     {
//         size_t action = policy[s];
//         size_t row = s * mdp.num_actions + action;
//         size_t begin = mdp.row_ptr[row];
//         size_t end = mdp.row_ptr[row + 1];
//         for (size_t offset = begin; offset < end; offset++)
//         {
//             float p = mdp.prob[offset];
//             float r = mdp.reward[offset];
//             R[s] += p * r;
//         }
//     }

//     return R;
// }

void generate_matrix_A_and_vector_R(
    const MDP &mdp,
    const vector<size_t> &policy,
    vector<float> &A,
    vector<float> &R
)
{
    size_t n = mdp.num_states;

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