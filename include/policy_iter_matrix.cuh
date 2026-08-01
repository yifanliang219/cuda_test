#pragma once

#include <stdio.h>
#include "cuda_runtime.h"
#include <vector>
#include "mdp_csr.h"
#include "policy_iter.h"

using namespace std;

void generate_matrix_A_and_vector_R_cpu(
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