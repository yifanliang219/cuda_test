#pragma once

#include <vector>
#include "mdp_csr.h"

using namespace std;

inline bool policy_improvement_cpu(const MDP &mdp, vector<size_t> &policy, const vector<float> &state_values)
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
