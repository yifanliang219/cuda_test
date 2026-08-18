#include <stdio.h>
#include <vector>
#include "mdp_csr.h"
#include "policy_iter.h"
#include "policy_improvement_cpu.h"

using namespace std;

// Gauss–Seidel style
bool policy_eval_cpu(const MDP &mdp, const vector<size_t> &policy, vector<float> &state_values, float tolerance)
{
    for (int i = 0; i < 10000; i++)
    {
        float delta = 0.0f;
        for (size_t s = 0; s < mdp.num_states; s++)
        {
            size_t action = policy[s];
            size_t row = s * mdp.num_actions + action;
            size_t begin = mdp.row_ptr[row];
            size_t end = mdp.row_ptr[row + 1];
            float state_value = 0.0f;
            for (size_t offset = begin; offset < end; offset++)
            {
                state_value += mdp.prob[offset] * (mdp.reward[offset] + mdp.gamma * state_values[mdp.next_state[offset]]);
            }
            delta = max(delta, fabsf(state_value - state_values[s]));
            state_values[s] = state_value;
        }
        if (delta < tolerance)
        {
            return true;
        }
    }
    return false;
}

PolicyIteration policy_iter_FP_cpu(const MDP &mdp, float tolerance)
{
    PolicyIteration iter = {vector<size_t>(mdp.num_states, 0), vector<float>(mdp.num_states, 0.0f), false, 0};
    for (int i = 0; i < 50; i++)
    {
        iter.num_iterations++;
        cout << "policy iteration CPU FP loop " << iter.num_iterations << endl;
        if (!policy_eval_cpu(mdp, iter.policy, iter.state_values, tolerance))
        {
            break;
        }
        if (policy_improvement_cpu(mdp, iter.policy, iter.state_values))
        {
            iter.converged = true;
            cout << "policy iteration CPU FP completed successfully." << endl;
            break;
        }
    }
    return iter;
}
