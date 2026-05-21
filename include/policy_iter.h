#include <stdio.h>

struct PolicyIteration
{
    vector<size_t> policy;
    vector<float> state_values;
    bool converged;
};

bool policy_eval(const MDP &mdp, const vector<size_t> &policy, vector<float> &state_values, float tolerance)
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
            delta = max(delta, fabs(state_value - state_values[s]));
            state_values[s] = state_value;
        }
        if (delta < tolerance)
        {
            return true;
        }
    }
    return false;
}

bool policy_improvement(const MDP &mdp, vector<size_t> &policy, const vector<float> &state_values)
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

PolicyIteration policy_iter(const MDP &mdp, float tolerance)
{
    PolicyIteration iter = {vector<size_t>(mdp.num_states, 0), vector<float>(mdp.num_states, 0.0f), false};
    for (int i = 0; i < 10000; i++)
    {
        if (!policy_eval(mdp, iter.policy, iter.state_values, tolerance))
        {
            cout << "policy evaluation failed to converge in iteration " << i << ".\n"
                 << endl;
            break;
        }
        if (policy_improvement(mdp, iter.policy, iter.state_values) == true)
        {
            cout << "policy iteration succeed in iteration " << i << ".\n"
                 << endl;
            iter.converged = true;
            cout << "optimal policy is (";
            for (size_t a : iter.policy)
            {
                cout << a << " ";
            }
            cout << ")\n";
            cout << "optimal state values are (";
            for (float s : iter.state_values)
            {
                cout << s << " ";
            }
            cout << ")\n";
            break;
        }
    }
    return iter;
}