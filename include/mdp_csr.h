#include <vector>
#include <random>
#include <algorithm>

using namespace std;

struct MDP
{
    size_t num_states;
    size_t num_actions;
    float gamma;

    vector<float> prob;
    vector<float> reward;
    vector<size_t> next_state;
    vector<size_t> row_ptr;
};

vector<MDP> generate_random_MDPs(size_t number, size_t states, size_t actions, float gamma, int seed)
{
    vector<MDP> mdps;
    mdps.reserve(number);
    mt19937 rng(seed);
    uniform_real_distribution<float> weight_dist(0.01f, 1.0f);
    uniform_real_distribution<float> reward_dist(-1.0f, 1.0f);
    size_t max_num_next_states = min(static_cast<size_t>(3), states);
    uniform_int_distribution<size_t> num_successors_dist(1, max_num_next_states);
    vector<size_t> state_list(states);
    size_t num_state_action_pairs = states * actions;
    iota(state_list.begin(), state_list.end(), 0);
    for (size_t i = 0; i < number; i++)
    {
        mdps.emplace_back();
        MDP &mdp = mdps.back();
        mdp.num_states = states;
        mdp.num_actions = actions;
        mdp.gamma = gamma;

        mdp.row_ptr.reserve(num_state_action_pairs + 1);
        mdp.row_ptr.push_back(0);
        vector<size_t> num_successors_list(num_state_action_pairs);
        size_t num_transitions = 0;
        for (size_t i = 0; i < num_state_action_pairs; i++)
        {
            size_t num_successors = num_successors_dist(rng);
            num_successors_list[i] = num_successors;
            num_transitions += num_successors;
        }
        mdp.prob.reserve(num_transitions);
        mdp.reward.reserve(num_transitions);
        mdp.next_state.reserve(num_transitions);
        for (size_t s = 0; s < states; s++)
        {
            for (size_t a = 0; a < actions; a++)
            {
                size_t row = s * actions + a;
                size_t num_successors = num_successors_list[row];
                size_t begin = mdp.row_ptr[row];
                mdp.row_ptr.push_back(begin + num_successors);
                shuffle(state_list.begin(), state_list.end(), rng);
                vector<float> weights(num_successors);
                float total_weights = 0;
                for (size_t i = 0; i < num_successors; i++)
                {
                    float weight = weight_dist(rng);
                    weights[i] = weight;
                    total_weights += weight;
                }
                for (size_t i = 0; i < num_successors; i++)
                {
                    mdp.prob.push_back(weights[i] / total_weights);
                    mdp.reward.push_back(reward_dist(rng));
                    mdp.next_state.push_back(state_list[i]);
                }
            }
        }
    }
    return mdps;
}
