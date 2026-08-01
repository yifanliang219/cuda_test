#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include "mdp_csr.h"
#include "cnpy.h"

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

    size_t max_num_next_states = std::min<size_t>(3, states);
    uniform_int_distribution<size_t> num_successors_dist(1, max_num_next_states);
    uniform_int_distribution<size_t> state_dist(0, states - 1);

    size_t num_state_action_pairs = states * actions;

    for (size_t mdp_index = 0; mdp_index < number; mdp_index++)
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

        for (size_t row = 0; row < num_state_action_pairs; row++)
        {
            size_t num_successors = num_successors_dist(rng);
            num_successors_list[row] = num_successors;
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

                array<size_t, 3> sampled_states;

                for (size_t i = 0; i < num_successors; i++)
                {
                    sampled_states[i] = state_dist(rng);
                }

                array<float, 3> weights;
                float total_weight = 0.0f;

                for (size_t i = 0; i < num_successors; i++)
                {
                    float weight = weight_dist(rng);
                    weights[i] = weight;
                    total_weight += weight;
                }

                for (size_t i = 0; i < num_successors; i++)
                {
                    mdp.prob.push_back(weights[i] / total_weight);
                    mdp.reward.push_back(reward_dist(rng));
                    mdp.next_state.push_back(sampled_states[i]);
                }
            }
        }
    }

    return mdps;
}

void print_MDP(const MDP &mdp)
{
    cout << "MDP information\n";
    cout << "---------------\n";
    cout << "num_states  = " << mdp.num_states << '\n';
    cout << "num_actions = " << mdp.num_actions << '\n';
    cout << "gamma       = " << mdp.gamma << '\n';
    cout << "num_rows    = " << mdp.num_states * mdp.num_actions << '\n';
    cout << "num_edges   = " << mdp.prob.size() << '\n';
    cout << '\n';

    cout << fixed << setprecision(4);

    for (size_t s = 0; s < mdp.num_states; ++s)
    {
        for (size_t a = 0; a < mdp.num_actions; ++a)
        {
            size_t row = s * mdp.num_actions + a;

            cout << "State " << s << ", Action " << a << ":\n";

            size_t begin = mdp.row_ptr[row];
            size_t end = mdp.row_ptr[row + 1];

            float prob_sum = 0.0f;

            for (size_t k = begin; k < end; ++k)
            {
                prob_sum += mdp.prob[k];

                cout << "    -> next_state = " << mdp.next_state[k]
                     << ", prob = " << mdp.prob[k]
                     << ", reward = " << mdp.reward[k]
                     << '\n';
            }

            cout << "    probability sum = " << prob_sum << "\n\n";
        }
    }
}

void save_mdp(const MDP &mdp, const string &filename)
{
    vector<uint64_t> metadata = {
        static_cast<uint64_t>(mdp.num_states),
        static_cast<uint64_t>(mdp.num_actions),
        static_cast<uint64_t>(mdp.prob.size()),
        static_cast<uint64_t>(mdp.row_ptr.size())};

    vector<uint64_t> next_state_64(mdp.next_state.begin(), mdp.next_state.end());
    vector<uint64_t> row_ptr_64(mdp.row_ptr.begin(), mdp.row_ptr.end());

    cnpy::npz_save(filename, "metadata", metadata.data(), {metadata.size()}, "w");
    cnpy::npz_save(filename, "gamma", &mdp.gamma, {1}, "a");
    cnpy::npz_save(filename, "prob", mdp.prob.data(), {mdp.prob.size()}, "a");
    cnpy::npz_save(filename, "reward", mdp.reward.data(), {mdp.reward.size()}, "a");
    cnpy::npz_save(filename, "next_state", next_state_64.data(), {next_state_64.size()}, "a");
    cnpy::npz_save(filename, "row_ptr", row_ptr_64.data(), {row_ptr_64.size()}, "a");
}

MDP load_mdp(const string &filename)
{
    MDP mdp;

    cnpy::NpyArray metadata_arr = cnpy::npz_load(filename, "metadata");
    cnpy::NpyArray gamma_arr = cnpy::npz_load(filename, "gamma");
    cnpy::NpyArray prob_arr = cnpy::npz_load(filename, "prob");
    cnpy::NpyArray reward_arr = cnpy::npz_load(filename, "reward");
    cnpy::NpyArray next_state_arr = cnpy::npz_load(filename, "next_state");
    cnpy::NpyArray row_ptr_arr = cnpy::npz_load(filename, "row_ptr");

    uint64_t *metadata = metadata_arr.data<uint64_t>();
    float *gamma = gamma_arr.data<float>();
    float *prob = prob_arr.data<float>();
    float *reward = reward_arr.data<float>();
    uint64_t *next_state = next_state_arr.data<uint64_t>();
    uint64_t *row_ptr = row_ptr_arr.data<uint64_t>();

    mdp.num_states = static_cast<size_t>(metadata[0]);
    mdp.num_actions = static_cast<size_t>(metadata[1]);
    mdp.gamma = gamma[0];

    size_t num_transitions = static_cast<size_t>(metadata[2]);
    size_t row_ptr_size = static_cast<size_t>(metadata[3]);

    mdp.prob.assign(prob, prob + num_transitions);
    mdp.reward.assign(reward, reward + num_transitions);

    mdp.next_state.resize(num_transitions);
    for (size_t i = 0; i < num_transitions; i++)
    {
        mdp.next_state[i] = static_cast<size_t>(next_state[i]);
    }

    mdp.row_ptr.resize(row_ptr_size);
    for (size_t i = 0; i < row_ptr_size; i++)
    {
        mdp.row_ptr[i] = static_cast<size_t>(row_ptr[i]);
    }

    if (mdp.prob.size() != mdp.reward.size() ||
        mdp.prob.size() != mdp.next_state.size())
    {
        throw std::runtime_error("Invalid loaded MDP: transition array sizes differ.");
    }

    if (mdp.row_ptr.size() != mdp.num_states * mdp.num_actions + 1)
    {
        throw std::runtime_error("Invalid loaded MDP: row_ptr size is incorrect.");
    }

    if (!mdp.row_ptr.empty() && mdp.row_ptr.back() != mdp.prob.size())
    {
        throw std::runtime_error("Invalid loaded MDP: row_ptr.back() does not match transition count.");
    }

    return mdp;
}