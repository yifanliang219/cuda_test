#pragma once

#include <vector>
#include "mdp_csr.h"

struct PolicyIteration
{
    vector<size_t> policy;
    vector<float> state_values;
    bool converged;
    size_t num_iterations;
};