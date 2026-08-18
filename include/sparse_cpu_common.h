#pragma once

#include <vector>
#include "mdp_csr.h"
#define EIGEN_NO_CUDA
#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace std;

using SpMat = Eigen::SparseMatrix<float>;
using Triplet = Eigen::Triplet<float>;

inline void generate_sparse_matrix_A_and_vector_R_cpu(const MDP &mdp, const vector<size_t> &policy, SpMat &A, Eigen::VectorXf &R, vector<Triplet> &triplets)
{
    size_t n = mdp.num_states;

    triplets.clear();
    R.setZero(n);

    for (size_t s = 0; s < n; s++)
    {
        triplets.emplace_back(s, s, 1.0f);

        size_t action = policy[s];
        size_t row = s * mdp.num_actions + action;
        size_t begin = mdp.row_ptr[row];
        size_t end = mdp.row_ptr[row + 1];

        for (size_t offset = begin; offset < end; offset++)
        {
            size_t next = mdp.next_state[offset];

            float p = mdp.prob[offset];
            float r = mdp.reward[offset];

            triplets.emplace_back(s, next, -mdp.gamma * p);
            R[s] += p * r;
        }
    }

    A.resize(n, n);
    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();
}
