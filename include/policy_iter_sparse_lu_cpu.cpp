#include <stdio.h>
#include <vector>
#include "mdp_csr.h"
#include "policy_iter.h"
#include "sparse_cpu_common.h"
#include "policy_improvement_cpu.h"

using namespace std;

void policy_eval_matrix_sparse_LU_cpu(const MDP &mdp, const vector<size_t> &policy, vector<float> &state_values, SpMat &A, Eigen::VectorXf &R, vector<Triplet> &triplets, Eigen::SparseLU<SpMat> &solver)
{
    size_t n = mdp.num_states;

    generate_sparse_matrix_A_and_vector_R_cpu(mdp, policy, A, R, triplets);

    solver.compute(A);

    Eigen::Map<Eigen::VectorXf> V_eigen(state_values.data(), n);
    V_eigen = solver.solve(R);
}

static PolicyIteration policy_iter_matrix_sparse_LU_cpu_impl(const MDP &mdp, float pivot_threshold)
{
    size_t n = mdp.num_states;
    PolicyIteration iter = {vector<size_t>(n, 0), vector<float>(n, 0.0f), false, 0};
    SpMat A(n, n);
    Eigen::VectorXf R(n);
    vector<Triplet> triplets;
    triplets.reserve(mdp.num_states + mdp.prob.size());
    Eigen::SparseLU<SpMat> solver;
    solver.setPivotThreshold(pivot_threshold);

    for (int i = 0; i < 50; i++)
    {
        iter.num_iterations++;
        cout << "policy iteration CPU Sparse LU loop " << iter.num_iterations << endl;
        policy_eval_matrix_sparse_LU_cpu(mdp, iter.policy, iter.state_values, A, R, triplets, solver);

        if (policy_improvement_cpu(mdp, iter.policy, iter.state_values))
        {
            iter.converged = true;
            cout << "policy iteration CPU Sparse LU completed successfully." << endl;
            break;
        }
    }
    return iter;
}

PolicyIteration policy_iter_matrix_sparse_LU_cpu(const MDP &mdp)
{
    return policy_iter_matrix_sparse_LU_cpu_impl(mdp, 1.0f);
}

PolicyIteration policy_iter_matrix_sparse_LU_no_pivot_cpu(const MDP &mdp)
{
    return policy_iter_matrix_sparse_LU_cpu_impl(mdp, 0.0f);
}
