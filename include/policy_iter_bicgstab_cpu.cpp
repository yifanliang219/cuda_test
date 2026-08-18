#include <stdio.h>
#include <vector>
#include "mdp_csr.h"
#include "policy_iter.h"
#include "sparse_cpu_common.h"
#include "policy_improvement_cpu.h"

using namespace std;

void policy_eval_matrix_BiCGSTAB_cpu(const MDP &mdp, const vector<size_t> &policy, vector<float> &state_values, SpMat &A, Eigen::VectorXf &R, vector<Triplet> &triplets, Eigen::BiCGSTAB<SpMat, Eigen::DiagonalPreconditioner<float>> &solver)
{
    size_t n = mdp.num_states;

    generate_sparse_matrix_A_and_vector_R_cpu(mdp, policy, A, R, triplets);

    solver.compute(A);

    Eigen::Map<Eigen::VectorXf> V_eigen(state_values.data(), n);
    Eigen::VectorXf V = solver.solveWithGuess(R, V_eigen);

    V_eigen = V;
}

PolicyIteration policy_iter_matrix_BiCGSTAB_cpu(const MDP &mdp, float tolerance)
{
    size_t n = mdp.num_states;
    PolicyIteration iter = {vector<size_t>(n, 0), vector<float>(n, 0.0f), false, 0};
    SpMat A(n, n);
    Eigen::VectorXf R(n);
    vector<Triplet> triplets;
    triplets.reserve(mdp.num_states + mdp.prob.size());
    Eigen::BiCGSTAB<SpMat, Eigen::DiagonalPreconditioner<float>> solver;
    solver.setTolerance(tolerance);
    solver.setMaxIterations(10000);

    for (int i = 0; i < 50; i++)
    {
        iter.num_iterations++;
        cout << "policy iteration CPU BiCGSTAB loop " << iter.num_iterations << endl;
        policy_eval_matrix_BiCGSTAB_cpu(mdp, iter.policy, iter.state_values, A, R, triplets, solver);

        if (policy_improvement_cpu(mdp, iter.policy, iter.state_values))
        {
            iter.converged = true;
            cout << "policy iteration CPU BiCGSTAB completed successfully." << endl;
            break;
        }
    }
    return iter;
}
