#pragma once

#include "mdp_csr.h"
#include "policy_iter.h"

PolicyIteration policy_iter_FP_cpu(const MDP &mdp, float tolerance);
PolicyIteration policy_iter_gpu(const MDP &mdp, float tolerance);
PolicyIteration policy_iter_gpu_better(const MDP &mdp, float tolerance);
PolicyIteration policy_iter_matrix_dense_LU_cpu(const MDP &mdp);
PolicyIteration policy_iter_matrix_sparse_LU_cpu(const MDP &mdp);
PolicyIteration policy_iter_matrix_sparse_LU_no_pivot_cpu(const MDP &mdp);
PolicyIteration policy_iter_matrix_BiCGSTAB_cpu(const MDP &mdp, float tolerance);
PolicyIteration policy_iter_matrix_custom_sparse_LU_cpu(const MDP &mdp);
PolicyIteration policy_iter_gaussian_no_pivot_cpu(const MDP &mdp);
PolicyIteration policy_iter_matrix_dense_LU_gpu(const MDP &mdp);
PolicyIteration policy_iter_matrix_sparse_LU_gpu(const MDP &mdp);
PolicyIteration policy_iter_matrix_sparse_LU_no_pivot_gpu(const MDP &mdp);
PolicyIteration policy_iter_matrix_sparse_BiCGSTAB_gpu(const MDP &mdp, float tolerance);
