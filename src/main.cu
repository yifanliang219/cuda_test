#include <stdio.h>
#include <stdlib.h>
#include <iomanip>
#include <sstream>
#include "cxtimers.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "thrust/device_vector.h"
#include "image_lib.h"
#include "matrix_lib.h"
#include "ch3_matrix.cuh"
#include "mdp_csr.h"
#include "policy_iter_fp.cuh"
#include "policy_iter_matrix.cuh"

using namespace std;

void loadMatrices(vector<float>& A, vector<float>& B, vector<float>& C)
{
	size_t num = 20;
	size_t width = 512;

	A = loadMatrix("data/A_f32_K20_N512.npy", num, width);
	B = loadMatrix("data/B_f32_K20_N512.npy", num, width);
	C = loadMatrix("data/C_f32_K20_N512.npy", num, width);
}

bool checkSamePolicy(const PolicyIteration& iter1, const PolicyIteration& iter2)
{
	for (size_t i = 0; i < iter1.state_values.size(); i++)
	{
		if (iter1.policy[i] != iter2.policy[i])
		{
			return false;
		}
	}
	return true;
}

bool checkSameValues(const PolicyIteration& iter1, const PolicyIteration& iter2)
{
	for (size_t i = 0; i < iter1.state_values.size(); i++)
	{
		if (fabs(iter1.state_values[i] - iter2.state_values[i]) > 0.0001f)
		{
			return false;
		}
	}
	return true;
}

double maxValueDiff(const PolicyIteration& iter1, const PolicyIteration& iter2)
{
	double worst = 0.0;
	for (size_t i = 0; i < iter1.state_values.size(); i++)
	{
		double diff = fabs(iter1.state_values[i] - iter2.state_values[i]);
		if (diff > worst)
		{
			worst = diff;
		}
	}
	return worst;
}

double maxRelativeValueDiff(const PolicyIteration& iter1, const PolicyIteration& iter2)
{
	const double epsilon = 1e-6;
	double worst = 0.0;
	for (size_t i = 0; i < iter1.state_values.size(); i++)
	{
		double denom = fmax(fabs(iter1.state_values[i]), epsilon);
		double ratio = fabs(iter1.state_values[i] - iter2.state_values[i]) / denom;
		if (ratio > worst)
		{
			worst = ratio;
		}
	}
	return worst;
}

#include <filesystem>

void generate_all_test_mdps()
{
	vector<size_t> num_states = { 4096, 40960, 409600 };
	vector<size_t> num_actions = { 4, 8 };

	for (size_t s : num_states)
	{
		for (size_t a : num_actions)
		{
			string state_action = to_string(s) + "_" + to_string(a);
			string folder = "data/" + state_action;

			std::filesystem::create_directories(folder);

			for (int i = 0; i < 10; i++)
			{
				vector<MDP> mdps = generate_random_MDPs(1, s, a, 4, 0.95f, 123 + i);

				string path = folder + "/" + state_action + "_" + to_string(i) + ".npz";

				save_mdp(mdps[0], path);

				cout << "Generated MDP " << i << ": " << path << endl;
			}
		}
	}
}

bool contains(const vector<string>& v, const string& s)
{
	return find(v.begin(), v.end(), s) != v.end();
}

vector<string> all_algs = {
	"CPU Fixed-Point",
	"CPU Sparse LU",
	"CPU Custom Sparse LU",
	"CPU Sparse BiCGSTAB",
	"GPU Fixed-Point",
	"GPU Sparse LU",
	"GPU Sparse BiCGSTAB" };

vector<string> all_fast_algs = {
	"CPU Fixed-Point",
	"CPU Sparse LU",
	"CPU Sparse BiCGSTAB",
	"GPU Fixed-Point",
	"GPU Sparse LU",
	"GPU Sparse BiCGSTAB" };

vector<string> LU_algs = {
	"CPU Sparse LU",
	"CPU Custom Sparse LU",
	"GPU Sparse LU" };

vector<string> library_LU_algs = {
	"GPU Fixed-Point",
	"CPU Sparse LU",
	"GPU Sparse LU" };

vector<string> gpu_LU_algs = {
	"GPU Fixed-Point",
	"GPU Sparse LU" };

void analysis(int argc, char* argv[], vector<string> algs)
{
	string num_states = "4096";
	string num_actions = "16";

	if (argc >= 2)
		num_states = argv[1];

	if (argc >= 3)
		num_actions = argv[2];

	int num_mdps = 10;

	if (argc >= 4)
		num_mdps = stoi(argv[3]);

	string state_action = num_states + "_" + num_actions;
	string folder = "data/" + state_action;

	cudaFree(0);
	cudaDeviceSynchronize();

	struct RunResultRow
	{
		string name;
		const PolicyIteration* iter;
		double time_ms;
	};

	vector<RunResultRow> run_results;

	struct AvgResultRow
	{
		string name;
		double total_time_ms = 0.0;
		size_t total_iterations = 0;
		int converged_count = 0;
		int same_policy_count = 0;
		int same_values_count = 0;
		double worst_max_value_diff = 0.0;
		double worst_max_relative_value_diff = 0.0;
	};

	vector<AvgResultRow> results;

	auto print_average_table = [&](int completed)
		{
			const int nameWidth = 26;
			const int numWidth = 16;
			const int countWidth = 16;

			const int totalWidth = nameWidth + numWidth + numWidth + countWidth + countWidth + countWidth + numWidth + numWidth;

			cout << "\nAverage results after " << completed << "/" << num_mdps << " MDPs\n";
			cout << string(totalWidth, '=') << "\n";

			cout << left << setw(nameWidth) << "Method"
				<< right << setw(numWidth) << "Avg Time (ms)"
				<< setw(numWidth) << "Avg Iter"
				<< setw(countWidth) << "Converged"
				<< setw(countWidth) << "Same Policy"
				<< setw(countWidth) << "Same Values"
				<< setw(numWidth) << "Worst MaxDV"
				<< setw(numWidth) << "Worst MaxDV%" << "\n";

			cout << string(totalWidth, '-') << "\n";

			cout << fixed << setprecision(3);

			auto formatNum = [](bool is_baseline, double value)
				{
					if (is_baseline)
						return string("-");
					ostringstream oss;
					oss << fixed << setprecision(6) << value;
					return oss.str();
				};

			auto formatPercent = [](bool is_baseline, double ratio)
				{
					if (is_baseline)
						return string("-");
					ostringstream oss;
					oss << fixed << setprecision(4) << (ratio * 100.0) << "%";
					return oss.str();
				};

			for (size_t j = 0; j < results.size(); j++)
			{
				const AvgResultRow& row = results[j];

				double avg_time = row.converged_count > 0 ? row.total_time_ms / row.converged_count : 0.0;
				double avg_iterations = row.converged_count > 0 ? static_cast<double>(row.total_iterations) / row.converged_count : 0.0;

				bool is_baseline = (j == 0);

				cout << left << setw(nameWidth) << row.name
					<< right << setw(numWidth) << avg_time
					<< setw(numWidth) << avg_iterations
					<< setw(countWidth) << (to_string(row.converged_count) + "/" + to_string(completed))
					<< setw(countWidth) << (is_baseline ? "-" : (to_string(row.same_policy_count) + "/" + to_string(completed)))
					<< setw(countWidth) << (is_baseline ? "-" : (to_string(row.same_values_count) + "/" + to_string(completed)))
					<< setw(numWidth) << formatNum(is_baseline, row.worst_max_value_diff)
					<< setw(numWidth) << formatPercent(is_baseline, row.worst_max_relative_value_diff)
					<< "\n";
			}

			cout << string(totalWidth, '=') << "\n"
				<< endl;
			cout.flush();
		};

	for (int i = 0; i < num_mdps; i++)
	{
		run_results.clear();

		string path = folder + "/" + state_action + "_" + to_string(i) + ".npz";

		cout << "\nRunning MDP " << i << ": " << path << endl;

		MDP loaded = load_mdp(path);

		cx::timer tim;

		PolicyIteration iter_cpu;
		PolicyIteration iter_matrix_sparse_LU_cpu;
		PolicyIteration iter_matrix_custom_sparse_LU_cpu;
		PolicyIteration iter_matrix_BiCGSTAB_cpu;
		PolicyIteration iter_FP_gpu;
		PolicyIteration iter_matrix_sparse_LU_gpu;
		PolicyIteration iter_BiCGSTAB_gpu;

		if (contains(algs, "CPU Fixed-Point"))
		{
			tim.reset();
			tim.start();
			iter_cpu = policy_iter_FP_cpu(loaded, 1e-6f);
			double cpu_FP_time = tim.lap_ms();
			run_results.push_back({ "CPU Fixed-Point", &iter_cpu, cpu_FP_time });
		}

		if (contains(algs, "CPU Sparse LU"))
		{
			tim.reset();
			tim.start();
			iter_matrix_sparse_LU_cpu = policy_iter_matrix_sparse_LU_cpu(loaded);
			double cpu_matrix_sparse_LU_time = tim.lap_ms();
			run_results.push_back({ "CPU Sparse LU", &iter_matrix_sparse_LU_cpu, cpu_matrix_sparse_LU_time });
		}

		if (contains(algs, "CPU Custom Sparse LU"))
		{
			tim.reset();
			tim.start();
			iter_matrix_custom_sparse_LU_cpu = policy_iter_matrix_custom_sparse_LU_cpu(loaded);
			double cpu_matrix_custom_sparse_LU_time = tim.lap_ms();
			run_results.push_back({ "CPU Custom Sparse LU", &iter_matrix_custom_sparse_LU_cpu, cpu_matrix_custom_sparse_LU_time });
		}

		if (contains(algs, "CPU Sparse BiCGSTAB"))
		{
			tim.reset();
			tim.start();
			iter_matrix_BiCGSTAB_cpu = policy_iter_matrix_BiCGSTAB_cpu(loaded, 1e-8f);
			double cpu_matrix_BiCGSTAB_time = tim.lap_ms();
			run_results.push_back({ "CPU Sparse BiCGSTAB", &iter_matrix_BiCGSTAB_cpu, cpu_matrix_BiCGSTAB_time });
		}

		if (contains(algs, "GPU Fixed-Point"))
		{
			cudaDeviceSynchronize();
			tim.reset();
			tim.start();
			iter_FP_gpu = policy_iter_gpu_better(loaded, 1e-6f);
			cudaDeviceSynchronize();
			double gpu_FP_time = tim.lap_ms();
			run_results.push_back({ "GPU Fixed-Point", &iter_FP_gpu, gpu_FP_time });
		}

		if (contains(algs, "GPU Sparse LU"))
		{
			size_t free_before, total_before;
			cudaMemGetInfo(&free_before, &total_before);

			cudaDeviceSynchronize();
			tim.reset();
			tim.start();
			iter_matrix_sparse_LU_gpu = policy_iter_matrix_sparse_LU_gpu(loaded);
			cudaDeviceSynchronize();
			double gpu_sparse_LU_time = tim.lap_ms();
			run_results.push_back({ "GPU Sparse LU", &iter_matrix_sparse_LU_gpu, gpu_sparse_LU_time });

			size_t free_after, total_after;
			cudaMemGetInfo(&free_after, &total_after);
			cout << "[DEBUG GPU Sparse LU] MDP " << i << ": time=" << gpu_sparse_LU_time
				<< "ms iters=" << iter_matrix_sparse_LU_gpu.num_iterations
				<< " converged=" << iter_matrix_sparse_LU_gpu.converged
				<< " free_before=" << (free_before / (1024 * 1024)) << "MB"
				<< " free_after=" << (free_after / (1024 * 1024)) << "MB" << endl;
		}

		if (contains(algs, "GPU Sparse BiCGSTAB"))
		{
			cudaDeviceSynchronize();
			tim.reset();
			tim.start();
			iter_BiCGSTAB_gpu = policy_iter_matrix_sparse_BiCGSTAB_gpu(loaded, 1e-6f);
			cudaDeviceSynchronize();
			double gpu_BiCGSTAB_time = tim.lap_ms();
			run_results.push_back({ "GPU Sparse BiCGSTAB", &iter_BiCGSTAB_gpu, gpu_BiCGSTAB_time });
		}

		if (results.empty())
		{
			for (const RunResultRow& row : run_results)
			{
				results.push_back({ row.name });
			}
		}

		if (!run_results.empty())
		{
			const PolicyIteration& baseline = *run_results[0].iter;

			for (size_t j = 0; j < run_results.size(); j++)
			{
				if (run_results[j].iter->converged)
				{
					results[j].total_time_ms += run_results[j].time_ms;
					results[j].total_iterations += run_results[j].iter->num_iterations;
					results[j].converged_count++;
				}

				if (j == 0)
				{
					continue;
				}

				if (checkSamePolicy(baseline, *run_results[j].iter))
				{
					results[j].same_policy_count++;
				}

				if (checkSameValues(baseline, *run_results[j].iter))
				{
					results[j].same_values_count++;
				}

				double max_diff = maxValueDiff(baseline, *run_results[j].iter);
				double max_relative_diff = maxRelativeValueDiff(baseline, *run_results[j].iter);
				cout << "  Max |dV| vs " << run_results[0].name << " for " << run_results[j].name
					<< ": " << max_diff << " (max |dV|/|baseline| = " << (max_relative_diff * 100.0) << "%)" << endl;

				if (max_diff > results[j].worst_max_value_diff)
				{
					results[j].worst_max_value_diff = max_diff;
				}

				if (max_relative_diff > results[j].worst_max_relative_value_diff)
				{
					results[j].worst_max_relative_value_diff = max_relative_diff;
				}
			}
		}

		cout << "Finished MDP " << i << endl;

		print_average_table(i + 1);
	}
}

int main(int argc, char* argv[])
{

	// vector<float> A, B, C;
	// loadMatrices(A, B, C);

	// const vector<float> A1(A.begin(), A.begin() + 512 * 512);
	// const vector<float> B1(B.begin(), B.begin() + 512 * 512);
	// const vector<float> C_cuda = singleMatrixMul(A1, B1, 512);
	// const vector<float> C_eigen = eigenRefMatrixMul(A1, B1, 1, 512);

	// vector<MDP> mdps = generate_random_MDPs(1, 4096, 64, 0.95f, 123);
	// save_mdp(mdps[0], "data/4096_64.npz");

	//generate_all_test_mdps();

	analysis(argc, argv, all_algs);

	return 0;
}