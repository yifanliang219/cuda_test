#include <iostream>
#include <vector>
#include <random>
#include <stdexcept>
#include <string>
#include <cnpy.h>
#define EIGEN_NO_CUDA
#include <Eigen/Dense>
#include <iomanip>

using namespace std;
using namespace cnpy;

vector<float> generateMatrices(size_t num, size_t width, int seed)
{
    mt19937 rng(seed);
    uniform_real_distribution<float> dist(0.0f, 1.0f);

    vector<float> data(num * width * width);

    for (float &x : data)
    {
        x = dist(rng);
    }

    return data;
}

void saveMatrix(
    const string &path,
    const vector<float> &data,
    size_t num,
    size_t width)
{
    vector<size_t> shape = {num, width, width};
    npy_save(path, data.data(), shape, "w");
}

vector<float> loadMatrix(
    const string &filename,
    size_t &num,
    size_t &width)
{
    NpyArray arr = npy_load(filename);

    num = arr.shape[0];
    width = arr.shape[1];

    float *ptr = arr.data<float>();

    return vector<float>(ptr, ptr + arr.num_vals);
}

vector<float> eigenRefMatrixMul(const vector<float> &A, const vector<float> &B, size_t num, size_t width)
{
    vector<float> C(A.size());

    size_t matrix_size = width * width;

    using RowMajorMatrixXf =
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    for (size_t k = 0; k < num; k++)
    {
        const float *A_ptr = A.data() + k * matrix_size;
        const float *B_ptr = B.data() + k * matrix_size;
        float *C_ptr = C.data() + k * matrix_size;

        Eigen::Map<const RowMajorMatrixXf> A_mat(
            A_ptr,
            static_cast<int>(width),
            static_cast<int>(width));

        Eigen::Map<const RowMajorMatrixXf> B_mat(
            B_ptr,
            static_cast<int>(width),
            static_cast<int>(width));

        Eigen::Map<RowMajorMatrixXf> C_mat(
            C_ptr,
            static_cast<int>(width),
            static_cast<int>(width));

        C_mat.noalias() = A_mat * B_mat;
    }

    return C;
}

void printMatrix(const vector<float> &M, size_t width)
{
    ostringstream oss;
    oss << fixed << setprecision(3);
    for(int row=0;row<width;row++){
        for(int col=0;col<width;col++){
            oss << M[row*width+col] << ", ";
        }
        oss << '\n';
    }
    cout << oss.str();
}