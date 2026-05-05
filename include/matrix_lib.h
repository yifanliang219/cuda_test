#include <iostream>
#include <vector>
#include <random>
#include <stdexcept>
#include <string>
#include <cnpy.h>

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