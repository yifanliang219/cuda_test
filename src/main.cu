#include <stdio.h>
#include <stdlib.h>
#include "cxtimers.h"
#include "cuda_runtime.h"
#include "thrust/device_vector.h"
#include "image_lib.h"
#include "chp3_gray.cuh"

using namespace std;

int main(int argc, char *argv[])
{
    string path = "data/pikachu.png";
    Image img_in = loadImageAsUnsignedCharVector(path);
    printf("Input image: height %d, width: %d, channels: %d, size: %d\n", img_in.height, img_in.width, img_in.channels, (int)img_in.data.capacity());

    Image img_same = Image{img_in.height, img_in.width, img_in.channels, img_in.data};
    saveImageAsPng(img_same, "data/pikachu_same.png");
    printf("Same image: height %d, width: %d, channels: %d, size: %d\n", img_same.height, img_same.width, img_same.channels, (int)img_same.data.capacity());

    vector<unsigned char> gray = colorToGray(img_in.data, img_in.height, img_in.width);
    Image img_out = Image{img_in.height, img_in.width, 1, move(gray)};
    printf("Output image: height %d, width: %d, channels: %d, size: %d\n", img_out.height, img_out.width, img_out.channels, (int)img_out.data.capacity());
    saveImageAsPng(img_out, "data/pikachu_gray.png");
    return 0;
}