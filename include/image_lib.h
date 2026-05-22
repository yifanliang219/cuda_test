#pragma once

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <stdexcept>
#include <vector>
#include <string>

struct Image
{
    int height;
    int width;
    int channels;
    std::vector<unsigned char> data;
};

Image loadImageAsUnsignedCharVector(const std::string &filename)
{
    int width = 0;
    int height = 0;
    int originalChannels = 0;

    int desiredChannels = 3;

    unsigned char *pixels = stbi_load(
        filename.c_str(),
        &width,
        &height,
        &originalChannels,
        desiredChannels);

    if (pixels == nullptr)
    {
        throw std::runtime_error("Failed to load image: " + filename);
    }

    int channels = desiredChannels;
    int totalSize = width * height * channels;

    std::vector<unsigned char> imageData(pixels, pixels + totalSize);

    stbi_image_free(pixels);

    return Image{height, width, channels, std::move(imageData)};
}

void saveImageAsPng(const Image &img, const std::string &filename)
{
    int strideInBytes = img.width * img.channels;

    int success = stbi_write_png(
        filename.c_str(),
        img.width,
        img.height,
        img.channels,
        img.data.data(),
        strideInBytes);

    if (success == 0)
    {
        throw std::runtime_error("Failed to write PNG image: " + filename);
    }
}