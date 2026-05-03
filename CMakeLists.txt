cmake_minimum_required(VERSION 3.18)
project(CudaProject LANGUAGES CXX CUDA)

find_package(OpenMP REQUIRED)

add_executable(main src/main.cu)

target_include_directories(main PRIVATE include)

target_link_libraries(main PRIVATE OpenMP::OpenMP_CXX)

target_compile_options(main PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
)

set_target_properties(main PROPERTIES
    CUDA_ARCHITECTURES 86
)
