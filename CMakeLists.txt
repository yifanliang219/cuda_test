cmake_minimum_required(VERSION 3.18)

project(CudaProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
find_package(ZLIB REQUIRED)

add_library(cnpy STATIC
    ${CMAKE_SOURCE_DIR}/include/third_party/cnpy/cnpy.cpp
)

target_include_directories(cnpy PUBLIC
    ${CMAKE_SOURCE_DIR}/include/third_party/cnpy
)

target_link_libraries(cnpy PUBLIC
    ZLIB::ZLIB
)

add_executable(main
    src/main.cu
)

target_include_directories(main PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_include_directories(main SYSTEM PRIVATE
    ${CMAKE_SOURCE_DIR}/include/third_party/stb_image
)

target_link_libraries(main PRIVATE
    OpenMP::OpenMP_CXX
    cnpy
)

target_compile_options(main PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
)

set_target_properties(main PROPERTIES
    CUDA_ARCHITECTURES 86
)