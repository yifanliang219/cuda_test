cmake_minimum_required(VERSION 3.18)

project(CudaProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(ZLIB REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_path(CUDSS_INCLUDE_DIR NAMES cudss.h PATHS /usr/include/libcudss/13)
find_library(CUDSS_LIBRARY NAMES cudss PATHS /usr/lib/x86_64-linux-gnu/libcudss/13)
if(NOT CUDSS_INCLUDE_DIR OR NOT CUDSS_LIBRARY)
    message(FATAL_ERROR "cuDSS not found (looked for cudss.h and libcudss.so under libcudss/13)")
endif()
add_library(cudss SHARED IMPORTED)
set_target_properties(cudss PROPERTIES
    IMPORTED_LOCATION ${CUDSS_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${CUDSS_INCLUDE_DIR}
)

get_target_property(EIGEN_INCLUDE_DIRS Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)

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
    ${EIGEN_INCLUDE_DIRS}
)

target_link_libraries(main PRIVATE
    cnpy
    Eigen3::Eigen
    CUDA::cudart
    CUDA::cusolver
    CUDA::cublas
    cudss
)

set_target_properties(main PROPERTIES
    CUDA_ARCHITECTURES 86
)