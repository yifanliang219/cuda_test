cmake_minimum_required(VERSION 3.19)

project(CudaProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>

        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    )
endif()

find_package(ZLIB REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDAToolkit REQUIRED)

find_package(cudss 0.8.0 REQUIRED)

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

if(WIN32)
    target_include_directories(main PRIVATE
        ${cudss_INCLUDE_DIR}
    )

    target_link_directories(main PRIVATE
        ${cudss_LIBRARY_DIR}
    )
endif()

target_link_libraries(main PRIVATE
    cnpy
    Eigen3::Eigen
    CUDA::cudart
    CUDA::cusolver
    CUDA::cublas
    cudss
)

set_target_properties(main PROPERTIES
    CUDA_ARCHITECTURES 89
)