cmake_minimum_required(VERSION 3.23)

set(CMAKE_VERBOSE_MAKEFILE ON)

project(mustard VERSION 1.0
        DESCRIPTION "MUlti-gpu Scheduling of TAsk gRaphs on the Device"
        LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(NVCC_VER_MIN 12.3)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build)

# CUDA
set(CMAKE_CUDA_ARCHITECTURES "80")
set(CUDA_ARCHITECTURES "80")
find_package(CUDAToolkit ${NVCC_VER_MIN} REQUIRED)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

enable_language(CUDA)

# find_package()
message("CMAKE_BINARY_DIR are now ${CMAKE_BINARY_DIR}")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_80,code=sm_80")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler")
message("CMAKE_CUDA_FLAGS are now ${CMAKE_CUDA_FLAGS}")
message("CUDA_NVCC_FLAGS are now ${CUDA_NVCC_FLAGS}")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_FORCE_INLINES -DVERBOSE")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda -use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

add_compile_options(-Wno-deprecated-declarations)

include_directories(SYSTEM PRIVATE "${CMAKE_SOURCE_DIR}/include")

add_subdirectory(baselines/cusolver_Mg)
add_subdirectory(baselines/starpu/lu)
add_subdirectory(mustard)