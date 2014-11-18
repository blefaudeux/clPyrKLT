CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(libcudaklt)

FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)

INCLUDE(FindCUDA)

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/headers)

INCLUDE_DIRECTORIES( ${CUDA_INCLUDE_PATH} ${OPENCV_INCLUDE_PATH} )

FILE(GLOB SOURCES "src/*.cu" "src/*.cpp" "src/*.c" "headers/*.h")

LIST(APPEND CMAKE_CXX_FLAGS "-O3 -ffast-math -Wall")

SET(CUDA_NVCC_FLAGS "-arch=sm_30" CACHE STRING "nvcc flags" FORCE)
SET(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)

LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)

# Build a static lib
CUDA_ADD_LIBRARY(CUDALK STATIC ${SOURCES})

