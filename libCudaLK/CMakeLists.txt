CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(libcudaklt)

FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(MPI REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)

INCLUDE(FindCUDA)

INCLUDE_DIRECTORIES(/usr/local/cuda/include ${MPI_INCLUDE_PATH})

FILE(GLOB SOURCES "*.cu" "*.cpp" "*.c" "*.h")

CUDA_ADD_LIBRARY(libcudalk ${LIB_TYPE} ${SOURCES}) #CUDA_ADD_EXECUTABLE(libcudalk ${SOURCES})

LIST(APPEND CMAKE_CXX_FLAGS "-std=c++0x -O3 -ffast-math -Wall")

SET(CUDA_NVCC_FLAGS "-arch=sm_30" CACHE STRING "nvcc flags" FORCE)
SET(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)

SET(LIB_TYPE STATIC) #set the lib type

LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)

#ADD_EXECUTABLE(demo ${SOURCES_DEMO})
#ADD_DEPENDENCIES(demo libcudalk)
#TARGET_LINK_LIBRARIES(demo libcudalk)
