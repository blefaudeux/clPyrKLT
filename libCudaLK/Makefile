################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE	     := CudaLK

# Cuda source files (compiled with cudacc)
CUFILES		     := cudaLK.cu

# Library parameters
CUDA_INSTALL_PATH     := /usr/local/cuda-5.0

# Paths to linked libraries : -> To correct if include error
PATH_CUDA     ?= /usr/local/cuda/lib64
PATH_CUDA_SDK ?= ~/NVIDIA_GPU_Computing_SDK
PATH_CUTIL    ?= $(PATH_CUDA_SDK)/CUDALibraries/common/lib

# Includes :
INCLUDES += -I .
INCLUDES += `pkg-config opencv --cflags`
INCLUDES += -I $(CUDA_INSTALL_PATH)/include/ -I $(PATH_CUDA_SDK)/C/common/inc/

# Link flags :
LDLIBS    += `pkg-config opencv --libs`
LDLIBS    += -lcudart

LDFLAGS   += -L/usr/lib
LDFLAGS   += -L $(PATH_CUDA) -L $(PATH_CUTIL)
LDFLAGS   += -fPIC

LINKFLAGS += $(LDFLAGS) $(LDLIBS)

# Compiler flags
verbose   := 1
fastmath  := 1

CPPFLAGS += -O3 -m64 -Wall $(INCLUDES) -fPIC  -march=corei7-avx
CXXFLAGS += -fPIC $(INCLUDES)

# Show registers usage
CUDACCFLAGS += -Xcompiler -fpic --ptxas-options=-v -use_fast_math -shared

STATIC_LIB := libCudaLK.a

################################################################################
# Rules and targets

include ./common.mk
