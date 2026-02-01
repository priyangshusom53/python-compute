#ifndef MATH_CONSTANTS
#define MATH_CONSTANTS

#include<limits>

#include "cudadefines.h"
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda/std/limits>
#endif

CPU_GPU INLINE float infinity_f() {
#ifdef __CUDA_ARCH__
    return __int_as_float(0x7f800000);
#else
    return std::numeric_limits<float>::infinity();
#endif
}

CPU_GPU INLINE float max_f(){
#ifdef __CUDA_ARCH__
    return ::cuda::std::numeric_limits<float>::max();
#else
    return std::numeric_limits<float>::max();
#endif
}

CPU_GPU INLINE float lowest_f(){
#ifdef __CUDA_ARCH__
    return ::cuda::std::numeric_limits<float>::lowest();
#else
    return std::numeric_limits<float>::lowest();
#endif
}

#endif