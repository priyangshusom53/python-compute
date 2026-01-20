#ifndef CUDA_DEFINES
#define CUDA_DEFINES

#if defined(__CUDACC__)
#define CPU_GPU __host__ __device__
#else CPU_GPU
#endif

#endif