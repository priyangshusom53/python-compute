#ifndef CUDA_DEFINES
#define CUDA_DEFINES

#if defined(__CUDACC__)
#define CPU_GPU __host__ __device__
#else 
#define CPU_GPU
#endif

#if defined(__CUDA_ARCH__)
#define GPU_CODE
#endif

#endif