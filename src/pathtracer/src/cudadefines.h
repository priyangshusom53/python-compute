#ifndef CUDA_DEFINES
#define CUDA_DEFINES

#if defined(__CUDACC__)
#define CPU_GPU __host__ __device__
#define CPU_ONLY __host__
#else 
#define CPU_GPU
#define CPU_ONLY
#endif

#if defined(__CUDA_ARCH__)
#define GPU_CODE
#endif

#endif