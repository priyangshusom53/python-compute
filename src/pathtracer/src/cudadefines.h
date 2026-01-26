#ifndef CUDA_DEFINES
#define CUDA_DEFINES

#if defined(__CUDACC__)
#define CPU_GPU __host__ __device__
#define CPU_ONLY __host__
#define GPU_ONLY __device__

#define CUDA_INLINE __inline__

#define CUDA_ALIGN(n) __align__(n)
#else 
#define CPU_GPU
#define CPU_ONLY
#define GPU_ONLY

#define CUDA_INLINE 
#define CUDA_ALIGN(n)
#endif

#if defined(__CUDA_ARCH__)
#define GPU_CODE
#endif

#endif