#ifndef CUDA_DEBUG_CUH
#define CUDA_DEBUG_CUH

#include <assert.h>

#ifdef DEBUG
#define CUDA_ASSERT(cond, message) \
   assert(cond &&message)
#else
#define CUDA_ASSERT(cond, message)
#endif

#endif