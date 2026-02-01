#ifndef CUDA_DEBUG_CUH
#define CUDA_DEBUG_CUH

#include <assert.h>

#ifdef DEBUG
#define CUDA_ASSERT(cond, msg)                          \
    do {                                                \
        if (!(cond)) {                                  \
            printf("ASSERT FAILED: %s\n", msg);         \
            asm("trap;");                               \
        }                                               \
    } while (0)
#else
#define CUDA_ASSERT(cond, message)
#endif

#endif