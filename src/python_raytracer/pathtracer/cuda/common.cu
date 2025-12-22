
#include "folder_1/include_1.cu"
#include "common1.cu"

__device__ float calc_half(float x)
{
   return x / 2; // Performs a simple operation on the GPU
}